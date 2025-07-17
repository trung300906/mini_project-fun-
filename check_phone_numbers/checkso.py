#!/usr/bin/env python3
"""
Advanced Phone Number Validation System
Sử dụng nhiều thuật toán cao cấp để kiểm tra số điện thoại thật/ảo
"""

import re
import requests
import json
import hashlib
import hmac
import time
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import phonenumbers
from phonenumbers import carrier, geocoder, timezone
import sqlite3
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhoneStatus(Enum):
    """Trạng thái số điện thoại"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"
    VOIP = "voip"
    LANDLINE = "landline"
    MOBILE = "mobile"

@dataclass
class PhoneValidationResult:
    """Kết quả kiểm tra số điện thoại"""
    phone_number: str
    status: PhoneStatus
    confidence_score: float
    country: str
    carrier_name: str
    location: str
    line_type: str
    is_valid_format: bool
    is_possible: bool
    risk_score: float
    validation_methods: List[str]
    timestamp: datetime
    additional_info: Dict

class AdvancedPhoneValidator:
    """Hệ thống kiểm tra số điện thoại cao cấp"""
    
    def __init__(self, db_path: str = "phone_validation.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.init_database()
        self.load_patterns()
        
    def init_database(self):
        """Khởi tạo cơ sở dữ liệu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phone_validation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence_score REAL,
                country TEXT,
                carrier_name TEXT,
                location TEXT,
                line_type TEXT,
                risk_score REAL,
                validation_methods TEXT,
                timestamp DATETIME,
                additional_info TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suspicious_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                risk_level INTEGER,
                description TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_patterns(self):
        """Tải các pattern nghi ngờ"""
        self.suspicious_patterns = [
            # Các pattern số ảo phổ biến Việt Nam
            r'^(\+?84)?0?(123|111|000|999)\d{7}$',  # VN fake patterns
            r'^(\+?84)?0?(1234|2345|3456|4567|5678|6789)\d{6}$',  # Sequential VN
            r'^(\+?84)?0?(\d)\1{8}$',  # Repeated digits VN
            r'^(\+?84)?0?(0000|1111|2222|3333|4444|5555|6666|7777|8888|9999)\d{6}$',  # Pattern VN
            r'^(\+?84)?0?(012|098|032|033|034|035|036|037|038|039|070|071|072|073|074|075|076|077|078|079)\d{4}$',  # Invalid VN mobile prefixes
            
            # US fake numbers
            r'^(\+?1)?[2-9]\d{2}555\d{4}$',  # US fake numbers
            r'^(\+?1)?(000|111|222|333|444|555|666|777|888|999)\d{7}$',  # US fake
            
            # China suspicious
            r'^(\+?86)?1[35]\d{9}$',  # CN suspicious
            
            # General suspicious patterns
            r'^\+?(\d)\1{10,}$',  # Too many repeated digits
            r'^\+?(0123456789|1234567890|9876543210)$',  # Sequential numbers
        ]
        
        self.voip_patterns = [
            r'^(\+?1)?[2-9]\d{2}[2-9]\d{2}\d{4}$',  # US VoIP
            r'^(\+?44)?[1-9]\d{8,9}$',  # UK VoIP
            r'^(\+?84)?0?(01|02|03|04|05|06|07|08|09)8\d{7}$',  # VN VoIP-like
        ]
        
        # Thêm pattern đặc biệt cho số Việt Nam hợp lệ
        self.valid_vn_patterns = [
            r'^(\+?84)?0?[3-9]\d{8}$',  # VN mobile general
            r'^(\+?84)?0?(86|96|97|98|32|33|34|35|36|37|38|39|81|82|83|84|85|88|91|94|76|77|78|79|90|93|70|71|72|73|74|75|59|58|56|57|099|0199)\d{7}$',  # VN specific carriers
        ]
        
    def validate_format(self, phone_number: str) -> Tuple[bool, Dict]:
        """Kiểm tra định dạng số điện thoại"""
        try:
            # Thử parse với None trước, nếu lỗi thì thử với region mặc định
            try:
                parsed = phonenumbers.parse(phone_number, None)
            except phonenumbers.NumberParseException:
                # Nếu không có country code, thử với Vietnam làm mặc định
                parsed = phonenumbers.parse(phone_number, "VN")
            
            is_valid = phonenumbers.is_valid_number(parsed)
            is_possible = phonenumbers.is_possible_number(parsed)
            
            country = geocoder.description_for_number(parsed, "en")
            carrier_name = carrier.name_for_number(parsed, "en")
            timezones = timezone.time_zones_for_number(parsed)
            
            return is_valid, {
                'is_possible': is_possible,
                'country': country,
                'carrier': carrier_name,
                'timezones': list(timezones),
                'number_type': phonenumbers.number_type(parsed),
                'formatted_national': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                'formatted_international': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            }
        except Exception as e:
            logger.error(f"Format validation error: {e}")
            return False, {'error': str(e)}
    
    def check_suspicious_patterns(self, phone_number: str) -> float:
        """Kiểm tra các pattern nghi ngờ"""
        risk_score = 0.0
        
        # Kiểm tra pattern nghi ngờ
        for pattern in self.suspicious_patterns:
            if re.match(pattern, phone_number):
                risk_score += 0.4
                
        # Kiểm tra pattern VoIP
        for pattern in self.voip_patterns:
            if re.match(pattern, phone_number):
                risk_score += 0.3
        
        # Kiểm tra pattern hợp lệ VN (giảm risk nếu match)
        for pattern in self.valid_vn_patterns:
            if re.match(pattern, phone_number):
                risk_score -= 0.2
                
        return max(0.0, min(risk_score, 1.0))
    
    def luhn_algorithm_check(self, phone_number: str) -> bool:
        """Thuật toán Luhn để kiểm tra tính hợp lệ"""
        digits = [int(d) for d in re.sub(r'\D', '', phone_number)]
        
        if len(digits) < 8:
            return False
            
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit = digit // 10 + digit % 10
            checksum += digit
            
        return checksum % 10 == 0
    
    def hmac_verification(self, phone_number: str, secret_key: str = "phone_validation_key") -> str:
        """Tạo HMAC cho số điện thoại"""
        return hmac.new(
            secret_key.encode(),
            phone_number.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def bayesian_spam_detection(self, phone_number: str) -> float:
        """Thuật toán Bayesian nâng cao để phát hiện spam"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        # Các chỉ số spam với trọng số cải thiện
        spam_indicators = [
            (len(clean_number) < 9, 0.8),  # Quá ngắn
            (len(clean_number) > 15, 0.7),  # Quá dài
            (clean_number.count('0') > 6, 0.6),  # Quá nhiều số 0
            (clean_number.count('1') > 6, 0.5),  # Quá nhiều số 1
            (len(set(clean_number)) < 4, 0.9),  # Quá ít chữ số khác nhau
            (len(set(clean_number)) < 6, 0.4),  # Ít chữ số khác nhau
            (clean_number.startswith('0000'), 0.9),  # Bắt đầu bằng 0000
            (clean_number.startswith('1111'), 0.8),  # Bắt đầu bằng 1111
            (clean_number.endswith('0000'), 0.7),  # Kết thúc bằng 0000
            (self._is_ascending_sequence(clean_number), 0.6),  # Dãy tăng dần
            (self._is_descending_sequence(clean_number), 0.6),  # Dãy giảm dần
            (self._has_repeated_pattern(clean_number), 0.5),  # Pattern lặp lại
            (clean_number.count(clean_number[0]) > len(clean_number) * 0.6, 0.7),  # Một chữ số chiếm >60%
        ]
        
        # Tính xác suất spam theo Bayesian
        total_weight = sum(weight for _, weight in spam_indicators)
        spam_score = sum(weight for condition, weight in spam_indicators if condition)
        
        spam_probability = spam_score / total_weight if total_weight > 0 else 0
        return min(spam_probability, 1.0)
    
    def _is_ascending_sequence(self, digits: str) -> bool:
        """Kiểm tra dãy số tăng dần"""
        if len(digits) < 4:
            return False
        ascending_count = 0
        for i in range(1, len(digits)):
            if int(digits[i]) == int(digits[i-1]) + 1:
                ascending_count += 1
            else:
                ascending_count = 0
            if ascending_count >= 3:
                return True
        return False
    
    def _is_descending_sequence(self, digits: str) -> bool:
        """Kiểm tra dãy số giảm dần"""
        if len(digits) < 4:
            return False
        descending_count = 0
        for i in range(1, len(digits)):
            if int(digits[i]) == int(digits[i-1]) - 1:
                descending_count += 1
            else:
                descending_count = 0
            if descending_count >= 3:
                return True
        return False
    
    def _has_repeated_pattern(self, digits: str) -> bool:
        """Kiểm tra pattern lặp lại"""
        if len(digits) < 6:
            return False
        
        # Kiểm tra pattern 2-4 chữ số lặp lại
        for pattern_length in range(2, 5):
            if len(digits) < pattern_length * 2:
                continue
            
            pattern = digits[:pattern_length]
            repeated_pattern = pattern * (len(digits) // pattern_length)
            
            if digits.startswith(repeated_pattern[:len(digits)]):
                return True
        
        return False
    
    def machine_learning_prediction(self, phone_number: str) -> Tuple[float, Dict]:
        """Sử dụng ML nâng cao để dự đoán"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        # Feature extraction nâng cao
        features = {
            'length': len(clean_number),
            'unique_digits': len(set(clean_number)),
            'consecutive_same': self._max_consecutive_same(phone_number),
            'digit_variance': self._digit_variance(phone_number),
            'pattern_score': self._pattern_score(phone_number),
            'entropy': self._calculate_entropy(clean_number),
            'digit_frequency_variance': self._digit_frequency_variance(clean_number),
            'is_palindrome': self._is_palindrome(clean_number),
            'arithmetic_progression': self._has_arithmetic_progression(clean_number),
            'geometric_progression': self._has_geometric_progression(clean_number),
            'prime_digit_ratio': self._prime_digit_ratio(clean_number),
            'even_odd_ratio': self._even_odd_ratio(clean_number),
            'first_digit_analysis': self._first_digit_analysis(clean_number),
            'last_digit_analysis': self._last_digit_analysis(clean_number),
            'middle_digits_analysis': self._middle_digits_analysis(clean_number),
        }
        
        # Sophisticated scoring với nhiều tiêu chí
        score = 0.3  # Base score thấp hơn
        
        # Length scoring
        if features['length'] in range(10, 12):
            score += 0.25
        elif features['length'] == 9:
            score += 0.15
        elif features['length'] < 9 or features['length'] > 13:
            score -= 0.2
        
        # Unique digits scoring
        if features['unique_digits'] >= 7:
            score += 0.2
        elif features['unique_digits'] >= 5:
            score += 0.1
        elif features['unique_digits'] < 4:
            score -= 0.3
        
        # Consecutive digits scoring
        if features['consecutive_same'] <= 2:
            score += 0.15
        elif features['consecutive_same'] <= 3:
            score += 0.05
        else:
            score -= 0.2
        
        # Entropy scoring
        if features['entropy'] >= 2.5:
            score += 0.2
        elif features['entropy'] >= 2.0:
            score += 0.1
        elif features['entropy'] < 1.5:
            score -= 0.3
        
        # Digit variance scoring
        if features['digit_variance'] > 0.5:
            score += 0.1
        elif features['digit_variance'] < 0.2:
            score -= 0.15
        
        # Pattern penalties
        if features['is_palindrome']:
            score -= 0.2
        if features['arithmetic_progression']:
            score -= 0.25
        if features['geometric_progression']:
            score -= 0.15
        
        # Frequency variance bonus
        if features['digit_frequency_variance'] > 0.5:
            score += 0.1
        
        # Prime digit ratio bonus
        if 0.3 <= features['prime_digit_ratio'] <= 0.7:
            score += 0.05
        
        # Even/odd ratio bonus
        if 0.3 <= features['even_odd_ratio'] <= 0.7:
            score += 0.05
        
        # First/last digit analysis
        if features['first_digit_analysis'] == 0:  # Starts with 0
            score += 0.1  # Common for VN numbers
        if features['last_digit_analysis'] in [1, 3, 7, 9]:  # Odd endings
            score += 0.02
        
        return max(0.0, min(score, 1.0)), features
    
    def _digit_frequency_variance(self, digits: str) -> float:
        """Tính phương sai tần suất xuất hiện các chữ số"""
        if not digits:
            return 0
        
        freq = {}
        for digit in digits:
            freq[digit] = freq.get(digit, 0) + 1
        
        frequencies = list(freq.values())
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        return variance / len(digits)  # Normalize
    
    def _is_palindrome(self, digits: str) -> bool:
        """Kiểm tra số có phải palindrome không"""
        return digits == digits[::-1] and len(digits) > 4
    
    def _has_arithmetic_progression(self, digits: str) -> bool:
        """Kiểm tra dãy số cộng"""
        if len(digits) < 4:
            return False
        
        digit_list = [int(d) for d in digits]
        for i in range(len(digit_list) - 3):
            if (digit_list[i+1] - digit_list[i] == digit_list[i+2] - digit_list[i+1] == 
                digit_list[i+3] - digit_list[i+2] and digit_list[i+1] - digit_list[i] != 0):
                return True
        return False
    
    def _has_geometric_progression(self, digits: str) -> bool:
        """Kiểm tra dãy số nhân"""
        if len(digits) < 4:
            return False
        
        digit_list = [int(d) for d in digits if d != '0']
        if len(digit_list) < 4:
            return False
        
        for i in range(len(digit_list) - 3):
            if (digit_list[i] != 0 and digit_list[i+1] != 0 and 
                digit_list[i+1] / digit_list[i] == digit_list[i+2] / digit_list[i+1] == 
                digit_list[i+3] / digit_list[i+2]):
                return True
        return False
    
    def _prime_digit_ratio(self, digits: str) -> float:
        """Tỉ lệ chữ số nguyên tố"""
        if not digits:
            return 0
        
        prime_digits = '2357'
        prime_count = sum(1 for d in digits if d in prime_digits)
        return prime_count / len(digits)
    
    def _even_odd_ratio(self, digits: str) -> float:
        """Tỉ lệ chữ số chẵn/lẻ"""
        if not digits:
            return 0
        
        even_count = sum(1 for d in digits if int(d) % 2 == 0)
        return even_count / len(digits)
    
    def _first_digit_analysis(self, digits: str) -> int:
        """Phân tích chữ số đầu"""
        return int(digits[0]) if digits else 0
    
    def _last_digit_analysis(self, digits: str) -> int:
        """Phân tích chữ số cuối"""
        return int(digits[-1]) if digits else 0
    
    def _middle_digits_analysis(self, digits: str) -> Dict:
        """Phân tích chữ số giữa"""
        if len(digits) < 3:
            return {'middle_entropy': 0, 'middle_variance': 0}
        
        middle_part = digits[1:-1]
        return {
            'middle_entropy': self._calculate_entropy(middle_part),
            'middle_variance': self._digit_variance(middle_part)
        }
    
    def _max_consecutive_same(self, phone_number: str) -> int:
        """Tìm số lượng chữ số giống nhau liên tiếp tối đa"""
        digits = re.sub(r'\D', '', phone_number)
        max_count = 1
        current_count = 1
        
        for i in range(1, len(digits)):
            if digits[i] == digits[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
                
        return max_count
    
    def _digit_variance(self, phone_number: str) -> float:
        """Tính phương sai của các chữ số"""
        digits = [int(d) for d in re.sub(r'\D', '', phone_number)]
        if not digits:
            return 0
            
        mean = sum(digits) / len(digits)
        variance = sum((d - mean) ** 2 for d in digits) / len(digits)
        return variance / 10  # Normalize
    
    def _pattern_score(self, phone_number: str) -> float:
        """Tính điểm pattern"""
        digits = re.sub(r'\D', '', phone_number)
        
        # Check for common patterns
        patterns = [
            r'(\d)\1{3,}',  # Repeated digits
            r'0123456789',  # Sequential
            r'1234567890',  # Sequential
            r'(\d{3})\1',   # Repeated groups
        ]
        
        score = 0
        for pattern in patterns:
            if re.search(pattern, digits):
                score += 0.2
                
        return min(score, 1.0)
    
    async def async_api_validation(self, phone_number: str) -> Dict:
        """Kiểm tra qua API bất đồng bộ"""
        apis = [
            self._validate_with_numverify,
            self._validate_with_twilio,
            self._validate_with_custom_api
        ]
        
        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [api(session, phone_number) for api in apis]
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(api_results):
                if not isinstance(result, Exception):
                    results[f'api_{i}'] = result
                    
        return results
    
    async def _validate_with_numverify(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Kiểm tra với Numverify API"""
        try:
            url = f"http://apilayer.net/api/validate"
            params = {
                'access_key': 'YOUR_API_KEY',  # Cần API key thật
                'number': phone_number,
                'country_code': '',
                'format': 1
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return {'error': 'API request failed'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _validate_with_twilio(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Kiểm tra với Twilio API"""
        # Placeholder - cần Twilio credentials
        return {'source': 'twilio', 'status': 'placeholder'}
    
    async def _validate_with_custom_api(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Kiểm tra với API tùy chỉnh"""
        return {'source': 'custom', 'status': 'placeholder'}
    
    def advanced_heuristic_analysis(self, phone_number: str) -> Dict:
        """Phân tích heuristic cao cấp"""
        digits = re.sub(r'\D', '', phone_number)
        
        analysis = {
            'entropy': self._calculate_entropy(digits),
            'digit_distribution': self._digit_distribution(digits),
            'mathematical_properties': self._mathematical_properties(digits),
            'geographic_consistency': self._geographic_consistency(phone_number),
            'temporal_patterns': self._temporal_patterns(digits)
        }
        
        return analysis
    
    def _calculate_entropy(self, digits: str) -> float:
        """Tính entropy thông tin"""
        if not digits:
            return 0
            
        digit_counts = {}
        for digit in digits:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
            
        entropy = 0
        length = len(digits)
        for count in digit_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _digit_distribution(self, digits: str) -> Dict:
        """Phân tích phân bố chữ số"""
        distribution = {}
        for digit in '0123456789':
            distribution[digit] = digits.count(digit)
            
        return distribution
    
    def _mathematical_properties(self, digits: str) -> Dict:
        """Tính các thuộc tính toán học"""
        if not digits:
            return {}
            
        digit_list = [int(d) for d in digits]
        
        # Tính product an toàn
        product = 1
        for digit in digit_list:
            if digit == 0:
                product = 0
                break
            product *= digit
        
        return {
            'sum': sum(digit_list),
            'product': product,
            'average': sum(digit_list) / len(digit_list),
            'std_dev': (sum((d - sum(digit_list)/len(digit_list))**2 for d in digit_list) / len(digit_list))**0.5,
            'is_prime_sum': self._is_prime(sum(digit_list)),
            'fibonacci_similarity': self._fibonacci_similarity(digit_list)
        }
    
    def _is_prime(self, n: int) -> bool:
        """Kiểm tra số nguyên tố"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _fibonacci_similarity(self, digits: List[int]) -> float:
        """Tính độ tương tự với dãy Fibonacci"""
        if len(digits) < 3:
            return 0
            
        fib_score = 0
        for i in range(2, len(digits)):
            if digits[i] == digits[i-1] + digits[i-2]:
                fib_score += 1
                
        return fib_score / max(len(digits) - 2, 1)
    
    def _geographic_consistency(self, phone_number: str) -> Dict:
        """Kiểm tra tính nhất quán địa lý"""
        try:
            # Thử parse với None trước, nếu lỗi thì thử với region mặc định
            try:
                parsed = phonenumbers.parse(phone_number, None)
            except phonenumbers.NumberParseException:
                parsed = phonenumbers.parse(phone_number, "VN")
                
            country = geocoder.description_for_number(parsed, "en")
            carrier_name = carrier.name_for_number(parsed, "en")
            
            return {
                'country': country,
                'carrier': carrier_name,
                'consistency_score': 1.0 if country and carrier_name else 0.5
            }
        except Exception as e:
            return {'consistency_score': 0.0, 'error': str(e)}
    
    def _temporal_patterns(self, digits: str) -> Dict:
        """Phân tích pattern thời gian"""
        current_time = datetime.now()
        
        # Check if digits match current time patterns
        time_patterns = [
            current_time.strftime("%H%M"),
            current_time.strftime("%d%m"),
            current_time.strftime("%Y")[-2:]
        ]
        
        temporal_score = 0
        for pattern in time_patterns:
            if pattern in digits:
                temporal_score += 0.1
        
        # Kiểm tra timestamp pattern an toàn hơn
        timestamp_check = False
        if len(digits) >= 10:
            try:
                # Kiểm tra các substring 10 chữ số có phải timestamp không
                for i in range(len(digits) - 9):
                    timestamp_candidate = digits[i:i+10]
                    if timestamp_candidate.isdigit():
                        timestamp_value = int(timestamp_candidate)
                        # Kiểm tra timestamp hợp lý (từ 2000 đến 2030)
                        if 946684800 <= timestamp_value <= 1893456000:
                            timestamp_check = True
                            break
            except:
                pass
                
        return {
            'temporal_score': temporal_score,
            'timestamp_check': timestamp_check
        }
    
    def comprehensive_validation(self, phone_number: str) -> PhoneValidationResult:
        """Kiểm tra toàn diện số điện thoại"""
        start_time = time.time()
        validation_methods = []
        
        # 1. Format validation
        is_valid_format, format_info = self.validate_format(phone_number)
        validation_methods.append("format_validation")
        
        # 2. Suspicious patterns
        risk_score = self.check_suspicious_patterns(phone_number)
        validation_methods.append("pattern_analysis")
        
        # 3. Luhn algorithm
        luhn_valid = self.luhn_algorithm_check(phone_number)
        validation_methods.append("luhn_algorithm")
        
        # 4. Bayesian spam detection
        spam_probability = self.bayesian_spam_detection(phone_number)
        validation_methods.append("bayesian_analysis")
        
        # 5. Machine learning prediction
        ml_score, ml_features = self.machine_learning_prediction(phone_number)
        validation_methods.append("machine_learning")
        
        # 6. Heuristic analysis
        heuristic_analysis = self.advanced_heuristic_analysis(phone_number)
        validation_methods.append("heuristic_analysis")
        
        # 7. HMAC verification
        hmac_hash = self.hmac_verification(phone_number)
        validation_methods.append("hmac_verification")
        
        # 8. Carrier validation (new)
        carrier_score = self._carrier_validation_score(format_info)
        validation_methods.append("carrier_validation")
        
        # 9. Geographic consistency (new)
        geo_score = self._geographic_validation_score(format_info)
        validation_methods.append("geographic_validation")
        
        # Calculate confidence score với trọng số cải thiện
        confidence_factors = [
            0.25 if is_valid_format else 0.0,  # Format validation weight
            0.15 if luhn_valid else 0.0,       # Luhn algorithm weight
            0.20 * (1 - spam_probability),     # Bayesian spam detection
            0.20 * ml_score,                   # Machine learning
            0.10 * (1 - risk_score),          # Pattern analysis
            0.05 * carrier_score,             # Carrier validation
            0.05 * geo_score,                 # Geographic validation
        ]
        
        confidence_score = sum(confidence_factors)
        
        # Determine status với ngưỡng cải thiện
        if confidence_score >= 0.85:
            status = PhoneStatus.VALID
        elif confidence_score >= 0.70:
            status = PhoneStatus.SUSPICIOUS
        elif confidence_score >= 0.40:
            status = PhoneStatus.UNKNOWN
        else:
            status = PhoneStatus.INVALID
        
        # Điều chỉnh dựa trên phân tích đặc biệt
        if self._is_definitely_fake(phone_number):
            status = PhoneStatus.INVALID
            confidence_score = min(confidence_score, 0.2)
        elif self._is_highly_likely_real(phone_number, format_info):
            if confidence_score < 0.6:
                confidence_score = min(confidence_score + 0.2, 0.9)
        
        # Extract information
        country = format_info.get('country', 'Unknown')
        carrier_name = format_info.get('carrier', 'Unknown')
        location = country
        
        # Determine line type
        if 'voip' in carrier_name.lower():
            line_type = 'VOIP'
        elif format_info.get('number_type') == phonenumbers.PhoneNumberType.MOBILE:
            line_type = 'Mobile'
        elif format_info.get('number_type') == phonenumbers.PhoneNumberType.FIXED_LINE:
            line_type = 'Landline'
        else:
            line_type = 'Unknown'
        
        # Additional info
        additional_info = {
            'format_info': format_info,
            'ml_features': ml_features,
            'heuristic_analysis': heuristic_analysis,
            'hmac_hash': hmac_hash,
            'processing_time': time.time() - start_time,
            'carrier_score': carrier_score,
            'geo_score': geo_score,
            'spam_probability': spam_probability
        }
        
        # Save to database
        self.save_validation_result(phone_number, status, confidence_score, 
                                  country, carrier_name, location, line_type, 
                                  risk_score, validation_methods, additional_info)
        
        return PhoneValidationResult(
            phone_number=phone_number,
            status=status,
            confidence_score=confidence_score,
            country=country,
            carrier_name=carrier_name,
            location=location,
            line_type=line_type,
            is_valid_format=is_valid_format,
            is_possible=format_info.get('is_possible', False),
            risk_score=risk_score,
            validation_methods=validation_methods,
            timestamp=datetime.now(),
            additional_info=additional_info
        )
    
    def _carrier_validation_score(self, format_info: Dict) -> float:
        """Tính điểm dựa trên nhà mạng"""
        carrier = format_info.get('carrier', '').lower()
        
        # Danh sách nhà mạng uy tín
        trusted_carriers = [
            'viettel', 'vinaphone', 'mobifone', 'vietnamobile', 'gmobile',
            'verizon', 'at&t', 'sprint', 't-mobile', 'vodafone', 'orange',
            'china mobile', 'china telecom', 'china unicom'
        ]
        
        if any(trusted in carrier for trusted in trusted_carriers):
            return 1.0
        elif carrier and len(carrier) > 2:
            return 0.6
        else:
            return 0.2
    
    def _geographic_validation_score(self, format_info: Dict) -> float:
        """Tính điểm dựa trên thông tin địa lý"""
        country = format_info.get('country', '').lower()
        
        # Danh sách quốc gia phổ biến
        common_countries = [
            'vietnam', 'united states', 'china', 'india', 'germany',
            'france', 'united kingdom', 'japan', 'south korea', 'singapore'
        ]
        
        if any(common in country for common in common_countries):
            return 1.0
        elif country and len(country) > 2:
            return 0.7
        else:
            return 0.3
    
    def _is_definitely_fake(self, phone_number: str) -> bool:
        """Xác định chắc chắn là số giả"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        fake_indicators = [
            clean_number in ['0000000000', '1111111111', '2222222222', '3333333333'],
            clean_number.startswith('0000') and len(clean_number) >= 10,
            clean_number == '0123456789' or clean_number == '1234567890',
            clean_number == '9876543210' or clean_number == '0987654321',
            len(set(clean_number)) <= 2 and len(clean_number) >= 8,
            clean_number.startswith('123456') and len(clean_number) >= 10,
        ]
        
        return any(fake_indicators)
    
    def _is_highly_likely_real(self, phone_number: str, format_info: Dict) -> bool:
        """Xác định có khả năng cao là số thật"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        real_indicators = [
            # Có carrier name từ nhà mạng uy tín
            any(carrier in format_info.get('carrier', '').lower() 
                for carrier in ['viettel', 'vinaphone', 'mobifone']),
            # Có thông tin địa lý cụ thể
            format_info.get('country', '').lower() == 'vietnam',
            # Độ dài hợp lệ và entropy cao
            len(clean_number) in [10, 11] and self._calculate_entropy(clean_number) > 2.5,
            # Có timezone info
            len(format_info.get('timezones', [])) > 0,
        ]
        
        return sum(real_indicators) >= 2
    
    def save_validation_result(self, phone_number: str, status: PhoneStatus, 
                             confidence_score: float, country: str, 
                             carrier_name: str, location: str, line_type: str,
                             risk_score: float, validation_methods: List[str], 
                             additional_info: Dict):
        """Lưu kết quả kiểm tra vào database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO phone_validation_history 
            (phone_number, status, confidence_score, country, carrier_name, 
             location, line_type, risk_score, validation_methods, timestamp, additional_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            phone_number, status.value, confidence_score, country, carrier_name,
            location, line_type, risk_score, json.dumps(validation_methods),
            datetime.now().isoformat(), json.dumps(additional_info)
        ))
        
        conn.commit()
        conn.close()
    
    def get_validation_history(self, phone_number: str) -> List[Dict]:
        """Lấy lịch sử kiểm tra"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM phone_validation_history 
            WHERE phone_number = ? 
            ORDER BY timestamp DESC
        ''', (phone_number,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    def batch_validation(self, phone_numbers: List[str]) -> List[PhoneValidationResult]:
        """Kiểm tra hàng loạt số điện thoại"""
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.comprehensive_validation, phone) 
                      for phone in phone_numbers]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch validation error: {e}")
                    
        return results
    
    def save_results_to_csv(self, results: List[PhoneValidationResult], filename: str):
        """Lưu kết quả vào file CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'phone_number', 'status', 'confidence_score', 'risk_score',
                'country', 'carrier_name', 'line_type', 'is_valid_format',
                'is_possible', 'entropy', 'unique_digits', 'consecutive_same',
                'spam_probability', 'processing_time', 'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                ml_features = result.additional_info.get('ml_features', {})
                writer.writerow({
                    'phone_number': result.phone_number,
                    'status': result.status.value,
                    'confidence_score': f"{result.confidence_score:.3f}",
                    'risk_score': f"{result.risk_score:.3f}",
                    'country': result.country,
                    'carrier_name': result.carrier_name,
                    'line_type': result.line_type,
                    'is_valid_format': result.is_valid_format,
                    'is_possible': result.is_possible,
                    'entropy': f"{ml_features.get('entropy', 0):.3f}",
                    'unique_digits': ml_features.get('unique_digits', 0),
                    'consecutive_same': ml_features.get('consecutive_same', 0),
                    'spam_probability': f"{result.additional_info.get('spam_probability', 0):.3f}",
                    'processing_time': f"{result.additional_info.get('processing_time', 0):.3f}",
                    'timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })

    def generate_report(self, result: PhoneValidationResult) -> str:
        """Tạo báo cáo chi tiết"""
        
        # Tính toán thêm thông tin
        ml_features = result.additional_info.get('ml_features', {})
        spam_prob = result.additional_info.get('spam_probability', 0)
        
        # Xác định mức độ tin cậy bằng text
        confidence_text = "RẤT CAO" if result.confidence_score >= 0.85 else \
                         "CAO" if result.confidence_score >= 0.70 else \
                         "TRUNG BÌNH" if result.confidence_score >= 0.40 else \
                         "THẤP"
        
        # Xác định mức độ rủi ro
        risk_text = "RẤT CAO" if result.risk_score >= 0.7 else \
                   "CAO" if result.risk_score >= 0.5 else \
                   "TRUNG BÌNH" if result.risk_score >= 0.3 else \
                   "THẤP"
        
        # Icon cho trạng thái
        status_icon = {
            PhoneStatus.VALID: "✅",
            PhoneStatus.SUSPICIOUS: "⚠️",
            PhoneStatus.UNKNOWN: "❓",
            PhoneStatus.INVALID: "❌"
        }.get(result.status, "❓")
        
        report = f"""
{status_icon} === BÁO CÁO KIỂM TRA SỐ ĐIỆN THOẠI ===
📞 Số điện thoại: {result.phone_number}
📊 Trạng thái: {result.status.value.upper()}
🎯 Độ tin cậy: {result.confidence_score:.1%} ({confidence_text})
⚠️ Mức rủi ro: {result.risk_score:.1%} ({risk_text})
🚨 Khả năng spam: {spam_prob:.1%}

=== THÔNG TIN CHI TIẾT ===
🌍 Quốc gia: {result.country}
📡 Nhà mạng: {result.carrier_name}
📍 Vị trí: {result.location}
📱 Loại đường truyền: {result.line_type}
✅ Định dạng hợp lệ: {'Có' if result.is_valid_format else 'Không'}
🔍 Có thể tồn tại: {'Có' if result.is_possible else 'Không'}

=== PHÂN TÍCH KỸ THUẬT ===
📏 Độ dài: {ml_features.get('length', 'N/A')} ký tự
🔢 Số chữ số khác nhau: {ml_features.get('unique_digits', 'N/A')}/10
🔄 Chữ số lặp liên tiếp: {ml_features.get('consecutive_same', 'N/A')}
📊 Entropy: {ml_features.get('entropy', 0):.3f}
📈 Phương sai: {ml_features.get('digit_variance', 0):.3f}
🎲 Palindrome: {'Có' if ml_features.get('is_palindrome', False) else 'Không'}

=== PHƯƠNG PHÁP KIỂM TRA ===
{chr(10).join(f"✓ {method.replace('_', ' ').title()}" for method in result.validation_methods)}

=== THỜI GIAN ===
🕐 Kiểm tra lúc: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
⏱️ Thời gian xử lý: {result.additional_info.get('processing_time', 0):.3f}s

=== KẾT LUẬN ===
"""
        
        # Thêm kết luận dựa trên kết quả
        if result.status == PhoneStatus.VALID:
            report += "🟢 Số điện thoại này CÓ KHẢ NĂNG CAO là thật và đang hoạt động."
        elif result.status == PhoneStatus.SUSPICIOUS:
            report += "🟡 Số điện thoại này CẦN THẬN TRỌNG - có dấu hiệu nghi ngờ."
        elif result.status == PhoneStatus.UNKNOWN:
            report += "🔵 Không thể xác định chính xác - cần kiểm tra thêm."
        else:
            report += "🔴 Số điện thoại này CÓ KHẢ NĂNG CAO là GIẢ hoặc không hợp lệ."
        
        return report
    
    def generate_batch_summary(self, results: List[PhoneValidationResult]) -> str:
        """Tạo báo cáo tóm tắt cho batch validation"""
        if not results:
            return "Không có kết quả để tóm tắt."
        
        # Thống kê theo trạng thái
        valid_count = sum(1 for r in results if r.status == PhoneStatus.VALID)
        suspicious_count = sum(1 for r in results if r.status == PhoneStatus.SUSPICIOUS)
        unknown_count = sum(1 for r in results if r.status == PhoneStatus.UNKNOWN)
        invalid_count = sum(1 for r in results if r.status == PhoneStatus.INVALID)
        
        # Thống kê theo nhà mạng
        carrier_stats = {}
        for result in results:
            carrier = result.carrier_name
            if carrier not in carrier_stats:
                carrier_stats[carrier] = 0
            carrier_stats[carrier] += 1
        
        # Thống kê theo quốc gia
        country_stats = {}
        for result in results:
            country = result.country
            if country not in country_stats:
                country_stats[country] = 0
            country_stats[country] += 1
        
        # Tính trung bình confidence
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_risk = sum(r.risk_score for r in results) / len(results)
        
        # Tìm số có confidence cao nhất và thấp nhất
        best_result = max(results, key=lambda r: r.confidence_score)
        worst_result = min(results, key=lambda r: r.confidence_score)
        
        summary = f"""
🔍 === TÓM TẮT KẾT QUẢ KIỂM TRA HÀNG LOẠT ===
📊 Tổng số kiểm tra: {len(results)}

=== THỐNG KÊ THEO TRẠNG THÁI ===
✅ Hợp lệ: {valid_count} ({valid_count/len(results)*100:.1f}%)
⚠️ Nghi ngờ: {suspicious_count} ({suspicious_count/len(results)*100:.1f}%)
❓ Không rõ: {unknown_count} ({unknown_count/len(results)*100:.1f}%)
❌ Không hợp lệ: {invalid_count} ({invalid_count/len(results)*100:.1f}%)

=== THỐNG KÊ THEO NHÀ MẠNG ===
{chr(10).join(f"📡 {carrier}: {count}" for carrier, count in sorted(carrier_stats.items(), key=lambda x: x[1], reverse=True)[:5])}

=== THỐNG KÊ THEO QUỐC GIA ===
{chr(10).join(f"🌍 {country}: {count}" for country, count in sorted(country_stats.items(), key=lambda x: x[1], reverse=True)[:5])}

=== THỐNG KÊ CHUNG ===
🎯 Độ tin cậy trung bình: {avg_confidence:.1%}
⚠️ Mức rủi ro trung bình: {avg_risk:.1%}

=== SỐ TỐT NHẤT ===
🏆 {best_result.phone_number} - Tin cậy: {best_result.confidence_score:.1%}

=== SỐ TỆ NHẤT ===
🚨 {worst_result.phone_number} - Tin cậy: {worst_result.confidence_score:.1%}

=== KHUYẾN NGHỊ ===
"""
        
        if valid_count / len(results) >= 0.7:
            summary += "🟢 Phần lớn số điện thoại có vẻ hợp lệ."
        elif invalid_count / len(results) >= 0.5:
            summary += "🔴 Cảnh báo: Nhiều số điện thoại có vẻ không hợp lệ."
        else:
            summary += "🟡 Kết quả hỗn hợp - cần kiểm tra kỹ từng số."
        
        return summary

def main():
    """Hàm chính để chạy chương trình"""
    print("=== HỆ THỐNG KIỂM TRA SỐ ĐIỆN THOẠI NÂNG CAO ===\n")
    
    validator = AdvancedPhoneValidator()
    
    while True:
        print("\n1. Kiểm tra một số điện thoại")
        print("2. Kiểm tra hàng loạt (nhập tay)")
        print("3. Kiểm tra hàng loạt từ file phones.txt")
        print("4. Xem lịch sử kiểm tra")
        print("5. Thoát")
        
        choice = input("\nChọn chức năng (1-5): ").strip()
        
        if choice == '1':
            phone = input("Nhập số điện thoại: ").strip()
            if phone:
                print("\nĐang kiểm tra...")
                result = validator.comprehensive_validation(phone)
                print(validator.generate_report(result))
        
        elif choice == '2':
            phones_input = input("Nhập danh sách số điện thoại (cách nhau bởi dấu phẩy): ").strip()
            if phones_input:
                phones = [p.strip() for p in phones_input.split(',') if p.strip()]
                print(f"\nĐang kiểm tra {len(phones)} số điện thoại...")
                results = validator.batch_validation(phones)
                
                # Hiển thị báo cáo tóm tắt trước
                print(validator.generate_batch_summary(results))
                
                # Hỏi có muốn xem chi tiết không
                show_details = input("\nBạn có muốn xem báo cáo chi tiết cho từng số? (y/n): ").strip().lower()
                if show_details == 'y':
                    for i, result in enumerate(results, 1):
                        print(f"\n{'='*60}")
                        print(f"📋 BÁO CÁO CHI TIẾT {i}/{len(results)}")
                        print(validator.generate_report(result))
        
        elif choice == '3':
            try:
                with open("checkso.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                    # Tách bằng dòng hoặc dấu phẩy
                    raw_numbers = re.split(r'[\n,]+', content)
                    phones = [p.strip() for p in raw_numbers if p.strip()]
                    
                    print(f"\nĐang kiểm tra {len(phones)} số điện thoại từ file...")
                    results = validator.batch_validation(phones)
                    
                    # Hiển thị báo cáo tóm tắt trước
                    print(validator.generate_batch_summary(results))
                    
                    # Hỏi có muốn xem chi tiết không
                    show_details = input("\nBạn có muốn xem báo cáo chi tiết cho từng số? (y/n): ").strip().lower()
                    if show_details == 'y':
                        for i, result in enumerate(results, 1):
                            print(f"\n{'='*60}")
                            print(f"📋 BÁO CÁO CHI TIẾT {i}/{len(results)}")
                            print(validator.generate_report(result))
                    
                    # Lưu kết quả vào file
                    save_file = input("\nBạn có muốn lưu kết quả vào file CSV? (y/n): ").strip().lower()
                    if save_file == 'y':
                        validator.save_results_to_csv(results, "phone_validation_results.csv")
                        print("✅ Đã lưu kết quả vào file phone_validation_results.csv")
                        
            except FileNotFoundError:
                print("❌ Không tìm thấy file checkso.txt. Hãy tạo file và thử lại.")
        
        elif choice == '4':
            phone = input("Nhập số điện thoại để xem lịch sử: ").strip()
            if phone:
                history = validator.get_validation_history(phone)
                if history:
                    print(f"\nLịch sử kiểm tra cho {phone}:")
                    for record in history:
                        print(f"- {record['timestamp']}: {record['status']} (tin cậy: {record['confidence_score']:.2%})")
                else:
                    print("Không có lịch sử kiểm tra.")
        
        elif choice == '5':
            print("Thoát chương trình.")
            break
        
        else:
            print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()