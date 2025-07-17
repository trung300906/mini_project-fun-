#!/usr/bin/env python3
"""
Pattern analysis for phone numbers
"""

import re
import math
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Phân tích pattern số điện thoại"""
    
    def __init__(self):
        self.load_patterns()
    
    def load_patterns(self):
        """Tải các pattern nghi ngờ và hợp lệ"""
        
        # Patterns nghi ngờ
        self.suspicious_patterns = [
            # Các pattern số ảo phổ biến Việt Nam
            r'^(\+?84)?0?(123|111|000|999)\d{7}$',
            r'^(\+?84)?0?(1234|2345|3456|4567|5678|6789)\d{6}$',
            r'^(\+?84)?0?(\d)\1{8}$',  # Repeated digits
            r'^(\+?84)?0?(0000|1111|2222|3333|4444|5555|6666|7777|8888|9999)\d{6}$',
            r'^(\+?84)?0?(012|098|032|033|034|035|036|037|038|039|070|071|072|073|074|075|076|077|078|079)\d{4}$',
            
            # US fake numbers
            r'^(\+?1)?[2-9]\d{2}555\d{4}$',
            r'^(\+?1)?(000|111|222|333|444|555|666|777|888|999)\d{7}$',
            
            # China suspicious
            r'^(\+?86)?1[35]\d{9}$',
            
            # General suspicious patterns
            r'^\+?(\d)\1{10,}$',  # Too many repeated digits
            r'^\+?(0123456789|1234567890|9876543210)$',  # Sequential numbers
            r'^\+?(\d{3})\1{2,}$',  # Repeated 3-digit patterns
            r'^\+?(\d{4})\1{2,}$',  # Repeated 4-digit patterns
        ]
        
        # VoIP patterns
        self.voip_patterns = [
            r'^(\+?1)?[2-9]\d{2}[2-9]\d{2}\d{4}$',  # US VoIP
            r'^(\+?44)?[1-9]\d{8,9}$',  # UK VoIP
            r'^(\+?84)?0?(01|02|03|04|05|06|07|08|09)8\d{7}$',  # VN VoIP-like
            r'^(\+?84)?0?1900\d{6}$',  # Premium rate VN
        ]
        
        # Patterns hợp lệ VN
        self.valid_vn_patterns = [
            r'^(\+?84)?0?[3-9]\d{8}$',  # VN mobile general
            r'^(\+?84)?0?(86|96|97|98|32|33|34|35|36|37|38|39|81|82|83|84|85|88|91|94|76|77|78|79|90|93|70|71|72|73|74|75|59|58|56|57|099|0199)\d{7}$',
            r'^(\+?84)?0?(24|28|222|233|234|235|236|237|238|239|240|241|242|243|244|245|246|247|248|249|250|251|252|253|254|255|256|257|258|259|260|261|262|263|264|265|266|267|268|269|270|271|272|273|274|275|276|277|278|279|280|281|282|283|284|285|286|287|288|289|290|291|292|293|294|295|296|297|298|299)\d{7}$',  # VN landline
        ]
        
        # Patterns nhà mạng VN
        self.vn_carrier_patterns = {
            'viettel': [
                r'^(\+?84)?0?(032|033|034|035|036|037|038|039)\d{7}$',
                r'^(\+?84)?0?(096|097|098|086)\d{7}$',
            ],
            'vinaphone': [
                r'^(\+?84)?0?(088|091|094|083|084|085|081|082)\d{7}$',
            ],
            'mobifone': [
                r'^(\+?84)?0?(070|071|072|073|074|075|076|077|078|079)\d{7}$',
                r'^(\+?84)?0?(090|093|089)\d{7}$',
            ],
            'vietnamobile': [
                r'^(\+?84)?0?(092|056|058)\d{7}$',
            ],
            'gmobile': [
                r'^(\+?84)?0?(099|059)\d{7}$',
            ]
        }
    
    def check_suspicious_patterns(self, phone_number: str) -> Tuple[float, List[str]]:
        """Kiểm tra patterns nghi ngờ"""
        risk_score = 0.0
        matched_patterns = []
        
        # Kiểm tra patterns nghi ngờ
        for i, pattern in enumerate(self.suspicious_patterns):
            if re.match(pattern, phone_number):
                risk_score += 0.4
                matched_patterns.append(f"suspicious_pattern_{i}")
        
        # Kiểm tra VoIP patterns
        for i, pattern in enumerate(self.voip_patterns):
            if re.match(pattern, phone_number):
                risk_score += 0.3
                matched_patterns.append(f"voip_pattern_{i}")
        
        # Kiểm tra patterns hợp lệ VN (giảm risk)
        for i, pattern in enumerate(self.valid_vn_patterns):
            if re.match(pattern, phone_number):
                risk_score -= 0.2
                matched_patterns.append(f"valid_vn_pattern_{i}")
        
        return max(0.0, min(risk_score, 1.0)), matched_patterns
    
    def identify_carrier_by_pattern(self, phone_number: str) -> Optional[str]:
        """Xác định nhà mạng qua pattern"""
        for carrier, patterns in self.vn_carrier_patterns.items():
            for pattern in patterns:
                if re.match(pattern, phone_number):
                    return carrier
        return None
    
    def analyze_digit_patterns(self, phone_number: str) -> Dict:
        """Phân tích patterns chữ số"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        return {
            'is_palindrome': self._is_palindrome(clean_number),
            'has_ascending_sequence': self._is_ascending_sequence(clean_number),
            'has_descending_sequence': self._is_descending_sequence(clean_number),
            'has_repeated_pattern': self._has_repeated_pattern(clean_number),
            'consecutive_same_count': self._max_consecutive_same(clean_number),
            'unique_digits_count': len(set(clean_number)),
            'digit_frequency': self._digit_frequency(clean_number),
            'pattern_complexity': self._calculate_pattern_complexity(clean_number),
            'symmetry_score': self._calculate_symmetry_score(clean_number),
            'randomness_score': self._calculate_randomness_score(clean_number)
        }
    
    def _is_palindrome(self, digits: str) -> bool:
        """Kiểm tra palindrome"""
        return digits == digits[::-1] and len(digits) > 4
    
    def _is_ascending_sequence(self, digits: str) -> bool:
        """Kiểm tra dãy tăng dần"""
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
        """Kiểm tra dãy giảm dần"""
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
        
        for pattern_length in range(2, 5):
            if len(digits) < pattern_length * 2:
                continue
            
            pattern = digits[:pattern_length]
            repeated_pattern = pattern * (len(digits) // pattern_length)
            
            if digits.startswith(repeated_pattern[:len(digits)]):
                return True
        
        return False
    
    def _max_consecutive_same(self, digits: str) -> int:
        """Đếm chữ số giống nhau liên tiếp tối đa"""
        if not digits:
            return 0
        
        max_count = 1
        current_count = 1
        
        for i in range(1, len(digits)):
            if digits[i] == digits[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        
        return max_count
    
    def _digit_frequency(self, digits: str) -> Dict[str, int]:
        """Tần suất xuất hiện các chữ số"""
        frequency = {}
        for digit in '0123456789':
            frequency[digit] = digits.count(digit)
        return frequency
    
    def _calculate_pattern_complexity(self, digits: str) -> float:
        """Tính độ phức tạp pattern"""
        if not digits:
            return 0.0
        
        # Các yếu tố đánh giá độ phức tạp
        factors = [
            len(set(digits)) / 10,  # Độ đa dạng chữ số
            1 - (self._max_consecutive_same(digits) / len(digits)),  # Ít lặp lại
            1 - int(self._is_palindrome(digits)),  # Không palindrome
            1 - int(self._is_ascending_sequence(digits)),  # Không tăng dần
            1 - int(self._is_descending_sequence(digits)),  # Không giảm dần
            1 - int(self._has_repeated_pattern(digits)),  # Không lặp pattern
        ]
        
        return sum(factors) / len(factors)
    
    def _calculate_symmetry_score(self, digits: str) -> float:
        """Tính điểm đối xứng"""
        if len(digits) < 2:
            return 0.0
        
        symmetry_score = 0.0
        mid = len(digits) // 2
        
        for i in range(mid):
            if digits[i] == digits[-(i+1)]:
                symmetry_score += 1
        
        return symmetry_score / mid
    
    def _calculate_randomness_score(self, digits: str) -> float:
        """Tính điểm ngẫu nhiên (entropy)"""
        if not digits:
            return 0.0
        
        digit_counts = {}
        for digit in digits:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        entropy = 0.0
        length = len(digits)
        for count in digit_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize entropy (max entropy for 10 digits is log2(10))
        max_entropy = math.log2(10)
        return entropy / max_entropy
    
    def advanced_pattern_analysis(self, phone_number: str) -> Dict:
        """Phân tích pattern nâng cao"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        # Phân tích cơ bản
        basic_patterns = self.analyze_digit_patterns(phone_number)
        
        # Phân tích nâng cao
        advanced_analysis = {
            'length_analysis': self._analyze_length(clean_number),
            'prefix_analysis': self._analyze_prefix(clean_number),
            'suffix_analysis': self._analyze_suffix(clean_number),
            'middle_analysis': self._analyze_middle_digits(clean_number),
            'mathematical_properties': self._analyze_mathematical_properties(clean_number),
            'geographical_patterns': self._analyze_geographical_patterns(clean_number),
            'temporal_patterns': self._analyze_temporal_patterns(clean_number),
            'linguistic_patterns': self._analyze_linguistic_patterns(clean_number),
        }
        
        return {**basic_patterns, **advanced_analysis}
    
    def _analyze_length(self, digits: str) -> Dict:
        """Phân tích độ dài"""
        length = len(digits)
        
        length_categories = {
            'very_short': length < 8,
            'short': 8 <= length < 10,
            'normal': 10 <= length <= 12,
            'long': 12 < length <= 15,
            'very_long': length > 15
        }
        
        return {
            'length': length,
            'category': next(k for k, v in length_categories.items() if v),
            'is_standard_length': 10 <= length <= 12,
            'length_score': 1.0 if 10 <= length <= 12 else max(0.0, 1.0 - abs(length - 11) * 0.1)
        }
    
    def _analyze_prefix(self, digits: str) -> Dict:
        """Phân tích prefix"""
        if len(digits) < 3:
            return {'error': 'Too short for prefix analysis'}
        
        prefix_2 = digits[:2]
        prefix_3 = digits[:3]
        prefix_4 = digits[:4]
        
        # Các prefix phổ biến
        common_prefixes = {
            'vietnam_mobile': ['032', '033', '034', '035', '036', '037', '038', '039', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '081', '082', '083', '084', '085', '086', '088', '089', '090', '091', '092', '093', '094', '096', '097', '098', '099'],
            'vietnam_landline': ['024', '028', '222', '233', '234', '235', '236', '237', '238', '239'],
            'international': ['001', '002', '003', '004', '005', '006', '007', '008', '009'],
            'premium': ['1900', '1901', '1902']
        }
        
        prefix_type = 'unknown'
        for p_type, prefixes in common_prefixes.items():
            if prefix_3 in prefixes or prefix_4 in prefixes:
                prefix_type = p_type
                break
        
        return {
            'prefix_2': prefix_2,
            'prefix_3': prefix_3,
            'prefix_4': prefix_4,
            'prefix_type': prefix_type,
            'is_common_prefix': prefix_type != 'unknown',
            'starts_with_zero': digits.startswith('0'),
            'starts_with_country_code': digits.startswith('84')
        }
    
    def _analyze_suffix(self, digits: str) -> Dict:
        """Phân tích suffix"""
        if len(digits) < 3:
            return {'error': 'Too short for suffix analysis'}
        
        suffix_2 = digits[-2:]
        suffix_3 = digits[-3:]
        suffix_4 = digits[-4:]
        
        # Phân tích tần suất ending
        ending_patterns = {
            'even_ending': int(digits[-1]) % 2 == 0,
            'odd_ending': int(digits[-1]) % 2 == 1,
            'zero_ending': digits.endswith('0'),
            'repeated_ending': len(set(suffix_2)) == 1,
            'sequential_ending': self._is_sequential(suffix_3),
        }
        
        return {
            'suffix_2': suffix_2,
            'suffix_3': suffix_3,
            'suffix_4': suffix_4,
            'last_digit': int(digits[-1]),
            **ending_patterns
        }
    
    def _analyze_middle_digits(self, digits: str) -> Dict:
        """Phân tích chữ số giữa"""
        if len(digits) < 5:
            return {'error': 'Too short for middle analysis'}
        
        middle_part = digits[2:-2]  # Bỏ 2 đầu và 2 cuối
        
        return {
            'middle_part': middle_part,
            'middle_length': len(middle_part),
            'middle_entropy': self._calculate_randomness_score(middle_part),
            'middle_complexity': self._calculate_pattern_complexity(middle_part),
            'middle_unique_digits': len(set(middle_part)),
            'middle_repeated_max': self._max_consecutive_same(middle_part)
        }
    
    def _analyze_mathematical_properties(self, digits: str) -> Dict:
        """Phân tích tính chất toán học"""
        if not digits:
            return {}
        
        digit_list = [int(d) for d in digits]
        
        # Tính tổng digits
        digit_sum = sum(digit_list)
        
        # Tính tích digits (an toàn)
        digit_product = 1
        for digit in digit_list:
            if digit == 0:
                digit_product = 0
                break
            digit_product *= digit
        
        return {
            'sum': digit_sum,
            'product': digit_product,
            'average': digit_sum / len(digit_list),
            'is_sum_prime': self._is_prime(digit_sum),
            'is_sum_even': digit_sum % 2 == 0,
            'digit_root': self._calculate_digit_root(digit_sum),
            'fibonacci_score': self._fibonacci_similarity(digit_list),
            'arithmetic_progression_score': self._arithmetic_progression_score(digit_list),
            'geometric_progression_score': self._geometric_progression_score(digit_list),
        }
    
    def _analyze_geographical_patterns(self, digits: str) -> Dict:
        """Phân tích pattern địa lý"""
        # Phân tích dựa trên prefix và patterns địa lý
        prefix_3 = digits[:3] if len(digits) >= 3 else ""
        
        # Mapping prefix với vùng miền (VN)
        region_mapping = {
            'north': ['024', '033', '034', '035', '036', '037', '038', '039', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079'],
            'central': ['032', '081', '082', '083', '084', '085'],
            'south': ['028', '088', '089', '090', '091', '092', '093', '094', '096', '097', '098', '099'],
            'nationwide': ['086']
        }
        
        detected_region = 'unknown'
        for region, prefixes in region_mapping.items():
            if prefix_3 in prefixes:
                detected_region = region
                break
        
        return {
            'detected_region': detected_region,
            'is_regional_pattern': detected_region != 'unknown',
            'prefix_region_consistency': detected_region in ['north', 'central', 'south', 'nationwide']
        }
    
    def _analyze_temporal_patterns(self, digits: str) -> Dict:
        """Phân tích pattern thời gian"""
        from datetime import datetime
        
        current_time = datetime.now()
        
        # Các pattern thời gian có thể có trong số
        time_patterns = {
            'contains_current_hour': current_time.strftime("%H") in digits,
            'contains_current_minute': current_time.strftime("%M") in digits,
            'contains_current_day': current_time.strftime("%d") in digits,
            'contains_current_month': current_time.strftime("%m") in digits,
            'contains_current_year_2d': current_time.strftime("%y") in digits,
            'contains_birth_year_pattern': any(year in digits for year in ['198', '199', '200', '201']),
            'contains_common_dates': any(date in digits for date in ['0101', '0102', '0103', '3112', '1505']),
        }
        
        return time_patterns
    
    def _analyze_linguistic_patterns(self, digits: str) -> Dict:
        """Phân tích pattern ngôn ngữ"""
        # Phân tích dựa trên tần suất xuất hiện chữ số trong ngôn ngữ
        
        # Tần suất chữ số trong tiếng Việt (dựa trên nghiên cứu)
        vietnamese_digit_frequency = {
            '0': 0.12, '1': 0.15, '2': 0.09, '3': 0.08, '4': 0.07,
            '5': 0.06, '6': 0.08, '7': 0.09, '8': 0.11, '9': 0.15
        }
        
        # Tính độ lệch so với tần suất tự nhiên
        actual_frequency = {}
        for digit in '0123456789':
            actual_frequency[digit] = digits.count(digit) / len(digits) if digits else 0
        
        frequency_deviation = 0
        for digit in '0123456789':
            expected = vietnamese_digit_frequency[digit]
            actual = actual_frequency[digit]
            frequency_deviation += abs(expected - actual)
        
        return {
            'frequency_deviation': frequency_deviation,
            'natural_frequency_score': max(0, 1 - frequency_deviation),
            'contains_lucky_numbers': any(lucky in digits for lucky in ['8', '9', '6']),
            'contains_unlucky_numbers': any(unlucky in digits for unlucky in ['4', '13']),
            'linguistic_naturalness': 1 - frequency_deviation / 2  # Normalize
        }
    
    def _is_sequential(self, digits: str) -> bool:
        """Kiểm tra dãy số liên tiếp"""
        if len(digits) < 3:
            return False
        
        for i in range(len(digits) - 2):
            if (int(digits[i+1]) == int(digits[i]) + 1 and 
                int(digits[i+2]) == int(digits[i+1]) + 1):
                return True
        return False
    
    def _is_prime(self, n: int) -> bool:
        """Kiểm tra số nguyên tố"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _calculate_digit_root(self, n: int) -> int:
        """Tính digital root"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n
    
    def _fibonacci_similarity(self, digits: List[int]) -> float:
        """Tính độ tương tự với dãy Fibonacci"""
        if len(digits) < 3:
            return 0.0
        
        fib_count = 0
        for i in range(2, len(digits)):
            if digits[i] == digits[i-1] + digits[i-2]:
                fib_count += 1
        
        return fib_count / max(len(digits) - 2, 1)
    
    def _arithmetic_progression_score(self, digits: List[int]) -> float:
        """Tính điểm cấp số cộng"""
        if len(digits) < 3:
            return 0.0
        
        progression_count = 0
        for i in range(2, len(digits)):
            if digits[i] - digits[i-1] == digits[i-1] - digits[i-2]:
                progression_count += 1
        
        return progression_count / max(len(digits) - 2, 1)
    
    def _geometric_progression_score(self, digits: List[int]) -> float:
        """Tính điểm cấp số nhân"""
        if len(digits) < 3:
            return 0.0
        
        progression_count = 0
        for i in range(2, len(digits)):
            if (digits[i-2] != 0 and digits[i-1] != 0 and 
                digits[i] / digits[i-1] == digits[i-1] / digits[i-2]):
                progression_count += 1
        
        return progression_count / max(len(digits) - 2, 1)
