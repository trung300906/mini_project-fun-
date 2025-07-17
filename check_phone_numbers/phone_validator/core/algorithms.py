#!/usr/bin/env python3
"""
Advanced algorithms for phone number validation
"""

import re
import hmac
import hashlib
import math
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AlgorithmValidator:
    """Các thuật toán validation nâng cao"""
    
    def __init__(self):
        self.secret_key = "advanced_phone_validation_2024"
    
    def luhn_algorithm_check(self, phone_number: str) -> Tuple[bool, Dict]:
        """Thuật toán Luhn để kiểm tra tính hợp lệ"""
        digits = [int(d) for d in re.sub(r'\D', '', phone_number)]
        
        if len(digits) < 8:
            return False, {'error': 'Too short for Luhn check'}
        
        # Thuật toán Luhn
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit = digit // 10 + digit % 10
            checksum += digit
        
        is_valid = checksum % 10 == 0
        
        return is_valid, {
            'checksum': checksum,
            'is_valid': is_valid,
            'algorithm': 'luhn',
            'digits_processed': len(digits)
        }
    
    def mod97_check(self, phone_number: str) -> Tuple[bool, Dict]:
        """Thuật toán MOD-97 (dùng trong IBAN)"""
        digits = re.sub(r'\D', '', phone_number)
        
        if len(digits) < 8:
            return False, {'error': 'Too short for MOD-97 check'}
        
        # Chuyển đổi digits thành số nguyên
        number = int(digits)
        
        # Tính MOD-97
        remainder = number % 97
        is_valid = remainder == 1
        
        return is_valid, {
            'remainder': remainder,
            'is_valid': is_valid,
            'algorithm': 'mod97',
            'original_number': digits
        }
    
    def damm_algorithm_check(self, phone_number: str) -> Tuple[bool, Dict]:
        """Thuật toán Damm để phát hiện lỗi"""
        digits = re.sub(r'\D', '', phone_number)
        
        if not digits:
            return False, {'error': 'No digits found'}
        
        # Bảng Damm
        damm_table = [
            [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
            [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
            [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
            [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
            [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
            [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
            [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
            [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
            [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
            [2, 5, 8, 1, 4, 3, 6, 7, 9, 0]
        ]
        
        interim_digit = 0
        for digit in digits:
            interim_digit = damm_table[interim_digit][int(digit)]
        
        is_valid = interim_digit == 0
        
        return is_valid, {
            'interim_digit': interim_digit,
            'is_valid': is_valid,
            'algorithm': 'damm',
            'digits_processed': len(digits)
        }
    
    def verhoeff_algorithm_check(self, phone_number: str) -> Tuple[bool, Dict]:
        """Thuật toán Verhoeff để kiểm tra"""
        digits = re.sub(r'\D', '', phone_number)
        
        if not digits:
            return False, {'error': 'No digits found'}
        
        # Bảng nhân dihedral D5
        d_table = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        
        # Bảng permutation
        p_table = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        
        # Inverse table
        inv_table = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
        
        check = 0
        for i, digit in enumerate(reversed(digits)):
            check = d_table[check][p_table[i % 8][int(digit)]]
        
        is_valid = check == 0
        
        return is_valid, {
            'check_digit': check,
            'is_valid': is_valid,
            'algorithm': 'verhoeff',
            'digits_processed': len(digits)
        }
    
    def hmac_verification(self, phone_number: str, secret_key: Optional[str] = None) -> Dict:
        """Tạo và xác thực HMAC"""
        if secret_key is None:
            secret_key = self.secret_key
        
        # Tạo HMAC
        hmac_hash = hmac.new(
            secret_key.encode(),
            phone_number.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Tạo HMAC với MD5
        hmac_md5 = hmac.new(
            secret_key.encode(),
            phone_number.encode(),
            hashlib.md5
        ).hexdigest()
        
        # Tạo HMAC với SHA1
        hmac_sha1 = hmac.new(
            secret_key.encode(),
            phone_number.encode(),
            hashlib.sha1
        ).hexdigest()
        
        return {
            'hmac_sha256': hmac_hash,
            'hmac_md5': hmac_md5,
            'hmac_sha1': hmac_sha1,
            'algorithm': 'hmac',
            'input': phone_number
        }
    
    def bayesian_spam_detection(self, phone_number: str) -> Tuple[float, Dict]:
        """Thuật toán Bayesian nâng cao để phát hiện spam"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return 0.0, {'error': 'No digits found'}
        
        # Các chỉ số spam với trọng số được cải thiện
        spam_indicators = [
            (len(clean_number) < 9, 0.8, 'too_short'),
            (len(clean_number) > 15, 0.7, 'too_long'),
            (clean_number.count('0') > 6, 0.6, 'too_many_zeros'),
            (clean_number.count('1') > 6, 0.5, 'too_many_ones'),
            (len(set(clean_number)) < 4, 0.9, 'too_few_unique_digits'),
            (len(set(clean_number)) < 6, 0.4, 'few_unique_digits'),
            (clean_number.startswith('0000'), 0.9, 'starts_with_0000'),
            (clean_number.startswith('1111'), 0.8, 'starts_with_1111'),
            (clean_number.endswith('0000'), 0.7, 'ends_with_0000'),
            (self._is_ascending_sequence(clean_number), 0.6, 'ascending_sequence'),
            (self._is_descending_sequence(clean_number), 0.6, 'descending_sequence'),
            (self._has_repeated_pattern(clean_number), 0.5, 'repeated_pattern'),
            (clean_number.count(clean_number[0]) > len(clean_number) * 0.6, 0.7, 'dominant_digit'),
            (self._is_all_same_digit(clean_number), 0.95, 'all_same_digit'),
            (self._is_common_fake_number(clean_number), 0.9, 'common_fake_number'),
            (self._has_suspicious_prefix(clean_number), 0.6, 'suspicious_prefix'),
            (self._has_test_pattern(clean_number), 0.8, 'test_pattern'),
        ]
        
        # Tính xác suất spam theo Bayesian
        total_weight = sum(weight for _, weight, _ in spam_indicators)
        spam_score = sum(weight for condition, weight, _ in spam_indicators if condition)
        triggered_indicators = [name for condition, _, name in spam_indicators if condition]
        
        spam_probability = spam_score / total_weight if total_weight > 0 else 0
        
        # Điều chỉnh với prior probability
        prior_spam_probability = 0.3  # 30% số điện thoại là spam
        
        # Bayesian update
        likelihood = spam_probability
        posterior = (likelihood * prior_spam_probability) / (
            likelihood * prior_spam_probability + (1 - likelihood) * (1 - prior_spam_probability)
        )
        
        return min(posterior, 1.0), {
            'spam_probability': spam_probability,
            'posterior_probability': posterior,
            'prior_probability': prior_spam_probability,
            'triggered_indicators': triggered_indicators,
            'total_indicators': len(spam_indicators),
            'triggered_count': len(triggered_indicators),
            'algorithm': 'bayesian_spam'
        }
    
    def advanced_entropy_analysis(self, phone_number: str) -> Dict:
        """Phân tích entropy nâng cao"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return {'error': 'No digits found'}
        
        # Shannon entropy
        shannon_entropy = self._calculate_shannon_entropy(clean_number)
        
        # Rényi entropy
        renyi_entropy = self._calculate_renyi_entropy(clean_number, alpha=2)
        
        # Kolmogorov complexity approximation
        kolmogorov_complexity = self._approximate_kolmogorov_complexity(clean_number)
        
        # Conditional entropy
        conditional_entropy = self._calculate_conditional_entropy(clean_number)
        
        # Mutual information
        mutual_information = self._calculate_mutual_information(clean_number)
        
        return {
            'shannon_entropy': shannon_entropy,
            'renyi_entropy': renyi_entropy,
            'kolmogorov_complexity': kolmogorov_complexity,
            'conditional_entropy': conditional_entropy,
            'mutual_information': mutual_information,
            'normalized_entropy': shannon_entropy / math.log2(10),
            'entropy_rate': shannon_entropy / len(clean_number),
            'algorithm': 'entropy_analysis'
        }
    
    def monte_carlo_validation(self, phone_number: str, iterations: int = 10000) -> Dict:
        """Validation sử dụng Monte Carlo"""
        import random
        
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return {'error': 'No digits found'}
        
        # Tạo số ngẫu nhiên cùng độ dài
        random_numbers = []
        for _ in range(iterations):
            random_number = ''.join(random.choices('0123456789', k=len(clean_number)))
            random_numbers.append(random_number)
        
        # So sánh với số gốc
        similarities = []
        for random_number in random_numbers:
            similarity = self._calculate_similarity(clean_number, random_number)
            similarities.append(similarity)
        
        # Thống kê
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        min_similarity = min(similarities)
        
        # Percentile của số gốc
        self_similarity = 1.0  # Similarity with itself
        percentile = sum(1 for s in similarities if s < self_similarity) / len(similarities) * 100
        
        return {
            'average_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'percentile': percentile,
            'is_unusual': percentile > 95,  # Top 5% unusual
            'iterations': iterations,
            'algorithm': 'monte_carlo'
        }
    
    def markov_chain_analysis(self, phone_number: str) -> Dict:
        """Phân tích chuỗi Markov"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if len(clean_number) < 2:
            return {'error': 'Too short for Markov analysis'}
        
        # Tạo transition matrix
        transitions = {}
        for i in range(len(clean_number) - 1):
            current = clean_number[i]
            next_digit = clean_number[i + 1]
            
            if current not in transitions:
                transitions[current] = {}
            if next_digit not in transitions[current]:
                transitions[current][next_digit] = 0
            transitions[current][next_digit] += 1
        
        # Normalize probabilities
        for current in transitions:
            total = sum(transitions[current].values())
            for next_digit in transitions[current]:
                transitions[current][next_digit] /= total
        
        # Tính entropy của transition matrix
        total_entropy = 0
        for current in transitions:
            for next_digit in transitions[current]:
                prob = transitions[current][next_digit]
                if prob > 0:
                    total_entropy -= prob * math.log2(prob)
        
        # Tính steady state probability
        steady_state = self._calculate_steady_state(transitions)
        
        return {
            'transition_matrix': transitions,
            'transition_entropy': total_entropy,
            'steady_state': steady_state,
            'is_random': total_entropy > 2.0,
            'algorithm': 'markov_chain'
        }
    
    def fourier_transform_analysis(self, phone_number: str) -> Dict:
        """Phân tích Fourier transform"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if len(clean_number) < 4:
            return {'error': 'Too short for Fourier analysis'}
        
        # Chuyển đổi digits thành signal
        signal = [int(d) for d in clean_number]
        
        # Discrete Fourier Transform (DFT) đơn giản
        n = len(signal)
        frequencies = []
        
        for k in range(n):
            real_part = 0
            imag_part = 0
            
            for t in range(n):
                angle = 2 * math.pi * k * t / n
                real_part += signal[t] * math.cos(angle)
                imag_part -= signal[t] * math.sin(angle)
            
            magnitude = math.sqrt(real_part**2 + imag_part**2)
            frequencies.append(magnitude)
        
        # Phân tích frequency domain
        max_frequency = max(frequencies)
        dominant_frequency = frequencies.index(max_frequency)
        
        # Tính spectral centroid
        spectral_centroid = sum(i * freq for i, freq in enumerate(frequencies)) / sum(frequencies)
        
        return {
            'frequencies': frequencies,
            'max_frequency': max_frequency,
            'dominant_frequency': dominant_frequency,
            'spectral_centroid': spectral_centroid,
            'has_periodic_pattern': max_frequency > sum(frequencies) / len(frequencies) * 2,
            'algorithm': 'fourier_transform'
        }
    
    def statistical_distribution_analysis(self, phone_number: str) -> Dict:
        """Phân tích phân phối thống kê"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return {'error': 'No digits found'}
        
        digits = [int(d) for d in clean_number]
        
        # Các thống kê cơ bản
        mean = sum(digits) / len(digits)
        variance = sum((d - mean)**2 for d in digits) / len(digits)
        std_dev = math.sqrt(variance)
        
        # Skewness
        skewness = sum((d - mean)**3 for d in digits) / (len(digits) * std_dev**3) if std_dev > 0 else 0
        
        # Kurtosis
        kurtosis = sum((d - mean)**4 for d in digits) / (len(digits) * std_dev**4) - 3 if std_dev > 0 else 0
        
        # Chi-square test for uniformity
        expected_freq = len(digits) / 10
        observed_freq = [digits.count(i) for i in range(10)]
        chi_square = sum((obs - expected_freq)**2 / expected_freq for obs in observed_freq)
        
        # Kolmogorov-Smirnov test approximation
        sorted_digits = sorted(digits)
        ks_statistic = max(abs(i/len(digits) - (sorted_digits[i]+1)/10) for i in range(len(digits)))
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'chi_square': chi_square,
            'ks_statistic': ks_statistic,
            'is_uniform': chi_square < 16.92,  # 95% confidence level
            'is_normally_distributed': abs(skewness) < 0.5 and abs(kurtosis) < 0.5,
            'algorithm': 'statistical_distribution'
        }
    
    # Helper methods
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
        
        for pattern_length in range(2, 5):
            if len(digits) < pattern_length * 2:
                continue
            
            pattern = digits[:pattern_length]
            repeated_pattern = pattern * (len(digits) // pattern_length)
            
            if digits.startswith(repeated_pattern[:len(digits)]):
                return True
        
        return False
    
    def _is_all_same_digit(self, digits: str) -> bool:
        """Kiểm tra tất cả cùng chữ số"""
        return len(set(digits)) == 1 and len(digits) > 4
    
    def _is_common_fake_number(self, digits: str) -> bool:
        """Kiểm tra số giả phổ biến"""
        fake_numbers = [
            '0000000000', '1111111111', '2222222222', '3333333333',
            '4444444444', '5555555555', '6666666666', '7777777777',
            '8888888888', '9999999999', '0123456789', '1234567890',
            '9876543210', '0987654321'
        ]
        return digits in fake_numbers
    
    def _has_suspicious_prefix(self, digits: str) -> bool:
        """Kiểm tra prefix nghi ngờ"""
        suspicious_prefixes = ['0000', '1111', '2222', '3333', '4444', '5555', '6666', '7777', '8888', '9999']
        return any(digits.startswith(prefix) for prefix in suspicious_prefixes)
    
    def _has_test_pattern(self, digits: str) -> bool:
        """Kiểm tra pattern test"""
        test_patterns = ['1234', '5678', '9012', '0000', '1111', '2222', '3333']
        return any(pattern in digits for pattern in test_patterns)
    
    def _calculate_shannon_entropy(self, digits: str) -> float:
        """Tính Shannon entropy"""
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
        
        return entropy
    
    def _calculate_renyi_entropy(self, digits: str, alpha: float = 2.0) -> float:
        """Tính Rényi entropy"""
        if not digits or alpha == 1.0:
            return self._calculate_shannon_entropy(digits)
        
        digit_counts = {}
        for digit in digits:
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        length = len(digits)
        sum_powers = sum((count / length) ** alpha for count in digit_counts.values())
        
        if sum_powers == 0:
            return 0.0
        
        return math.log2(sum_powers) / (1 - alpha)
    
    def _approximate_kolmogorov_complexity(self, digits: str) -> float:
        """Xấp xỉ Kolmogorov complexity"""
        # Sử dụng compression ratio như xấp xỉ
        import zlib
        
        compressed = zlib.compress(digits.encode())
        compression_ratio = len(compressed) / len(digits)
        
        # Normalize to 0-1 range
        return min(compression_ratio, 1.0)
    
    def _calculate_conditional_entropy(self, digits: str) -> float:
        """Tính conditional entropy H(X|Y)"""
        if len(digits) < 2:
            return 0.0
        
        # H(X_i | X_{i-1})
        transitions = {}
        for i in range(len(digits) - 1):
            current = digits[i]
            next_digit = digits[i + 1]
            
            if current not in transitions:
                transitions[current] = []
            transitions[current].append(next_digit)
        
        conditional_entropy = 0.0
        for current in transitions:
            next_digits = transitions[current]
            total = len(next_digits)
            
            # Tính entropy cho distribution of next digits given current
            next_counts = {}
            for next_digit in next_digits:
                next_counts[next_digit] = next_counts.get(next_digit, 0) + 1
            
            entropy = 0.0
            for count in next_counts.values():
                prob = count / total
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            # Weight by probability of current digit
            current_prob = total / (len(digits) - 1)
            conditional_entropy += current_prob * entropy
        
        return conditional_entropy
    
    def _calculate_mutual_information(self, digits: str) -> float:
        """Tính mutual information"""
        if len(digits) < 2:
            return 0.0
        
        # I(X;Y) = H(X) - H(X|Y)
        entropy_x = self._calculate_shannon_entropy(digits[:-1])
        conditional_entropy = self._calculate_conditional_entropy(digits)
        
        return entropy_x - conditional_entropy
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Tính similarity giữa hai string"""
        if len(str1) != len(str2):
            return 0.0
        
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        return matches / len(str1)
    
    def _calculate_steady_state(self, transitions: Dict) -> Dict:
        """Tính steady state probability của Markov chain"""
        # Simplified calculation - just return uniform distribution
        digits = list(transitions.keys())
        if not digits:
            return {}
        
        steady_state = {}
        for digit in '0123456789':
            steady_state[digit] = 1.0 / 10
        
        return steady_state
