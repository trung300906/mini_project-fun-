#!/usr/bin/env python3
"""
Machine Learning module for phone number validation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import re
import math
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PhoneMLFeatureExtractor:
    """Trích xuất features cho Machine Learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def extract_features(self, phone_number: str) -> Dict:
        """Trích xuất features từ số điện thoại"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return self._get_empty_features()
        
        features = {
            # Basic features
            'length': len(clean_number),
            'unique_digits': len(set(clean_number)),
            'starts_with_zero': int(clean_number.startswith('0')),
            'starts_with_country_code': int(clean_number.startswith('84')),
            
            # Digit analysis
            'digit_0_count': clean_number.count('0'),
            'digit_1_count': clean_number.count('1'),
            'digit_2_count': clean_number.count('2'),
            'digit_3_count': clean_number.count('3'),
            'digit_4_count': clean_number.count('4'),
            'digit_5_count': clean_number.count('5'),
            'digit_6_count': clean_number.count('6'),
            'digit_7_count': clean_number.count('7'),
            'digit_8_count': clean_number.count('8'),
            'digit_9_count': clean_number.count('9'),
            
            # Pattern features
            'consecutive_same_max': self._max_consecutive_same(clean_number),
            'is_palindrome': int(self._is_palindrome(clean_number)),
            'is_ascending': int(self._is_ascending_sequence(clean_number)),
            'is_descending': int(self._is_descending_sequence(clean_number)),
            'has_repeated_pattern': int(self._has_repeated_pattern(clean_number)),
            
            # Statistical features
            'mean': np.mean([int(d) for d in clean_number]),
            'std': np.std([int(d) for d in clean_number]),
            'variance': np.var([int(d) for d in clean_number]),
            'median': np.median([int(d) for d in clean_number]),
            'entropy': self._calculate_entropy(clean_number),
            'digit_range': max(int(d) for d in clean_number) - min(int(d) for d in clean_number),
            
            # Advanced features
            'first_digit': int(clean_number[0]) if clean_number else 0,
            'last_digit': int(clean_number[-1]) if clean_number else 0,
            'sum_digits': sum(int(d) for d in clean_number),
            'product_digits': self._product_digits(clean_number),
            'digital_root': self._digital_root(sum(int(d) for d in clean_number)),
            
            # Frequency features
            'even_count': sum(1 for d in clean_number if int(d) % 2 == 0),
            'odd_count': sum(1 for d in clean_number if int(d) % 2 == 1),
            'prime_count': sum(1 for d in clean_number if int(d) in [2, 3, 5, 7]),
            'zero_count': clean_number.count('0'),
            
            # Position features
            'middle_digit': int(clean_number[len(clean_number)//2]) if clean_number else 0,
            'second_digit': int(clean_number[1]) if len(clean_number) > 1 else 0,
            'second_last_digit': int(clean_number[-2]) if len(clean_number) > 1 else 0,
            
            # Ratio features
            'unique_ratio': len(set(clean_number)) / len(clean_number) if clean_number else 0,
            'even_ratio': sum(1 for d in clean_number if int(d) % 2 == 0) / len(clean_number) if clean_number else 0,
            'zero_ratio': clean_number.count('0') / len(clean_number) if clean_number else 0,
            'repeated_ratio': self._max_consecutive_same(clean_number) / len(clean_number) if clean_number else 0,
        }
        
        # Thêm features về prefix
        features.update(self._extract_prefix_features(clean_number))
        
        # Thêm features về suffix
        features.update(self._extract_suffix_features(clean_number))
        
        return features
    
    def _get_empty_features(self) -> Dict:
        """Trả về features mặc định cho số rỗng"""
        return {f'feature_{i}': 0.0 for i in range(50)}
    
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
    
    def _is_palindrome(self, digits: str) -> bool:
        """Kiểm tra palindrome"""
        return digits == digits[::-1] and len(digits) > 4
    
    def _is_ascending_sequence(self, digits: str) -> bool:
        """Kiểm tra dãy tăng dần"""
        if len(digits) < 4:
            return False
        
        for i in range(len(digits) - 3):
            if (int(digits[i+1]) == int(digits[i]) + 1 and
                int(digits[i+2]) == int(digits[i+1]) + 1 and
                int(digits[i+3]) == int(digits[i+2]) + 1):
                return True
        return False
    
    def _is_descending_sequence(self, digits: str) -> bool:
        """Kiểm tra dãy giảm dần"""
        if len(digits) < 4:
            return False
        
        for i in range(len(digits) - 3):
            if (int(digits[i+1]) == int(digits[i]) - 1 and
                int(digits[i+2]) == int(digits[i+1]) - 1 and
                int(digits[i+3]) == int(digits[i+2]) - 1):
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
    
    def _calculate_entropy(self, digits: str) -> float:
        """Tính entropy"""
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
    
    def _product_digits(self, digits: str) -> int:
        """Tính tích các chữ số"""
        if not digits:
            return 0
        
        product = 1
        for digit in digits:
            digit_val = int(digit)
            if digit_val == 0:
                return 0
            product *= digit_val
        
        return product
    
    def _digital_root(self, n: int) -> int:
        """Tính digital root"""
        while n >= 10:
            n = sum(int(digit) for digit in str(n))
        return n
    
    def _extract_prefix_features(self, digits: str) -> Dict:
        """Trích xuất features từ prefix"""
        features = {}
        
        if len(digits) >= 3:
            prefix_3 = digits[:3]
            
            # Vietnam mobile prefixes
            vn_mobile_prefixes = [
                '032', '033', '034', '035', '036', '037', '038', '039',
                '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
                '081', '082', '083', '084', '085', '086', '088', '089',
                '090', '091', '092', '093', '094', '096', '097', '098', '099'
            ]
            
            # Vietnam landline prefixes
            vn_landline_prefixes = ['024', '028', '222', '233', '234', '235', '236', '237', '238', '239']
            
            features['is_vn_mobile_prefix'] = int(prefix_3 in vn_mobile_prefixes)
            features['is_vn_landline_prefix'] = int(prefix_3 in vn_landline_prefixes)
            features['is_known_prefix'] = int(prefix_3 in vn_mobile_prefixes + vn_landline_prefixes)
            
            # Carrier features
            features['is_viettel_prefix'] = int(prefix_3 in ['032', '033', '034', '035', '036', '037', '038', '039', '096', '097', '098', '086'])
            features['is_vinaphone_prefix'] = int(prefix_3 in ['088', '091', '094', '083', '084', '085', '081', '082'])
            features['is_mobifone_prefix'] = int(prefix_3 in ['070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '090', '093', '089'])
            features['is_vietnamobile_prefix'] = int(prefix_3 in ['092', '056', '058'])
            features['is_gmobile_prefix'] = int(prefix_3 in ['099', '059'])
        else:
            features.update({
                'is_vn_mobile_prefix': 0,
                'is_vn_landline_prefix': 0,
                'is_known_prefix': 0,
                'is_viettel_prefix': 0,
                'is_vinaphone_prefix': 0,
                'is_mobifone_prefix': 0,
                'is_vietnamobile_prefix': 0,
                'is_gmobile_prefix': 0
            })
        
        return features
    
    def _extract_suffix_features(self, digits: str) -> Dict:
        """Trích xuất features từ suffix"""
        features = {}
        
        if len(digits) >= 3:
            suffix_3 = digits[-3:]
            
            features['suffix_sum'] = sum(int(d) for d in suffix_3)
            features['suffix_product'] = np.prod([int(d) for d in suffix_3])
            features['suffix_is_sequential'] = int(self._is_sequential(suffix_3))
            features['suffix_is_repeated'] = int(len(set(suffix_3)) == 1)
            features['suffix_entropy'] = self._calculate_entropy(suffix_3)
        else:
            features.update({
                'suffix_sum': 0,
                'suffix_product': 0,
                'suffix_is_sequential': 0,
                'suffix_is_repeated': 0,
                'suffix_entropy': 0
            })
        
        return features
    
    def _is_sequential(self, digits: str) -> bool:
        """Kiểm tra dãy số liên tiếp"""
        if len(digits) < 3:
            return False
        
        for i in range(len(digits) - 2):
            if (int(digits[i+1]) == int(digits[i]) + 1 and
                int(digits[i+2]) == int(digits[i+1]) + 1):
                return True
        return False
    
    def prepare_training_data(self, phone_numbers: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị dữ liệu training"""
        features_list = []
        
        for phone in phone_numbers:
            features = self.extract_features(phone)
            features_list.append(list(features.values()))
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Chuẩn hóa features
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        return X_scaled, y
    
    def transform_features(self, phone_numbers: List[str]) -> np.ndarray:
        """Transform features cho prediction"""
        if not self.is_fitted:
            raise ValueError("Feature extractor chưa được fit")
        
        features_list = []
        
        for phone in phone_numbers:
            features = self.extract_features(phone)
            features_list.append(list(features.values()))
        
        X = np.array(features_list)
        return self.scaler.transform(X)

class PhoneMLClassifier:
    """Machine Learning classifier cho phone validation"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        }
        
        self.ensemble_model = None
        self.feature_extractor = PhoneMLFeatureExtractor()
        self.is_trained = False
        self.model_path = "phone_ml_models"
    
    def create_ensemble_model(self):
        """Tạo ensemble model"""
        estimators = [(name, model) for name, model in self.models.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
    
    def train_models(self, phone_numbers: List[str], labels: List[int], test_size: float = 0.2):
        """Training các models"""
        logger.info("Preparing training data...")
        
        # Chuẩn bị dữ liệu
        X, y = self.feature_extractor.prepare_training_data(phone_numbers, labels)
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train từng model
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, F1: {results[name]['f1']:.4f}")
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        self.create_ensemble_model()
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1': f1_score(y_test, y_pred_ensemble),
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        logger.info(f"Ensemble - Accuracy: {results['ensemble']['accuracy']:.4f}, F1: {results['ensemble']['f1']:.4f}")
        
        self.is_trained = True
        
        return results
    
    def predict_single(self, phone_number: str) -> Dict:
        """Predict cho một số điện thoại"""
        if not self.is_trained:
            raise ValueError("Models chưa được train")
        
        # Transform features
        X = self.feature_extractor.transform_features([phone_number])
        
        # Predict với từng model
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
            
            predictions[name] = {
                'prediction': int(pred),
                'probability': float(proba) if proba is not None else None
            }
        
        # Ensemble prediction
        if self.ensemble_model:
            ensemble_pred = self.ensemble_model.predict(X)[0]
            ensemble_proba = self.ensemble_model.predict_proba(X)[0, 1]
            
            predictions['ensemble'] = {
                'prediction': int(ensemble_pred),
                'probability': float(ensemble_proba)
            }
        
        # Tính confidence score
        valid_probabilities = [p['probability'] for p in predictions.values() if p['probability'] is not None]
        confidence_score = np.mean(valid_probabilities) if valid_probabilities else 0.5
        
        # Tính consensus
        all_predictions = [p['prediction'] for p in predictions.values()]
        consensus = np.mean(all_predictions)
        
        return {
            'phone_number': phone_number,
            'individual_predictions': predictions,
            'consensus_prediction': int(consensus > 0.5),
            'confidence_score': confidence_score,
            'consensus_strength': abs(consensus - 0.5) * 2,
            'is_reliable': confidence_score > 0.7
        }
    
    def predict_batch(self, phone_numbers: List[str]) -> List[Dict]:
        """Predict cho nhiều số điện thoại"""
        return [self.predict_single(phone) for phone in phone_numbers]
    
    def save_models(self, path: Optional[str] = None):
        """Lưu models"""
        if path is None:
            path = self.model_path
        
        os.makedirs(path, exist_ok=True)
        
        # Lưu từng model
        for name, model in self.models.items():
            model_file = os.path.join(path, f"{name}.pkl")
            joblib.dump(model, model_file)
        
        # Lưu ensemble model
        if self.ensemble_model:
            ensemble_file = os.path.join(path, "ensemble.pkl")
            joblib.dump(self.ensemble_model, ensemble_file)
        
        # Lưu feature extractor
        feature_file = os.path.join(path, "feature_extractor.pkl")
        joblib.dump(self.feature_extractor, feature_file)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load models"""
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path {path} không tồn tại")
        
        # Load từng model
        for name in self.models.keys():
            model_file = os.path.join(path, f"{name}.pkl")
            if os.path.exists(model_file):
                self.models[name] = joblib.load(model_file)
        
        # Load ensemble model
        ensemble_file = os.path.join(path, "ensemble.pkl")
        if os.path.exists(ensemble_file):
            self.ensemble_model = joblib.load(ensemble_file)
        
        # Load feature extractor
        feature_file = os.path.join(path, "feature_extractor.pkl")
        if os.path.exists(feature_file):
            self.feature_extractor = joblib.load(feature_file)
        
        self.is_trained = True
        logger.info(f"Models loaded from {path}")
    
    def cross_validate(self, phone_numbers: List[str], labels: List[int], cv: int = 5) -> Dict:
        """Cross validation"""
        X, y = self.feature_extractor.prepare_training_data(phone_numbers, labels)
        
        cv_results = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_results[name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores.tolist()
            }
        
        return cv_results
    
    def get_feature_importance(self) -> Dict:
        """Lấy feature importance"""
        if not self.is_trained:
            raise ValueError("Models chưa được train")
        
        importance_results = {}
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                importance_results['random_forest'] = rf_model.feature_importances_.tolist()
        
        # Gradient Boosting feature importance
        if 'gradient_boost' in self.models:
            gb_model = self.models['gradient_boost']
            if hasattr(gb_model, 'feature_importances_'):
                importance_results['gradient_boost'] = gb_model.feature_importances_.tolist()
        
        return importance_results
    
    def generate_synthetic_data(self, n_valid: int = 1000, n_invalid: int = 1000) -> Tuple[List[str], List[int]]:
        """Tạo dữ liệu synthetic để training"""
        import random
        
        phone_numbers = []
        labels = []
        
        # Tạo số hợp lệ
        vn_prefixes = ['032', '033', '034', '035', '036', '037', '038', '039',
                      '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
                      '081', '082', '083', '084', '085', '086', '088', '089',
                      '090', '091', '092', '093', '094', '096', '097', '098', '099']
        
        for _ in range(n_valid):
            prefix = random.choice(vn_prefixes)
            suffix = ''.join(random.choices('0123456789', k=7))
            phone = f"0{prefix}{suffix}"
            phone_numbers.append(phone)
            labels.append(1)  # Valid
        
        # Tạo số không hợp lệ
        for _ in range(n_invalid):
            # Tạo các pattern không hợp lệ
            pattern_type = random.choice(['repeated', 'sequential', 'random_invalid', 'too_short', 'too_long'])
            
            if pattern_type == 'repeated':
                digit = random.choice('0123456789')
                phone = digit * random.randint(8, 12)
            elif pattern_type == 'sequential':
                start = random.randint(0, 6)
                phone = ''.join(str((start + i) % 10) for i in range(10))
            elif pattern_type == 'random_invalid':
                # Prefix không hợp lệ
                invalid_prefix = random.choice(['000', '111', '222', '333', '444', '555', '666', '777', '888', '999'])
                suffix = ''.join(random.choices('0123456789', k=7))
                phone = f"0{invalid_prefix}{suffix}"
            elif pattern_type == 'too_short':
                phone = ''.join(random.choices('0123456789', k=random.randint(3, 7)))
            else:  # too_long
                phone = ''.join(random.choices('0123456789', k=random.randint(13, 20)))
            
            phone_numbers.append(phone)
            labels.append(0)  # Invalid
        
        return phone_numbers, labels
