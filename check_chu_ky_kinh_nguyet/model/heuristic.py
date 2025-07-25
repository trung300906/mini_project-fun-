import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats as scipy_stats  # Tránh xung đột với module stats
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import lru_cache

class AdvancedCyclePredictor:
    def __init__(self, data_file: str, symptoms_dict_file: str):
        # Setup logging FIRST
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.symptoms_dict_file = symptoms_dict_file
        self.data = self.load_data()
        self.symptoms_dict = self.load_symptoms_dict()
        self.symptom_reverse_map = self.create_reverse_map()
        self.symptom_clusters = self.define_symptom_clusters()
        self._cached_stats = None
        
    def load_data(self) -> List[Dict]:
        """Tải dữ liệu lịch sử chu kỳ với error handling"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate data structure
                    return self._validate_data(data)
            return []
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading data: {e}")
            return []
    
    def _validate_data(self, data: List[Dict]) -> List[Dict]:
        """Validate and clean data entries"""
        valid_data = []
        for entry in data:
            if self._is_valid_entry(entry):
                valid_data.append(entry)
            else:
                self.logger.warning(f"Invalid entry skipped: {entry}")
        return valid_data
    
    def _is_valid_entry(self, entry: Dict) -> bool:
        required_fields = ['start_date', 'end_date']
        if not all(field in entry for field in required_fields):
            return False
        
        start = self._parse_date(entry['start_date'])
        end = self._parse_date(entry['end_date'])
        
        if not start or not end:
            return False
        
        return start <= end and self._validate_date_range(start, end)

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string với multiple format support"""
        formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        self.logger.warning(f"Cannot parse date: {date_str}")
        return None
    
    def _validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate date range is reasonable"""
        if start_date > end_date:
            return False
        
        period_length = (end_date - start_date).days
        # Period length should be between 1-10 days
        return 1 <= period_length <= 10
    
    def load_symptoms_dict(self) -> Dict:
        """Tải từ điển triệu chứng với error handling"""
        try:
            if os.path.exists(self.symptoms_dict_file):
                with open(self.symptoms_dict_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading symptoms dict: {e}")
            return {}
    
    @lru_cache(maxsize=1)
    def create_reverse_map(self) -> Dict[int, str]:
        """Tạo bản đồ ngược từ index sang tên triệu chứng với caching"""
        return {v: k for k, v in self.symptoms_dict.items()}
    
    def define_symptom_clusters(self) -> Dict[str, List[int]]:
        """Định nghĩa các cụm triệu chứng liên quan"""
        return {
            'pain': [0, 3, 7, 20, 21, 22, 51, 54, 55, 56, 57, 143, 184],
            'mood': [2, 13, 14, 15, 58, 59, 60, 61, 62, 63, 64, 510, 511, 512],
            'digestive': [1, 9, 11, 12, 23, 24, 81, 83, 84, 164, 165, 197, 199],
            'sensory': [25, 26, 27, 73, 74, 230, 231, 232, 233, 246, 247, 248],
            'skin': [5, 31, 32, 33, 34, 48, 49, 163, 251, 252, 417, 418, 419, 422]
        }
    
    def calculate_cycle_statistics(self, force_refresh: bool = False) -> Optional[Dict]:
        """Tính toán thống kê về chu kỳ với caching"""
        if self._cached_stats is not None and not force_refresh:
            return self._cached_stats
            
        if not self.data:
            return None
        
        # Use pandas for more efficient calculations
        df_data = []
        for entry in self.data:
            start_date = self._parse_date(entry['start_date'])
            end_date = self._parse_date(entry['end_date'])
            
            if not start_date or not end_date:
                continue
                
            period_length = (end_date - start_date).days
            cycle_length = entry.get('cycle_length', None)
            
            # Validate cycle length
            if cycle_length is not None:
                if not isinstance(cycle_length, (int, float)) or cycle_length <= 0 or cycle_length > 50:
                    cycle_length = None
            
            df_data.append({
                'start_date': start_date,
                'end_date': end_date,
                'period_length': period_length,
                'cycle_length': cycle_length if cycle_length and cycle_length > 0 else None
            })
        
        if not df_data:
            return None
            
        df = pd.DataFrame(df_data)
        
        # Calculate statistics
        cycle_lengths = df['cycle_length'].dropna()
        period_lengths = df['period_length']
        
        stats = {
            'avg_cycle': cycle_lengths.mean() if not cycle_lengths.empty else 28,
            'std_cycle': cycle_lengths.std() if len(cycle_lengths) > 1 else 3,
            'min_cycle': cycle_lengths.min() if not cycle_lengths.empty else 25,
            'max_cycle': cycle_lengths.max() if not cycle_lengths.empty else 32,
            'avg_period': period_lengths.mean() if not period_lengths.empty else 5,
            'std_period': period_lengths.std() if len(period_lengths) > 1 else 1,
            'cycle_count': len(self.data),
            'regularity_score': self._calculate_regularity_score(cycle_lengths)
        }
        
        self._cached_stats = stats
        return stats
    
    def _calculate_regularity_score(self, cycle_lengths: pd.Series) -> float:
        """Tính điểm số độ đều đặn của chu kỳ (0-1)"""
        if cycle_lengths.empty or len(cycle_lengths) < 2:
            return 0.5
        
        std = cycle_lengths.std()
        # Chuyển đổi std thành điểm từ 0-1 (std càng thấp điểm càng cao)
        return max(0, min(1, 1 - (std / 10)))
    
    def predict_next_cycle(self) -> Dict:
        """Dự đoán chu kỳ tiếp theo với heuristic nâng cao"""
        if not self.data:
            return {
                'prediction': None,
                'confidence': 'none',
                'message': 'Không có dữ liệu lịch sử'
            }
        
        stats = self.calculate_cycle_statistics()
        if not stats:
            return {
                'prediction': None,
                'confidence': 'none',
                'message': 'Dữ liệu không hợp lệ'
            }
        
        last_cycle = self.data[-1]
        last_start = self._parse_date(last_cycle['start_date'])
        last_end = self._parse_date(last_cycle['end_date'])
        
        if not last_start or not last_end:
            return {
                'prediction': None,
                'confidence': 'none',
                'message': 'Dữ liệu chu kỳ cuối không hợp lệ'
            }
        
        last_period_length = (last_end - last_start).days
        
        # Dự đoán dựa trên mô hình kết hợp
        next_start = self.combined_prediction(last_end, stats)
        
        # Dự đoán độ dài kỳ kinh
        predicted_period_length = self.predict_period_length(last_period_length, stats)
        next_end = next_start + timedelta(days=predicted_period_length)
        
        # Dự đoán triệu chứng
        predicted_symptoms = self.predict_symptoms()
        
        # Tính độ tin cậy
        confidence = self.calculate_confidence(stats)
        
        return {
            'predicted_start': next_start.strftime("%d-%m-%Y"),
            'predicted_end': next_end.strftime("%d-%m-%Y"),
            'predicted_period_length': predicted_period_length,
            'predicted_symptoms': predicted_symptoms,
            'confidence': confidence,
            'confidence_score': self._get_confidence_score(confidence),
            'stats': stats
        }

    def predict_symptoms(self) -> List[str]:
        """Dự đoán triệu chứng với machine learning approach"""
        if not self.data or not self.symptoms_dict:
            return []
        
        symptom_matrices = []
        for entry in self.data:
            if 'symptoms' in entry and entry['symptoms']:
                symptom_matrices.append(entry['symptoms'])
        
        if not symptom_matrices:
            return []
        
        symptom_array = np.array(symptom_matrices)
        symptom_probs = np.mean(symptom_array, axis=0)
        
        if len(symptom_matrices) > 1:
            recent_weight = 0.4
            last_symptoms = np.array(self.data[-1].get('symptoms', []))
            if len(last_symptoms) == len(symptom_probs):
                symptom_probs = (1 - recent_weight) * symptom_probs + recent_weight * last_symptoms
        
        threshold = max(0.3, np.percentile(symptom_probs, 75))
        high_prob_indices = np.where(symptom_probs >= threshold)[0]
        
        clustered_symptoms = self.get_related_symptoms(high_prob_indices.tolist())
        symptom_scores = [
            (sym, symptom_probs[self.symptoms_dict.get(sym, -1)]) 
            for sym in clustered_symptoms 
            if sym in self.symptoms_dict
        ]
        
        symptom_scores.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in symptom_scores[:8]]

    def _get_confidence_score(self, confidence: str) -> float:
        """Chuyển đổi confidence thành số"""
        confidence_map = {
            'none': 0.0,
            'low': 0.2,
            'medium': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        return confidence_map.get(confidence, 0.0)
    
    def combined_prediction(self, last_end: datetime, stats: Dict) -> datetime:
        """Kết hợp nhiều phương pháp dự đoán"""
        predictions = []
        
        # Phương pháp 1: Dùng độ dài chu kỳ trung bình
        avg_method = last_end + timedelta(days=int(stats['avg_cycle']))
        predictions.append(('avg', avg_method))
        
        # Phương pháp 2: Dùng chu kỳ gần nhất
        if self.data[-1].get('cycle_length'):
            last_method = last_end + timedelta(days=self.data[-1]['cycle_length'])
            predictions.append(('last', last_method))
        
        # Phương pháp 3: Điều chỉnh theo xu hướng
        trend_method = self.trend_adjusted_prediction(last_end, stats)
        predictions.append(('trend', trend_method))
        
        # Phương pháp 4: Machine learning approach (nếu có đủ dữ liệu)
        if stats['cycle_count'] >= 10:
            ml_method = self.ml_prediction(last_end, stats)
            if ml_method:
                predictions.append(('ml', ml_method))
        
        # Kết hợp trọng số
        weights = self.calculate_method_weights(stats, [p[0] for p in predictions])
        
        weighted_days = sum(
            weights.get(method, 0) * (pred_date - last_end).days 
            for method, pred_date in predictions
        )
        
        return last_end + timedelta(days=int(weighted_days))
    
    def ml_prediction(self, last_end: datetime, stats: Dict) -> Optional[datetime]:
        """Dự đoán sử dụng machine learning đơn giản"""
        try:
            # Chuẩn bị dữ liệu time series
            dates = []
            for entry in self.data[-10:]:  # Chỉ dùng 10 chu kỳ gần nhất
                start_date = self._parse_date(entry['start_date'])
                if start_date:
                    dates.append(start_date.timestamp())
            
            if len(dates) < 5:
                return None
            
            # Linear regression đơn giản
            x = np.arange(len(dates))
            y = np.array(dates)
            
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
            
            # Dự đoán điểm tiếp theo
            next_timestamp = slope * len(dates) + intercept
            predicted_date = datetime.fromtimestamp(next_timestamp)
            
            # Điều chỉnh dự đoán để hợp lý
            days_diff = (predicted_date - last_end).days
            if 20 <= days_diff <= 40:  # Chu kỳ hợp lý
                return predicted_date
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
        
        return None
    
    def calculate_method_weights(self, stats: Dict, available_methods: List[str]) -> Dict[str, float]:
        """Tính trọng số cho các phương pháp dự đoán dựa trên chất lượng dữ liệu"""
        cycle_count = stats['cycle_count']
        regularity = stats['regularity_score']
        
        weights = {}
        
        # Base weights
        if 'avg' in available_methods:
            weights['avg'] = 0.3 + (regularity * 0.2)
        
        if 'last' in available_methods:
            if cycle_count < 3:
                weights['last'] = 0.6
            else:
                weights['last'] = 0.2
        
        if 'trend' in available_methods:
            if cycle_count >= 5:
                weights['trend'] = 0.2 + (regularity * 0.1)
            else:
                weights['trend'] = 0.1
        
        if 'ml' in available_methods and cycle_count >= 10:
            weights['ml'] = 0.3 * regularity
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def trend_adjusted_prediction(self, last_end: datetime, stats: Dict) -> datetime:
        """Dự đoán điều chỉnh theo xu hướng với xử lý outliers"""
        if len(self.data) < 3:
            return last_end + timedelta(days=int(stats['avg_cycle']))
        
        # Lấy nhiều chu kỳ hơn nếu có
        num_cycles = min(6, len(self.data))
        cycles = self.data[-num_cycles:]
        
        try:
            starts = []
            for c in cycles:
                start_date = self._parse_date(c['start_date'])
                if start_date:
                    starts.append(start_date)
            
            if len(starts) < 2:
                return last_end + timedelta(days=int(stats['avg_cycle']))
            
            starts.sort()
            
            # Tính khoảng cách giữa các chu kỳ
            diffs = [(starts[i+1] - starts[i]).days for i in range(len(starts)-1)]
            
            # Remove outliers using IQR method
            if len(diffs) >= 4:
                q1, q3 = np.percentile(diffs, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                diffs = [d for d in diffs if lower_bound <= d <= upper_bound]
            
            trend = np.mean(diffs) if diffs else stats['avg_cycle']
            
            # Điều chỉnh theo độ lệch so với trung bình với smoothing
            deviation = trend - stats['avg_cycle']
            adjustment = deviation * (0.5 * stats['regularity_score'])  # Điều chỉnh dựa trên độ đều đặn
            
            predicted_cycle_length = stats['avg_cycle'] + adjustment
            return last_end + timedelta(days=int(predicted_cycle_length))
            
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Trend prediction failed: {e}")
            return last_end + timedelta(days=int(stats['avg_cycle']))
    
    def predict_period_length(self, last_period_length: int, stats: Dict) -> int:
        """Dự đoán độ dài kỳ kinh tiếp theo với adaptive weighting"""
        if stats['cycle_count'] < 3:
            return max(3, min(8, last_period_length))  # Giới hạn trong khoảng hợp lý
        
        # Adaptive weighting dựa trên regularity
        regularity = stats['regularity_score']
        last_weight = 0.4 + (0.3 * (1 - regularity))  # Ít đều đặn thì tin chu kỳ gần nhất hơn
        avg_weight = 1 - last_weight
        
        predicted = int(last_weight * last_period_length + avg_weight * stats['avg_period'])
        return max(3, min(8, predicted))  # Giới hạn trong khoảng hợp lý
    
    def symptoms_to_vector(self, symptoms: Union[List[str], List[int], str]) -> List[int]:
        """Convert symptoms to binary vector"""
        if not symptoms:
            return [0] * len(self.symptoms_dict)
        
        vector = [0] * len(self.symptoms_dict)
        
        if isinstance(symptoms, str):
            # If symptoms is a JSON string
            try:
                symptoms = json.loads(symptoms)
            except json.JSONDecodeError:
                symptoms = []
        
        if isinstance(symptoms, list):
            for symptom in symptoms:
                if isinstance(symptom, str) and symptom in self.symptoms_dict:
                    # Symptom name to index
                    idx = self.symptoms_dict[symptom]
                    if 0 <= idx < len(vector):
                        vector[idx] = 1
                elif isinstance(symptom, int) and 0 <= symptom < len(vector):
                    # Direct index
                    vector[symptom] = 1
        
        return vector
    
    def vector_to_symptoms(self, vector: List[int]) -> List[str]:
        """Convert binary vector to symptom names"""
        symptoms = []
        for idx, value in enumerate(vector):
            if value == 1 and idx in self.symptom_reverse_map:
                symptoms.append(self.symptom_reverse_map[idx])
        return symptoms
        """Dự đoán triệu chứng với machine learning approach"""
        if not self.data or not self.symptoms_dict:
            return []
        
        # Tạo ma trận triệu chứng
        symptom_matrices = []
        for entry in self.data:
            if 'symptoms' in entry and entry['symptoms']:
                symptom_matrices.append(entry['symptoms'])
        
        if not symptom_matrices:
            return []
        
        # Convert to numpy array for efficient computation
        symptom_array = np.array(symptom_matrices)
        
        # Tính probability cho mỗi triệu chứng
        symptom_probs = np.mean(symptom_array, axis=0)
        
        # Weighted probability với chu kỳ gần nhất
        if len(symptom_matrices) > 1:
            recent_weight = 0.4
            last_symptoms = np.array(self.data[-1].get('symptoms', []))
            if len(last_symptoms) == len(symptom_probs):
                symptom_probs = (1 - recent_weight) * symptom_probs + recent_weight * last_symptoms
        
        # Dynamic threshold dựa trên distribution
        threshold = max(0.3, np.percentile(symptom_probs, 75))
        
        # Lấy triệu chứng có probability cao
        high_prob_indices = np.where(symptom_probs >= threshold)[0]
        
        # Nhóm triệu chứng có liên quan
        clustered_symptoms = self.get_related_symptoms(high_prob_indices.tolist())
        
        # Sắp xếp theo probability và giới hạn số lượng
        symptom_scores = [(sym, symptom_probs[self.symptoms_dict.get(sym, -1)]) 
                         for sym in clustered_symptoms 
                         if sym in self.symptoms_dict]
        
        symptom_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [sym for sym, score in symptom_scores[:8]]  # Top 8 symptoms
    
    def get_related_symptoms(self, symptom_indices: List[int]) -> List[str]:
        """Mở rộng dựa trên các cụm triệu chứng liên quan với scoring"""
        expanded_symptoms = set(symptom_indices)
        
        # Thêm các triệu chứng trong cùng cụm với weighted approach
        cluster_scores = defaultdict(int)
        
        for idx in symptom_indices:
            for cluster_name, cluster_symptoms in self.symptom_clusters.items():
                if idx in cluster_symptoms:
                    cluster_scores[cluster_name] += 1
        
        # Chỉ thêm triệu chứng từ cluster có điểm cao
        for cluster_name, score in cluster_scores.items():
            if score >= 2:  # Cluster có ít nhất 2 triệu chứng được chọn
                expanded_symptoms.update(self.symptom_clusters[cluster_name])
        
        # Chuyển về tên triệu chứng với safe lookup
        valid_symptoms = []
        for idx in expanded_symptoms:
            if idx in self.symptom_reverse_map:
                valid_symptoms.append(self.symptom_reverse_map[idx])
            else:
                # Log unknown symptoms for debugging
                self.logger.debug(f"Unknown symptom index: {idx}")
        
        return valid_symptoms
    
    def calculate_confidence(self, stats: Dict) -> str:
        """Tính độ tin cậy với nhiều yếu tố"""
        cycle_count = stats['cycle_count']
        std_cycle = stats['std_cycle']
        regularity = stats['regularity_score']
        
        # Base confidence từ số lượng dữ liệu
        if cycle_count == 0:
            return 'none'
        elif cycle_count < 3:
            base_confidence = 1
        elif cycle_count < 6:
            base_confidence = 2
        elif cycle_count < 12:
            base_confidence = 3
        else:
            base_confidence = 4
        
        # Điều chỉnh dựa trên regularity
        regularity_bonus = int(regularity * 2)  # 0-2 bonus
        
        # Điều chỉnh dựa trên standard deviation
        if std_cycle < 2:
            std_bonus = 1
        elif std_cycle < 4:
            std_bonus = 0
        else:
            std_bonus = -1
        
        final_score = base_confidence + regularity_bonus + std_bonus
        
        if final_score <= 1:
            return 'low'
        elif final_score <= 3:
            return 'medium'
        elif final_score <= 5:
            return 'high'
        else:
            return 'very_high'
    
    def get_cycle_history(self) -> List[Dict]:
        """Chuẩn bị dữ liệu lịch sử cho hiển thị với additional insights"""
        history = []
        
        for i, entry in enumerate(self.data):
            start_date = self._parse_date(entry['start_date'])
            end_date = self._parse_date(entry['end_date'])
            
            if not start_date or not end_date:
                self.logger.warning(f"Invalid dates in history entry {i}")
                continue
                
            length = (end_date - start_date).days
            
            # Lấy tên triệu chứng với safe processing
            symptom_names = []
            if 'symptoms' in entry and entry['symptoms']:
                try:
                    symptoms_data = entry['symptoms']
                    if isinstance(symptoms_data, list):
                        # Handle both binary vector and symptom names/indices
                        if all(isinstance(x, (int, float)) for x in symptoms_data):
                            # Binary vector format
                            symptom_indices = [j for j, val in enumerate(symptoms_data) if val == 1]
                        else:
                            # List of names or indices
                            symptom_indices = []
                            for symptom in symptoms_data:
                                if isinstance(symptom, str) and symptom in self.symptoms_dict:
                                    symptom_indices.append(self.symptoms_dict[symptom])
                                elif isinstance(symptom, int):
                                    symptom_indices.append(symptom)
                    else:
                        # Other formats - convert using helper method
                        vector = self.symptoms_to_vector(symptoms_data)
                        symptom_indices = [j for j, val in enumerate(vector) if val == 1]
                    
                    # Convert indices to names safely
                    for idx in symptom_indices:
                        if idx in self.symptom_reverse_map:
                            symptom_names.append(self.symptom_reverse_map[idx])
                        
                except Exception as e:
                    self.logger.warning(f"Error processing symptoms in entry {i}: {e}")
            
            # Tính cycle length nếu có chu kỳ trước
            cycle_length = entry.get('cycle_length', 'N/A')
            if i > 0 and cycle_length == 'N/A':
                prev_start = self._parse_date(self.data[i-1]['start_date'])
                if prev_start:
                    cycle_length = (start_date - prev_start).days
            
            history.append({
                'start_date': entry['start_date'],
                'end_date': entry['end_date'],
                'length': length,
                'cycle_length': cycle_length,
                'symptoms': symptom_names,
                'symptom_count': len(symptom_names),
                'created_at': entry.get('created_at', ''),
                'cycle_index': i + 1
            })
        
        return history
    
    def invalidate_cache(self):
        """Xóa cache khi dữ liệu thay đổi"""
        self._cached_stats = None
        self.create_reverse_map.cache_clear()
    
    def get_insights(self) -> Dict:
        """Generate insights about cycle patterns"""
        stats = self.calculate_cycle_statistics()
        if not stats:
            return {}
        
        insights = {
            'cycle_regularity': 'đều đặn' if stats['regularity_score'] > 0.7 else 
                               'tương đối đều đặn' if stats['regularity_score'] > 0.4 else 'không đều đặn',
            'average_cycle_text': f"Chu kỳ trung bình: {stats['avg_cycle']:.1f} ngày",
            'period_length_text': f"Độ dài kinh nguyệt trung bình: {stats['avg_period']:.1f} ngày",
            'data_quality': 'tốt' if stats['cycle_count'] >= 6 else 
                           'trung bình' if stats['cycle_count'] >= 3 else 'cần thêm dữ liệu',
            'recommendation': self._get_recommendation(stats)
        }
        
        return insights
    
    def _get_recommendation(self, stats: Dict) -> str:
        """Generate personalized recommendations"""
        recommendations = []
        
        if stats['cycle_count'] < 6:
            recommendations.append("Tiếp tục ghi chép để có dự đoán chính xác hơn")
        
        if stats['std_cycle'] > 5:
            recommendations.append("Chu kỳ khá thay đổi, nên tham khảo ý kiến bác sĩ")
        
        if stats['regularity_score'] > 0.8:
            recommendations.append("Chu kỳ rất đều đặn, dự đoán có độ tin cậy cao")
        
        return "; ".join(recommendations) if recommendations else "Chu kỳ bình thường"