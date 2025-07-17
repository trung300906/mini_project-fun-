#!/usr/bin/env python3
"""
Report generator for phone validation results
"""

import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Tạo báo cáo cho phone validation"""
    
    def __init__(self):
        self.report_templates = {
            'single': self._get_single_report_template(),
            'batch': self._get_batch_report_template(),
            'analytics': self._get_analytics_report_template()
        }
    
    def generate_single_report(self, result: Dict) -> str:
        """Tạo báo cáo cho một số điện thoại"""
        
        # Tính toán thêm thông tin
        ml_features = result.get('ml_predictions', {})
        ai_analysis = result.get('ai_analysis', {})
        
        # Xác định mức độ tin cậy
        confidence_text = self._get_confidence_text(result['confidence_score'])
        risk_text = self._get_risk_text(result['risk_score'])
        
        # Icon cho trạng thái
        status_icon = self._get_status_icon(result['status'])
        
        # Tạo context cho template
        context = {
            'result': result,
            'confidence_text': confidence_text,
            'risk_text': risk_text,
            'status_icon': status_icon,
            'ml_features': ml_features,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': result.get('processing_time', 0)
        }
        
        # Render template
        template = Template(self.report_templates['single'])
        return template.render(**context)
    
    def generate_batch_report(self, results: List[Dict]) -> str:
        """Tạo báo cáo cho batch validation"""
        if not results:
            return "Không có kết quả để tạo báo cáo."
        
        # Tính toán thống kê
        stats = self._calculate_batch_statistics(results)
        
        # Phân tích theo nhà mạng
        carrier_analysis = self._analyze_by_carrier(results)
        
        # Phân tích theo quốc gia
        country_analysis = self._analyze_by_country(results)
        
        # Phân tích theo thời gian
        time_analysis = self._analyze_by_time(results)
        
        # Tìm pattern thú vị
        patterns = self._find_interesting_patterns(results)
        
        # Khuyến nghị
        recommendations = self._generate_recommendations(results, stats)
        
        # Context cho template
        context = {
            'results': results,
            'stats': stats,
            'carrier_analysis': carrier_analysis,
            'country_analysis': country_analysis,
            'time_analysis': time_analysis,
            'patterns': patterns,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Render template
        template = Template(self.report_templates['batch'])
        return template.render(**context)
    
    def generate_analytics_report(self, results: List[Dict], period: str = '7d') -> str:
        """Tạo báo cáo phân tích chi tiết"""
        if not results:
            return "Không có dữ liệu để phân tích."
        
        # Phân tích trend
        trend_analysis = self._analyze_trends(results, period)
        
        # Phân tích ML performance
        ml_performance = self._analyze_ml_performance(results)
        
        # Phân tích AI confidence
        ai_confidence = self._analyze_ai_confidence(results)
        
        # Phân tích fraud detection
        fraud_analysis = self._analyze_fraud_patterns(results)
        
        # Phân tích validation methods
        method_analysis = self._analyze_validation_methods(results)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(results)
        
        # Context cho template
        context = {
            'results': results,
            'period': period,
            'trend_analysis': trend_analysis,
            'ml_performance': ml_performance,
            'ai_confidence': ai_confidence,
            'fraud_analysis': fraud_analysis,
            'method_analysis': method_analysis,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Render template
        template = Template(self.report_templates['analytics'])
        return template.render(**context)
    
    def export_to_csv(self, results: List[Dict], filename: str, detailed: bool = False):
        """Export kết quả ra CSV"""
        if not results:
            logger.warning("No results to export")
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if detailed:
                    fieldnames = [
                        'phone_number', 'status', 'confidence_score', 'risk_score',
                        'country', 'carrier_name', 'line_type', 'is_valid_format',
                        'is_possible', 'ai_confidence', 'fraud_probability',
                        'processing_time', 'timestamp'
                    ]
                    
                    # Thêm ML features
                    if results[0].get('ml_predictions'):
                        ml_features = results[0]['ml_predictions']
                        fieldnames.extend([f'ml_{key}' for key in ml_features.keys()])
                    
                    # Thêm validation methods
                    fieldnames.append('validation_methods')
                else:
                    fieldnames = [
                        'phone_number', 'status', 'confidence_score', 'risk_score',
                        'country', 'carrier_name', 'line_type', 'timestamp'
                    ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {}
                    
                    # Basic fields
                    for field in fieldnames:
                        if field.startswith('ml_'):
                            ml_key = field[3:]  # Remove 'ml_' prefix
                            ml_predictions = result.get('ml_predictions', {})
                            row[field] = ml_predictions.get(ml_key, '')
                        elif field == 'validation_methods':
                            row[field] = ', '.join(result.get('validation_methods', []))
                        elif field == 'timestamp':
                            timestamp = result.get('timestamp')
                            if timestamp:
                                row[field] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                row[field] = ''
                        else:
                            row[field] = result.get(field, '')
                    
                    writer.writerow(row)
            
            logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def export_to_excel(self, results: List[Dict], filename: str):
        """Export kết quả ra Excel với multiple sheets"""
        if not results:
            logger.warning("No results to export")
            return
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main results sheet
                df_main = pd.DataFrame(results)
                df_main.to_excel(writer, sheet_name='Results', index=False)
                
                # Statistics sheet
                stats = self._calculate_batch_statistics(results)
                df_stats = pd.DataFrame([stats])
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Carrier analysis sheet
                carrier_analysis = self._analyze_by_carrier(results)
                df_carrier = pd.DataFrame(carrier_analysis)
                df_carrier.to_excel(writer, sheet_name='Carrier_Analysis', index=False)
                
                # Country analysis sheet
                country_analysis = self._analyze_by_country(results)
                df_country = pd.DataFrame(country_analysis)
                df_country.to_excel(writer, sheet_name='Country_Analysis', index=False)
            
            logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
    
    def export_to_json(self, results: List[Dict], filename: str, pretty: bool = True):
        """Export kết quả ra JSON"""
        if not results:
            logger.warning("No results to export")
            return
        
        try:
            # Convert datetime objects to string
            serializable_results = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, datetime):
                        serializable_result[key] = value.isoformat()
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                if pretty:
                    json.dump(serializable_results, jsonfile, indent=2, ensure_ascii=False)
                else:
                    json.dump(serializable_results, jsonfile, ensure_ascii=False)
            
            logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def create_visualizations(self, results: List[Dict], output_dir: str = "reports"):
        """Tạo visualizations"""
        if not results:
            logger.warning("No results to visualize")
            return
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Status distribution
            self._create_status_distribution_chart(results, output_dir)
            
            # Confidence score distribution
            self._create_confidence_distribution_chart(results, output_dir)
            
            # Risk score distribution
            self._create_risk_distribution_chart(results, output_dir)
            
            # Carrier analysis
            self._create_carrier_analysis_chart(results, output_dir)
            
            # Country analysis
            self._create_country_analysis_chart(results, output_dir)
            
            # Processing time analysis
            self._create_processing_time_chart(results, output_dir)
            
            logger.info(f"Visualizations created in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict:
        """Tính toán thống kê batch"""
        if not results:
            return {}
        
        total_count = len(results)
        
        # Đếm theo status
        status_counts = {}
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Tính trung bình
        confidence_scores = [r.get('confidence_score', 0) for r in results]
        risk_scores = [r.get('risk_score', 0) for r in results]
        processing_times = [r.get('processing_time', 0) for r in results]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        avg_risk = sum(risk_scores) / len(risk_scores)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        return {
            'total_count': total_count,
            'valid_count': status_counts.get('valid', 0),
            'invalid_count': status_counts.get('invalid', 0),
            'suspicious_count': status_counts.get('suspicious', 0),
            'unknown_count': status_counts.get('unknown', 0),
            'valid_percentage': status_counts.get('valid', 0) / total_count * 100,
            'invalid_percentage': status_counts.get('invalid', 0) / total_count * 100,
            'suspicious_percentage': status_counts.get('suspicious', 0) / total_count * 100,
            'unknown_percentage': status_counts.get('unknown', 0) / total_count * 100,
            'avg_confidence': avg_confidence,
            'avg_risk': avg_risk,
            'avg_processing_time': avg_processing_time,
            'max_confidence': max(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_risk': max(risk_scores),
            'min_risk': min(risk_scores)
        }
    
    def _analyze_by_carrier(self, results: List[Dict]) -> List[Dict]:
        """Phân tích theo nhà mạng"""
        carrier_stats = {}
        
        for result in results:
            carrier = result.get('carrier_name', 'Unknown')
            
            if carrier not in carrier_stats:
                carrier_stats[carrier] = {
                    'carrier': carrier,
                    'count': 0,
                    'valid_count': 0,
                    'invalid_count': 0,
                    'suspicious_count': 0,
                    'unknown_count': 0,
                    'total_confidence': 0,
                    'total_risk': 0
                }
            
            stats = carrier_stats[carrier]
            stats['count'] += 1
            
            status = result.get('status', 'unknown')
            stats[f'{status}_count'] += 1
            
            stats['total_confidence'] += result.get('confidence_score', 0)
            stats['total_risk'] += result.get('risk_score', 0)
        
        # Tính percentages và averages
        for carrier, stats in carrier_stats.items():
            count = stats['count']
            stats['valid_percentage'] = stats['valid_count'] / count * 100
            stats['invalid_percentage'] = stats['invalid_count'] / count * 100
            stats['suspicious_percentage'] = stats['suspicious_count'] / count * 100
            stats['unknown_percentage'] = stats['unknown_count'] / count * 100
            stats['avg_confidence'] = stats['total_confidence'] / count
            stats['avg_risk'] = stats['total_risk'] / count
        
        return list(carrier_stats.values())
    
    def _analyze_by_country(self, results: List[Dict]) -> List[Dict]:
        """Phân tích theo quốc gia"""
        country_stats = {}
        
        for result in results:
            country = result.get('country', 'Unknown')
            
            if country not in country_stats:
                country_stats[country] = {
                    'country': country,
                    'count': 0,
                    'valid_count': 0,
                    'invalid_count': 0,
                    'suspicious_count': 0,
                    'unknown_count': 0,
                    'total_confidence': 0,
                    'total_risk': 0
                }
            
            stats = country_stats[country]
            stats['count'] += 1
            
            status = result.get('status', 'unknown')
            stats[f'{status}_count'] += 1
            
            stats['total_confidence'] += result.get('confidence_score', 0)
            stats['total_risk'] += result.get('risk_score', 0)
        
        # Tính percentages và averages
        for country, stats in country_stats.items():
            count = stats['count']
            stats['valid_percentage'] = stats['valid_count'] / count * 100
            stats['invalid_percentage'] = stats['invalid_count'] / count * 100
            stats['suspicious_percentage'] = stats['suspicious_count'] / count * 100
            stats['unknown_percentage'] = stats['unknown_count'] / count * 100
            stats['avg_confidence'] = stats['total_confidence'] / count
            stats['avg_risk'] = stats['total_risk'] / count
        
        return list(country_stats.values())
    
    def _analyze_by_time(self, results: List[Dict]) -> Dict:
        """Phân tích theo thời gian"""
        if not results:
            return {}
        
        # Group by hour
        hourly_stats = {}
        
        for result in results:
            timestamp = result.get('timestamp')
            if timestamp:
                hour = timestamp.hour
                
                if hour not in hourly_stats:
                    hourly_stats[hour] = {
                        'hour': hour,
                        'count': 0,
                        'valid_count': 0,
                        'invalid_count': 0,
                        'total_confidence': 0,
                        'total_risk': 0
                    }
                
                stats = hourly_stats[hour]
                stats['count'] += 1
                
                if result.get('status') == 'valid':
                    stats['valid_count'] += 1
                elif result.get('status') == 'invalid':
                    stats['invalid_count'] += 1
                
                stats['total_confidence'] += result.get('confidence_score', 0)
                stats['total_risk'] += result.get('risk_score', 0)
        
        # Tính averages
        for hour, stats in hourly_stats.items():
            count = stats['count']
            stats['avg_confidence'] = stats['total_confidence'] / count
            stats['avg_risk'] = stats['total_risk'] / count
            stats['valid_percentage'] = stats['valid_count'] / count * 100
            stats['invalid_percentage'] = stats['invalid_count'] / count * 100
        
        return {
            'hourly_stats': list(hourly_stats.values()),
            'peak_hour': max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['count']) if hourly_stats else None,
            'total_processing_time': sum(r.get('processing_time', 0) for r in results)
        }
    
    def _find_interesting_patterns(self, results: List[Dict]) -> List[Dict]:
        """Tìm patterns thú vị"""
        patterns = []
        
        # Pattern 1: Carrier với tỷ lệ invalid cao
        carrier_analysis = self._analyze_by_carrier(results)
        for carrier_stats in carrier_analysis:
            if carrier_stats['invalid_percentage'] > 50 and carrier_stats['count'] > 10:
                patterns.append({
                    'type': 'high_invalid_carrier',
                    'description': f"Carrier '{carrier_stats['carrier']}' có tỷ lệ invalid cao ({carrier_stats['invalid_percentage']:.1f}%)",
                    'severity': 'high' if carrier_stats['invalid_percentage'] > 80 else 'medium',
                    'data': carrier_stats
                })
        
        # Pattern 2: Confidence thấp bất thường
        low_confidence_results = [r for r in results if r.get('confidence_score', 0) < 0.3]
        if len(low_confidence_results) > len(results) * 0.2:  # > 20%
            patterns.append({
                'type': 'low_confidence_pattern',
                'description': f"Có {len(low_confidence_results)} số ({len(low_confidence_results)/len(results)*100:.1f}%) có confidence thấp bất thường",
                'severity': 'medium',
                'data': {'count': len(low_confidence_results), 'percentage': len(low_confidence_results)/len(results)*100}
            })
        
        # Pattern 3: Processing time bất thường
        processing_times = [r.get('processing_time', 0) for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        slow_results = [r for r in results if r.get('processing_time', 0) > avg_processing_time * 3]
        
        if len(slow_results) > 0:
            patterns.append({
                'type': 'slow_processing',
                'description': f"Có {len(slow_results)} số xử lý chậm bất thường (>3x thời gian trung bình)",
                'severity': 'low',
                'data': {'count': len(slow_results), 'avg_time': avg_processing_time}
            })
        
        return patterns
    
    def _generate_recommendations(self, results: List[Dict], stats: Dict) -> List[str]:
        """Tạo khuyến nghị"""
        recommendations = []
        
        # Recommendation dựa trên tỷ lệ invalid
        if stats.get('invalid_percentage', 0) > 30:
            recommendations.append("Tỷ lệ số invalid cao (>30%). Nên kiểm tra nguồn dữ liệu và cải thiện quy trình validation.")
        
        # Recommendation dựa trên confidence
        if stats.get('avg_confidence', 0) < 0.6:
            recommendations.append("Confidence trung bình thấp (<60%). Nên cải thiện thuật toán ML và thêm data training.")
        
        # Recommendation dựa trên processing time
        if stats.get('avg_processing_time', 0) > 2.0:
            recommendations.append("Thời gian xử lý chậm (>2s). Nên tối ưu hóa thuật toán và cách sử dụng API.")
        
        # Recommendation dựa trên carrier diversity
        carrier_analysis = self._analyze_by_carrier(results)
        if len(carrier_analysis) < 3:
            recommendations.append("Ít nhà mạng được phát hiện. Nên mở rộng database carrier và cải thiện carrier detection.")
        
        # Recommendation dựa trên suspicious rate
        if stats.get('suspicious_percentage', 0) > 20:
            recommendations.append("Tỷ lệ suspicious cao (>20%). Nên review và fine-tune các thuật toán phát hiện suspicious.")
        
        return recommendations
    
    def _analyze_trends(self, results: List[Dict], period: str) -> Dict:
        """Phân tích trend"""
        # Placeholder implementation
        return {
            'trend_direction': 'stable',
            'confidence_trend': 'improving',
            'risk_trend': 'stable',
            'volume_trend': 'increasing'
        }
    
    def _analyze_ml_performance(self, results: List[Dict]) -> Dict:
        """Phân tích performance ML"""
        ml_predictions = [r.get('ml_predictions', {}) for r in results]
        
        if not ml_predictions or not ml_predictions[0]:
            return {'error': 'No ML predictions found'}
        
        # Tính accuracy của từng model
        model_performance = {}
        
        for result in results:
            ml_pred = result.get('ml_predictions', {})
            actual_status = result.get('status', 'unknown')
            
            for model_name, prediction in ml_pred.items():
                if model_name not in model_performance:
                    model_performance[model_name] = {
                        'correct': 0,
                        'total': 0,
                        'accuracy': 0.0
                    }
                
                model_performance[model_name]['total'] += 1
                
                # Simplified accuracy calculation
                if prediction.get('prediction') == 1 and actual_status == 'valid':
                    model_performance[model_name]['correct'] += 1
                elif prediction.get('prediction') == 0 and actual_status in ['invalid', 'suspicious']:
                    model_performance[model_name]['correct'] += 1
        
        # Tính accuracy percentages
        for model_name, perf in model_performance.items():
            if perf['total'] > 0:
                perf['accuracy'] = perf['correct'] / perf['total'] * 100
        
        return model_performance
    
    def _analyze_ai_confidence(self, results: List[Dict]) -> Dict:
        """Phân tích AI confidence"""
        ai_confidences = [r.get('ai_confidence', 0) for r in results]
        
        if not ai_confidences:
            return {'error': 'No AI confidence data found'}
        
        return {
            'average_confidence': sum(ai_confidences) / len(ai_confidences),
            'max_confidence': max(ai_confidences),
            'min_confidence': min(ai_confidences),
            'high_confidence_count': len([c for c in ai_confidences if c > 0.8]),
            'low_confidence_count': len([c for c in ai_confidences if c < 0.5])
        }
    
    def _analyze_fraud_patterns(self, results: List[Dict]) -> Dict:
        """Phân tích fraud patterns"""
        fraud_probabilities = [r.get('fraud_probability', 0) for r in results]
        
        if not fraud_probabilities:
            return {'error': 'No fraud probability data found'}
        
        high_fraud_count = len([p for p in fraud_probabilities if p > 0.7])
        medium_fraud_count = len([p for p in fraud_probabilities if 0.3 < p <= 0.7])
        low_fraud_count = len([p for p in fraud_probabilities if p <= 0.3])
        
        return {
            'average_fraud_probability': sum(fraud_probabilities) / len(fraud_probabilities),
            'high_fraud_count': high_fraud_count,
            'medium_fraud_count': medium_fraud_count,
            'low_fraud_count': low_fraud_count,
            'high_fraud_percentage': high_fraud_count / len(results) * 100,
            'fraud_risk_level': 'high' if high_fraud_count > len(results) * 0.2 else 'medium' if high_fraud_count > len(results) * 0.1 else 'low'
        }
    
    def _analyze_validation_methods(self, results: List[Dict]) -> Dict:
        """Phân tích validation methods"""
        method_usage = {}
        
        for result in results:
            methods = result.get('validation_methods', [])
            for method in methods:
                method_usage[method] = method_usage.get(method, 0) + 1
        
        total_results = len(results)
        method_percentages = {
            method: count / total_results * 100
            for method, count in method_usage.items()
        }
        
        return {
            'method_usage': method_usage,
            'method_percentages': method_percentages,
            'most_used_method': max(method_usage, key=method_usage.get) if method_usage else None,
            'total_methods': len(method_usage)
        }
    
    def _calculate_quality_metrics(self, results: List[Dict]) -> Dict:
        """Tính quality metrics"""
        if not results:
            return {}
        
        # Consistency score
        consistency_scores = []
        for result in results:
            ml_predictions = result.get('ml_predictions', {})
            if ml_predictions:
                predictions = [p.get('prediction', 0) for p in ml_predictions.values()]
                if predictions:
                    consistency = 1 - (max(predictions) - min(predictions))
                    consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # Reliability score
        high_confidence_count = len([r for r in results if r.get('confidence_score', 0) > 0.8])
        reliability_score = high_confidence_count / len(results) * 100
        
        # Coverage score
        complete_results = len([r for r in results if r.get('carrier_name') and r.get('country')])
        coverage_score = complete_results / len(results) * 100
        
        return {
            'consistency_score': avg_consistency * 100,
            'reliability_score': reliability_score,
            'coverage_score': coverage_score,
            'overall_quality': (avg_consistency * 100 + reliability_score + coverage_score) / 3
        }
    
    def _create_status_distribution_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart phân bố status"""
        try:
            status_counts = {}
            for result in results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            plt.figure(figsize=(10, 6))
            plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            plt.title('Phone Number Status Distribution')
            plt.savefig(f'{output_dir}/status_distribution.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating status distribution chart: {e}")
    
    def _create_confidence_distribution_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart phân bố confidence"""
        try:
            confidence_scores = [r.get('confidence_score', 0) for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Score Distribution')
            plt.savefig(f'{output_dir}/confidence_distribution.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating confidence distribution chart: {e}")
    
    def _create_risk_distribution_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart phân bố risk"""
        try:
            risk_scores = [r.get('risk_score', 0) for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.hist(risk_scores, bins=20, alpha=0.7, color='red')
            plt.xlabel('Risk Score')
            plt.ylabel('Frequency')
            plt.title('Risk Score Distribution')
            plt.savefig(f'{output_dir}/risk_distribution.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating risk distribution chart: {e}")
    
    def _create_carrier_analysis_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart phân tích carrier"""
        try:
            carrier_counts = {}
            for result in results:
                carrier = result.get('carrier_name', 'Unknown')
                carrier_counts[carrier] = carrier_counts.get(carrier, 0) + 1
            
            # Chỉ lấy top 10 carriers
            top_carriers = sorted(carrier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            plt.figure(figsize=(12, 8))
            carriers, counts = zip(*top_carriers)
            plt.bar(carriers, counts)
            plt.xlabel('Carrier')
            plt.ylabel('Count')
            plt.title('Top 10 Carriers by Phone Number Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/carrier_analysis.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating carrier analysis chart: {e}")
    
    def _create_country_analysis_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart phân tích country"""
        try:
            country_counts = {}
            for result in results:
                country = result.get('country', 'Unknown')
                country_counts[country] = country_counts.get(country, 0) + 1
            
            # Chỉ lấy top 10 countries
            top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            plt.figure(figsize=(12, 8))
            countries, counts = zip(*top_countries)
            plt.bar(countries, counts)
            plt.xlabel('Country')
            plt.ylabel('Count')
            plt.title('Top 10 Countries by Phone Number Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/country_analysis.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating country analysis chart: {e}")
    
    def _create_processing_time_chart(self, results: List[Dict], output_dir: str):
        """Tạo chart thời gian xử lý"""
        try:
            processing_times = [r.get('processing_time', 0) for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.hist(processing_times, bins=20, alpha=0.7, color='green')
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Processing Time Distribution')
            plt.savefig(f'{output_dir}/processing_time.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating processing time chart: {e}")
    
    def _get_confidence_text(self, confidence_score: float) -> str:
        """Chuyển đổi confidence score thành text"""
        if confidence_score >= 0.9:
            return "RẤT CAO"
        elif confidence_score >= 0.8:
            return "CAO"
        elif confidence_score >= 0.7:
            return "TRUNG BÌNH CAO"
        elif confidence_score >= 0.6:
            return "TRUNG BÌNH"
        elif confidence_score >= 0.5:
            return "TRUNG BÌNH THẤP"
        else:
            return "THẤP"
    
    def _get_risk_text(self, risk_score: float) -> str:
        """Chuyển đổi risk score thành text"""
        if risk_score >= 0.8:
            return "RẤT CAO"
        elif risk_score >= 0.6:
            return "CAO"
        elif risk_score >= 0.4:
            return "TRUNG BÌNH"
        elif risk_score >= 0.2:
            return "THẤP"
        else:
            return "RẤT THẤP"
    
    def _get_status_icon(self, status: str) -> str:
        """Lấy icon cho status"""
        icons = {
            'valid': '✅',
            'invalid': '❌',
            'suspicious': '⚠️',
            'unknown': '❓'
        }
        return icons.get(status, '❓')
    
    def _get_single_report_template(self) -> str:
        """Template cho single report"""
        return '''
{{ status_icon }} === BÁO CÁO KIỂM TRA SỐ ĐIỆN THOẠI ===
📞 Số điện thoại: {{ result.phone_number }}
📊 Trạng thái: {{ result.status.upper() }}
🎯 Độ tin cậy: {{ "%.1f" | format(result.confidence_score * 100) }}% ({{ confidence_text }})
⚠️ Mức rủi ro: {{ "%.1f" | format(result.risk_score * 100) }}% ({{ risk_text }})
🚨 Khả năng gian lận: {{ "%.1f" | format(result.fraud_probability * 100) }}%
🤖 AI Confidence: {{ "%.1f" | format(result.ai_confidence * 100) }}%

=== THÔNG TIN CHI TIẾT ===
🌍 Quốc gia: {{ result.country }}
📡 Nhà mạng: {{ result.carrier_name }}
📍 Vị trí: {{ result.location }}
📱 Loại đường truyền: {{ result.line_type }}
✅ Định dạng hợp lệ: {{ "Có" if result.is_valid_format else "Không" }}
🔍 Có thể tồn tại: {{ "Có" if result.is_possible else "Không" }}

=== PHÂN TÍCH MACHINE LEARNING ===
{% if ml_features %}
{% for model_name, prediction in ml_features.items() %}
🔧 {{ model_name }}: {{ "Valid" if prediction.prediction == 1 else "Invalid" }} ({{ "%.1f" | format(prediction.probability * 100) }}%)
{% endfor %}
{% endif %}

=== PHÂN TÍCH AI ===
{% if ai_analysis %}
🧠 Neural Network: {{ "%.1f" | format(ai_analysis.neural_network_prediction * 100) }}%
🔮 Deep Learning: {{ "%.1f" | format(ai_analysis.deep_learning_confidence * 100) }}%
🎯 Ensemble Score: {{ "%.1f" | format(ai_analysis.ensemble_score * 100) }}%
{% endif %}

=== PHƯƠNG PHÁP KIỂM TRA ===
{% for method in result.validation_methods %}
✓ {{ method.replace('_', ' ').title() }}
{% endfor %}

=== THỜI GIAN ===
🕐 Kiểm tra lúc: {{ timestamp }}
⏱️ Thời gian xử lý: {{ "%.3f" | format(processing_time) }}s

=== KẾT LUẬN ===
{% if result.status == 'valid' %}
🟢 Số điện thoại này CÓ KHẢ NĂNG CAO là thật và đang hoạt động.
{% elif result.status == 'suspicious' %}
🟡 Số điện thoại này CẦN THẬN TRỌNG - có dấu hiệu nghi ngờ.
{% elif result.status == 'unknown' %}
🔵 Không thể xác định chính xác - cần kiểm tra thêm.
{% else %}
🔴 Số điện thoại này CÓ KHẢ NĂNG CAO là GIẢ hoặc không hợp lệ.
{% endif %}
'''
    
    def _get_batch_report_template(self) -> str:
        """Template cho batch report"""
        return '''
🔍 === BÁO CÁO KIỂM TRA HÀNG LOẠT ===
📊 Tổng số kiểm tra: {{ stats.total_count }}
🕐 Thời gian tạo báo cáo: {{ timestamp }}

=== THỐNG KÊ TỔNG QUAN ===
✅ Hợp lệ: {{ stats.valid_count }} ({{ "%.1f" | format(stats.valid_percentage) }}%)
❌ Không hợp lệ: {{ stats.invalid_count }} ({{ "%.1f" | format(stats.invalid_percentage) }}%)
⚠️ Nghi ngờ: {{ stats.suspicious_count }} ({{ "%.1f" | format(stats.suspicious_percentage) }}%)
❓ Không rõ: {{ stats.unknown_count }} ({{ "%.1f" | format(stats.unknown_percentage) }}%)

=== THỐNG KÊ CHẤT LƯỢNG ===
🎯 Độ tin cậy trung bình: {{ "%.1f" | format(stats.avg_confidence * 100) }}%
⚠️ Mức rủi ro trung bình: {{ "%.1f" | format(stats.avg_risk * 100) }}%
⏱️ Thời gian xử lý trung bình: {{ "%.3f" | format(stats.avg_processing_time) }}s

=== PHÂN TÍCH THEO NHÀ MẠNG ===
{% for carrier in carrier_analysis[:5] %}
📡 {{ carrier.carrier }}: {{ carrier.count }} số ({{ "%.1f" | format(carrier.valid_percentage) }}% hợp lệ)
{% endfor %}

=== PHÂN TÍCH THEO QUỐC GIA ===
{% for country in country_analysis[:5] %}
🌍 {{ country.country }}: {{ country.count }} số ({{ "%.1f" | format(country.valid_percentage) }}% hợp lệ)
{% endfor %}

=== PATTERNS THÚ VỊ ===
{% for pattern in patterns %}
🔍 {{ pattern.description }}
{% endfor %}

=== KHUYẾN NGHỊ ===
{% for recommendation in recommendations %}
💡 {{ recommendation }}
{% endfor %}
'''
    
    def _get_analytics_report_template(self) -> str:
        """Template cho analytics report"""
        return '''
📈 === BÁO CÁO PHÂN TÍCH CHI TIẾT ===
📊 Thời gian: {{ timestamp }}
📅 Kỳ phân tích: {{ period }}

=== CHẤT LƯỢNG HỆ THỐNG ===
🎯 Consistency Score: {{ "%.1f" | format(quality_metrics.consistency_score) }}%
🔒 Reliability Score: {{ "%.1f" | format(quality_metrics.reliability_score) }}%
📊 Coverage Score: {{ "%.1f" | format(quality_metrics.coverage_score) }}%
⭐ Overall Quality: {{ "%.1f" | format(quality_metrics.overall_quality) }}%

=== HIỆU SUẤT MACHINE LEARNING ===
{% for model_name, performance in ml_performance.items() %}
🤖 {{ model_name }}: {{ "%.1f" | format(performance.accuracy) }}% accuracy ({{ performance.correct }}/{{ performance.total }})
{% endfor %}

=== PHÂN TÍCH AI CONFIDENCE ===
🧠 AI Confidence trung bình: {{ "%.1f" | format(ai_confidence.average_confidence * 100) }}%
📈 Confidence cao (>80%): {{ ai_confidence.high_confidence_count }} số
📉 Confidence thấp (<50%): {{ ai_confidence.low_confidence_count }} số

=== PHÂN TÍCH FRAUD DETECTION ===
🚨 Fraud Risk Level: {{ fraud_analysis.fraud_risk_level.upper() }}
📊 High Risk: {{ fraud_analysis.high_fraud_count }} số ({{ "%.1f" | format(fraud_analysis.high_fraud_percentage) }}%)
📊 Medium Risk: {{ fraud_analysis.medium_fraud_count }} số
📊 Low Risk: {{ fraud_analysis.low_fraud_count }} số

=== PHÂN TÍCH VALIDATION METHODS ===
{% for method, percentage in method_analysis.method_percentages.items() %}
🔧 {{ method.replace('_', ' ').title() }}: {{ "%.1f" | format(percentage) }}%
{% endfor %}

=== TREND ANALYSIS ===
📈 Trend Direction: {{ trend_analysis.trend_direction.upper() }}
🎯 Confidence Trend: {{ trend_analysis.confidence_trend.upper() }}
⚠️ Risk Trend: {{ trend_analysis.risk_trend.upper() }}
📊 Volume Trend: {{ trend_analysis.volume_trend.upper() }}
'''
