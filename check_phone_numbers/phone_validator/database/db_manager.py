#!/usr/bin/env python3
"""
Database module for phone validation system
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class PhoneValidationDatabase:
    """Database cho phone validation system"""
    
    def __init__(self, db_path: str = "phone_validation.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Khởi tạo database và các tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table chính cho validation results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS phone_validation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    normalized_number TEXT,
                    status TEXT NOT NULL,
                    confidence_score REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    country TEXT,
                    carrier_name TEXT,
                    location TEXT,
                    line_type TEXT,
                    is_valid_format BOOLEAN,
                    is_possible BOOLEAN,
                    validation_methods TEXT,
                    ai_confidence REAL,
                    ml_predictions TEXT,
                    fraud_probability REAL,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    additional_info TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table cho suspicious patterns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS suspicious_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL UNIQUE,
                    pattern_type TEXT NOT NULL,
                    risk_level INTEGER DEFAULT 1,
                    description TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table cho ML features
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    feature_vector TEXT NOT NULL,
                    extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (phone_number) REFERENCES phone_validation_history(phone_number)
                )
            ''')
            
            # Table cho AI predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction INTEGER,
                    confidence REAL,
                    probability REAL,
                    processing_time REAL,
                    predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (phone_number) REFERENCES phone_validation_history(phone_number)
                )
            ''')
            
            # Table cho carrier information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS carrier_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    carrier_name TEXT NOT NULL UNIQUE,
                    country TEXT,
                    carrier_type TEXT,
                    is_trusted BOOLEAN DEFAULT FALSE,
                    reputation_score REAL DEFAULT 0.5,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table cho batch processing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    total_count INTEGER,
                    processed_count INTEGER DEFAULT 0,
                    valid_count INTEGER DEFAULT 0,
                    invalid_count INTEGER DEFAULT 0,
                    suspicious_count INTEGER DEFAULT 0,
                    unknown_count INTEGER DEFAULT 0,
                    average_confidence REAL,
                    average_risk REAL,
                    processing_time REAL,
                    status TEXT DEFAULT 'processing',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME
                )
            ''')
            
            # Table cho validation statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_validations INTEGER DEFAULT 0,
                    valid_numbers INTEGER DEFAULT 0,
                    invalid_numbers INTEGER DEFAULT 0,
                    suspicious_numbers INTEGER DEFAULT 0,
                    unknown_numbers INTEGER DEFAULT 0,
                    average_confidence REAL,
                    average_risk REAL,
                    top_carrier TEXT,
                    top_country TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # Indexes cho performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON phone_validation_history(phone_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON phone_validation_history(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON phone_validation_history(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_carrier ON phone_validation_history(carrier_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON phone_validation_history(country)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_id ON batch_processing(batch_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON validation_statistics(date)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager cho database connection"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()
    
    def save_validation_result(self, result: Dict) -> int:
        """Lưu kết quả validation"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO phone_validation_history (
                        phone_number, normalized_number, status, confidence_score,
                        risk_score, risk_level, country, carrier_name, location,
                        line_type, is_valid_format, is_possible, validation_methods,
                        ai_confidence, ml_predictions, fraud_probability,
                        processing_time, additional_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['phone_number'],
                    result.get('normalized_number', ''),
                    result['status'],
                    result['confidence_score'],
                    result['risk_score'],
                    result.get('risk_level', 'unknown'),
                    result['country'],
                    result['carrier_name'],
                    result['location'],
                    result['line_type'],
                    result['is_valid_format'],
                    result['is_possible'],
                    json.dumps(result.get('validation_methods', [])),
                    result.get('ai_confidence', 0.0),
                    json.dumps(result.get('ml_predictions', {})),
                    result.get('fraud_probability', 0.0),
                    result.get('processing_time', 0.0),
                    json.dumps(result.get('additional_info', {}))
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                # Update statistics
                self._update_daily_statistics(result)
                
                return result_id
    
    def get_validation_history(self, phone_number: str, limit: int = 10) -> List[Dict]:
        """Lấy lịch sử validation cho một số"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM phone_validation_history 
                WHERE phone_number = ? 
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (phone_number, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['validation_methods'] = json.loads(result['validation_methods'] or '[]')
                result['ml_predictions'] = json.loads(result['ml_predictions'] or '{}')
                result['additional_info'] = json.loads(result['additional_info'] or '{}')
                results.append(result)
            
            return results
    
    def search_validations(self, filters: Dict, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Tìm kiếm validations với filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            where_clauses = []
            params = []
            
            if 'status' in filters:
                where_clauses.append('status = ?')
                params.append(filters['status'])
            
            if 'carrier_name' in filters:
                where_clauses.append('carrier_name LIKE ?')
                params.append(f"%{filters['carrier_name']}%")
            
            if 'country' in filters:
                where_clauses.append('country = ?')
                params.append(filters['country'])
            
            if 'date_from' in filters:
                where_clauses.append('timestamp >= ?')
                params.append(filters['date_from'])
            
            if 'date_to' in filters:
                where_clauses.append('timestamp <= ?')
                params.append(filters['date_to'])
            
            if 'confidence_min' in filters:
                where_clauses.append('confidence_score >= ?')
                params.append(filters['confidence_min'])
            
            if 'risk_max' in filters:
                where_clauses.append('risk_score <= ?')
                params.append(filters['risk_max'])
            
            where_clause = ' AND '.join(where_clauses) if where_clauses else '1=1'
            
            query = f'''
                SELECT * FROM phone_validation_history 
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['validation_methods'] = json.loads(result['validation_methods'] or '[]')
                result['ml_predictions'] = json.loads(result['ml_predictions'] or '{}')
                result['additional_info'] = json.loads(result['additional_info'] or '{}')
                results.append(result)
            
            return results
    
    def save_ml_features(self, phone_number: str, features: Dict):
        """Lưu ML features"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO ml_features (phone_number, feature_vector)
                    VALUES (?, ?)
                ''', (phone_number, json.dumps(features)))
                
                conn.commit()
    
    def save_ai_prediction(self, phone_number: str, model_name: str, prediction: Dict):
        """Lưu AI prediction"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO ai_predictions (
                        phone_number, model_name, prediction, confidence,
                        probability, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    phone_number,
                    model_name,
                    prediction.get('prediction', 0),
                    prediction.get('confidence', 0.0),
                    prediction.get('probability', 0.0),
                    prediction.get('processing_time', 0.0)
                ))
                
                conn.commit()
    
    def get_ai_predictions(self, phone_number: str) -> List[Dict]:
        """Lấy AI predictions cho một số"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ai_predictions 
                WHERE phone_number = ?
                ORDER BY predicted_at DESC
            ''', (phone_number,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def add_suspicious_pattern(self, pattern: str, pattern_type: str, risk_level: int = 1, description: str = ""):
        """Thêm suspicious pattern"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO suspicious_patterns (
                        pattern, pattern_type, risk_level, description
                    ) VALUES (?, ?, ?, ?)
                ''', (pattern, pattern_type, risk_level, description))
                
                conn.commit()
    
    def get_suspicious_patterns(self, pattern_type: Optional[str] = None) -> List[Dict]:
        """Lấy suspicious patterns"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if pattern_type:
                cursor.execute('''
                    SELECT * FROM suspicious_patterns 
                    WHERE pattern_type = ? AND is_active = TRUE
                    ORDER BY risk_level DESC
                ''', (pattern_type,))
            else:
                cursor.execute('''
                    SELECT * FROM suspicious_patterns 
                    WHERE is_active = TRUE
                    ORDER BY risk_level DESC
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_carrier_info(self, carrier_name: str, info: Dict):
        """Cập nhật thông tin carrier"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO carrier_info (
                        carrier_name, country, carrier_type, is_trusted, reputation_score
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    carrier_name,
                    info.get('country', ''),
                    info.get('carrier_type', ''),
                    info.get('is_trusted', False),
                    info.get('reputation_score', 0.5)
                ))
                
                conn.commit()
    
    def get_carrier_info(self, carrier_name: str) -> Optional[Dict]:
        """Lấy thông tin carrier"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM carrier_info WHERE carrier_name = ?
            ''', (carrier_name,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def start_batch_processing(self, batch_id: str, total_count: int) -> int:
        """Bắt đầu batch processing"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO batch_processing (batch_id, total_count)
                    VALUES (?, ?)
                ''', (batch_id, total_count))
                
                batch_record_id = cursor.lastrowid
                conn.commit()
                
                return batch_record_id
    
    def update_batch_progress(self, batch_id: str, processed_count: int, stats: Dict):
        """Cập nhật tiến trình batch"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE batch_processing SET
                        processed_count = ?,
                        valid_count = ?,
                        invalid_count = ?,
                        suspicious_count = ?,
                        unknown_count = ?,
                        average_confidence = ?,
                        average_risk = ?
                    WHERE batch_id = ?
                ''', (
                    processed_count,
                    stats.get('valid_count', 0),
                    stats.get('invalid_count', 0),
                    stats.get('suspicious_count', 0),
                    stats.get('unknown_count', 0),
                    stats.get('average_confidence', 0.0),
                    stats.get('average_risk', 0.0),
                    batch_id
                ))
                
                conn.commit()
    
    def complete_batch_processing(self, batch_id: str, processing_time: float):
        """Hoàn thành batch processing"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE batch_processing SET
                        status = 'completed',
                        processing_time = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE batch_id = ?
                ''', (processing_time, batch_id))
                
                conn.commit()
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Lấy trạng thái batch"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM batch_processing WHERE batch_id = ?
            ''', (batch_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_daily_statistics(self, date: str) -> Optional[Dict]:
        """Lấy thống kê theo ngày"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM validation_statistics WHERE date = ?
            ''', (date,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_statistics_range(self, date_from: str, date_to: str) -> List[Dict]:
        """Lấy thống kê trong khoảng thời gian"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM validation_statistics 
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            ''', (date_from, date_to))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def _update_daily_statistics(self, result: Dict):
        """Cập nhật thống kê hàng ngày"""
        today = datetime.now().date().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Lấy thống kê hiện tại
            cursor.execute('''
                SELECT * FROM validation_statistics WHERE date = ?
            ''', (today,))
            
            current_stats = cursor.fetchone()
            
            if current_stats:
                # Cập nhật
                new_total = current_stats['total_validations'] + 1
                
                # Đếm theo status
                status_counts = {
                    'valid': current_stats['valid_numbers'],
                    'invalid': current_stats['invalid_numbers'],
                    'suspicious': current_stats['suspicious_numbers'],
                    'unknown': current_stats['unknown_numbers']
                }
                
                if result['status'] in status_counts:
                    status_counts[result['status']] += 1
                
                # Tính average confidence và risk
                old_avg_conf = current_stats['average_confidence'] or 0
                old_avg_risk = current_stats['average_risk'] or 0
                
                new_avg_conf = (old_avg_conf * (new_total - 1) + result['confidence_score']) / new_total
                new_avg_risk = (old_avg_risk * (new_total - 1) + result['risk_score']) / new_total
                
                cursor.execute('''
                    UPDATE validation_statistics SET
                        total_validations = ?,
                        valid_numbers = ?,
                        invalid_numbers = ?,
                        suspicious_numbers = ?,
                        unknown_numbers = ?,
                        average_confidence = ?,
                        average_risk = ?
                    WHERE date = ?
                ''', (
                    new_total,
                    status_counts['valid'],
                    status_counts['invalid'],
                    status_counts['suspicious'],
                    status_counts['unknown'],
                    new_avg_conf,
                    new_avg_risk,
                    today
                ))
            else:
                # Tạo mới
                status_counts = {
                    'valid': 1 if result['status'] == 'valid' else 0,
                    'invalid': 1 if result['status'] == 'invalid' else 0,
                    'suspicious': 1 if result['status'] == 'suspicious' else 0,
                    'unknown': 1 if result['status'] == 'unknown' else 0
                }
                
                cursor.execute('''
                    INSERT INTO validation_statistics (
                        date, total_validations, valid_numbers, invalid_numbers,
                        suspicious_numbers, unknown_numbers, average_confidence,
                        average_risk, top_carrier, top_country
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    today, 1, status_counts['valid'], status_counts['invalid'],
                    status_counts['suspicious'], status_counts['unknown'],
                    result['confidence_score'], result['risk_score'],
                    result['carrier_name'], result['country']
                ))
            
            conn.commit()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Dọn dẹp dữ liệu cũ"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Xóa validation history cũ
                cursor.execute('''
                    DELETE FROM phone_validation_history 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Xóa ML features cũ
                cursor.execute('''
                    DELETE FROM ml_features 
                    WHERE extracted_at < ?
                ''', (cutoff_date,))
                
                # Xóa AI predictions cũ
                cursor.execute('''
                    DELETE FROM ai_predictions 
                    WHERE predicted_at < ?
                ''', (cutoff_date,))
                
                # Xóa batch processing cũ
                cursor.execute('''
                    DELETE FROM batch_processing 
                    WHERE created_at < ?
                ''', (cutoff_date,))
                
                conn.commit()
                
                # Vacuum để giảm kích thước file
                cursor.execute('VACUUM')
                
                logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def get_database_stats(self) -> Dict:
        """Lấy thống kê database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Đếm records trong từng table
            tables = [
                'phone_validation_history',
                'suspicious_patterns',
                'ml_features',
                'ai_predictions',
                'carrier_info',
                'batch_processing',
                'validation_statistics'
            ]
            
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Thống kê tổng quan
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_validations,
                    AVG(confidence_score) as avg_confidence,
                    AVG(risk_score) as avg_risk,
                    COUNT(DISTINCT carrier_name) as unique_carriers,
                    COUNT(DISTINCT country) as unique_countries
                FROM phone_validation_history
            ''')
            
            general_stats = cursor.fetchone()
            stats.update(dict(general_stats))
            
            return stats
