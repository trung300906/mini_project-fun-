#!/usr/bin/env python3
"""
Data models and enums for phone validation system
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

class PhoneStatus(Enum):
    """Trạng thái số điện thoại"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"
    VOIP = "voip"
    LANDLINE = "landline"
    MOBILE = "mobile"

class ValidationMethod(Enum):
    """Phương pháp kiểm tra"""
    FORMAT_VALIDATION = "format_validation"
    PATTERN_ANALYSIS = "pattern_analysis"
    LUHN_ALGORITHM = "luhn_algorithm"
    BAYESIAN_ANALYSIS = "bayesian_analysis"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NEURAL_NETWORK = "neural_network"
    HEURISTIC_ANALYSIS = "heuristic_analysis"
    HMAC_VERIFICATION = "hmac_verification"
    CARRIER_VALIDATION = "carrier_validation"
    GEOGRAPHIC_VALIDATION = "geographic_validation"
    API_VALIDATION = "api_validation"
    AI_PREDICTION = "ai_prediction"
    ENSEMBLE_LEARNING = "ensemble_learning"

class RiskLevel(Enum):
    """Mức độ rủi ro"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

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
    risk_level: RiskLevel
    validation_methods: List[ValidationMethod]
    timestamp: datetime
    additional_info: Dict
    processing_time: float
    ai_confidence: float
    ml_predictions: Dict
    fraud_probability: float
    recommendation: str

@dataclass
class MLFeatures:
    """Machine Learning Features"""
    length: int
    unique_digits: int
    consecutive_same: int
    digit_variance: float
    pattern_score: float
    entropy: float
    digit_frequency_variance: float
    is_palindrome: bool
    arithmetic_progression: bool
    geometric_progression: bool
    prime_digit_ratio: float
    even_odd_ratio: float
    first_digit_analysis: int
    last_digit_analysis: int
    middle_digits_analysis: Dict
    mathematical_properties: Dict
    temporal_patterns: Dict
    geographic_consistency: Dict

@dataclass
class AIAnalysis:
    """AI Analysis Result"""
    neural_network_prediction: float
    deep_learning_confidence: float
    ensemble_score: float
    fraud_detection_score: float
    anomaly_detection_score: float
    similarity_score: float
    clustering_result: str
    recommendation_score: float
    risk_factors: List[str]
    positive_indicators: List[str]

@dataclass
class ValidationConfig:
    """Cấu hình kiểm tra"""
    enable_api_validation: bool = True
    enable_ml_prediction: bool = True
    enable_ai_analysis: bool = True
    enable_deep_learning: bool = True
    confidence_threshold: float = 0.7
    risk_threshold: float = 0.5
    max_processing_time: float = 30.0
    cache_results: bool = True
    save_to_database: bool = True
    generate_detailed_report: bool = True
    parallel_processing: bool = True
    max_workers: int = 10

@dataclass
class BatchValidationResult:
    """Kết quả kiểm tra hàng loạt"""
    total_count: int
    valid_count: int
    invalid_count: int
    suspicious_count: int
    unknown_count: int
    average_confidence: float
    average_risk: float
    processing_time: float
    results: List[PhoneValidationResult]
    summary_stats: Dict
    recommendations: List[str]
