#!/usr/bin/env python3
"""
Advanced Phone Validator Package

A comprehensive phone number validation system with AI, ML, and multi-method validation.
"""

__version__ = "1.0.0"
__author__ = "Phone Validator Team"
__email__ = "support@phonevalidator.com"
__description__ = "Advanced phone number validation with AI and ML"

# Main imports
from .main import AdvancedPhoneValidator, validate_phone, validate_phones
from .core.models import PhoneValidationResult, PhoneStatus, ValidationMethod, RiskLevel
from .utils.config import ConfigManager, AppConfig

# Core validators
from .core.validators import BasicPhoneValidator
from .core.patterns import PatternAnalyzer
from .core.algorithms import AlgorithmValidator

# ML/AI validators
from .ml.classifier import PhoneMLClassifier
from .ai.deep_learning import PhoneAIValidator

# Database and API
from .database.db_manager import PhoneValidationDatabase
from .api.validators import APIValidator

# Utilities
from .utils.cache import PhoneValidationCache
from .utils.logging import get_logger
from .reports.generator import ReportGenerator

# Default configuration
DEFAULT_CONFIG = AppConfig()

# Package-level convenience functions
def quick_validate(phone_number: str, country_code: str = "VN") -> dict:
    """
    Quick validation function for simple use cases
    
    Args:
        phone_number: Phone number to validate
        country_code: Country code (default: VN)
        
    Returns:
        Dictionary with validation result
    """
    validator = AdvancedPhoneValidator()
    result = validator.validate(phone_number, country_code)
    
    return {
        'phone_number': result.phone_number,
        'is_valid': result.status == PhoneStatus.VALID,
        'status': result.status.value,
        'confidence': result.confidence_score,
        'risk': result.risk_score,
        'country': result.country,
        'carrier': result.carrier_name,
        'line_type': result.line_type
    }

def batch_validate(phone_numbers: list, country_code: str = "VN") -> list:
    """
    Batch validation function for simple use cases
    
    Args:
        phone_numbers: List of phone numbers to validate
        country_code: Country code (default: VN)
        
    Returns:
        List of validation results
    """
    validator = AdvancedPhoneValidator()
    results = validator.validate_batch(phone_numbers, country_code)
    
    return [
        {
            'phone_number': result.phone_number,
            'is_valid': result.status == PhoneStatus.VALID,
            'status': result.status.value,
            'confidence': result.confidence_score,
            'risk': result.risk_score,
            'country': result.country,
            'carrier': result.carrier_name,
            'line_type': result.line_type
        }
        for result in results
    ]

# Package metadata
__all__ = [
    # Main classes
    'AdvancedPhoneValidator',
    'PhoneValidationResult',
    'PhoneStatus',
    'ValidationMethod',
    'RiskLevel',
    
    # Convenience functions
    'validate_phone',
    'validate_phones',
    'quick_validate',
    'batch_validate',
    
    # Core validators
    'BasicPhoneValidator',
    'PatternAnalyzer',
    'AlgorithmValidator',
    
    # ML/AI validators
    'PhoneMLClassifier',
    'PhoneAIValidator',
    
    # Database and API
    'PhoneValidationDatabase',
    'APIValidator',
    
    # Utilities
    'ConfigManager',
    'AppConfig',
    'PhoneValidationCache',
    'ReportGenerator',
    'get_logger',
    
    # Constants
    'DEFAULT_CONFIG',
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]
