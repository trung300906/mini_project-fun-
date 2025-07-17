#!/usr/bin/env python3
"""
Main phone validator - integrates all modules
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import all modules
from .core.models import PhoneValidationResult, PhoneStatus, ValidationMethod, RiskLevel
from .core.validators import BasicPhoneValidator
from .core.patterns import PatternAnalyzer
from .core.algorithms import AlgorithmValidator
from .ml.classifier import PhoneMLClassifier
from .ai.deep_learning import PhoneAIValidator
from .database.db_manager import PhoneValidationDatabase
from .api.validators import APIValidator
from .reports.generator import ReportGenerator
from .utils.config import ConfigManager, AppConfig
from .utils.cache import PhoneValidationCache, global_cache_manager
from .utils.logging import setup_logging, ValidationLogger, BatchValidationLogger, PerformanceLogger

logger = logging.getLogger(__name__)

class AdvancedPhoneValidator:
    """Advanced phone validator with AI, ML, and comprehensive analysis"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize validator with configuration"""
        self.config = config or ConfigManager().get_config()
        
        # Setup logging
        self.logger = setup_logging(self.config.logging.__dict__)
        self.validation_logger = ValidationLogger(self.logger)
        self.batch_logger = BatchValidationLogger(self.logger)
        self.performance_logger = PerformanceLogger(self.logger)
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'ml_predictions': 0,
            'ai_predictions': 0,
            'errors': 0
        }
        
        self.logger.info("Advanced Phone Validator initialized successfully")
    
    def _initialize_components(self):
        """Initialize all validator components"""
        try:
            # Core validators
            self.basic_validator = BasicPhoneValidator()
            self.pattern_analyzer = PatternAnalyzer()
            self.algorithm_validator = AlgorithmValidator()
            
            # ML/AI validators
            if self.config.validation.enable_ml_validation:
                self.ml_classifier = PhoneMLClassifier(self.config.ml)
            else:
                self.ml_classifier = None
            
            if self.config.validation.enable_ai_validation:
                self.ai_validator = PhoneAIValidator(self.config.ai)
            else:
                self.ai_validator = None
            
            # Database
            self.database = PhoneValidationDatabase(self.config.database.path)
            
            # API validator
            if self.config.validation.enable_api_validation:
                self.api_validator = APIValidator(self.config.api)
            else:
                self.api_validator = None
            
            # Cache
            if self.config.cache_enabled:
                self.cache = PhoneValidationCache(global_cache_manager)
            else:
                self.cache = None
            
            # Report generator
            self.report_generator = ReportGenerator()
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def validate(self, 
                 phone_number: str, 
                 country_code: Optional[str] = None,
                 use_cache: bool = True,
                 validation_methods: Optional[List[str]] = None) -> PhoneValidationResult:
        """
        Validate a single phone number
        
        Args:
            phone_number: Phone number to validate
            country_code: Country code (default: VN)
            use_cache: Whether to use cache
            validation_methods: Specific validation methods to use
            
        Returns:
            PhoneValidationResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            self.stats['total_validations'] += 1
            
            # Normalize inputs
            phone_number = phone_number.strip()
            country_code = country_code or self.config.validation.default_country
            
            self.validation_logger.log_validation_start(phone_number, "comprehensive")
            
            # Check cache first
            if use_cache and self.cache:
                cached_result = self.cache.get_validation_result(phone_number)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    self.validation_logger.log_cache_hit(phone_number, "validation")
                    return PhoneValidationResult(**cached_result)
                else:
                    self.stats['cache_misses'] += 1
                    self.validation_logger.log_cache_miss(phone_number, "validation")
            
            # Perform validation
            result = self._perform_comprehensive_validation(
                phone_number, 
                country_code, 
                validation_methods
            )
            
            # Store in cache
            if use_cache and self.cache:
                self.cache.set_validation_result(
                    phone_number, 
                    result.__dict__, 
                    ttl=self.config.cache_ttl
                )
            
            # Store in database
            if self.database:
                self.database.save_validation_result(result)
            
            # Log result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.timestamp = datetime.now()
            
            self.validation_logger.log_validation_result(
                phone_number, 
                result.__dict__, 
                processing_time
            )
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            self.validation_logger.log_validation_error(phone_number, e, "comprehensive")
            
            # Return error result
            return PhoneValidationResult(
                phone_number=phone_number,
                status=PhoneStatus.UNKNOWN,
                confidence_score=0.0,
                risk_score=1.0,
                error=str(e),
                timestamp=datetime.now(),
                processing_time=time.time() - start_time
            )
    
    def _perform_comprehensive_validation(self, 
                                        phone_number: str, 
                                        country_code: str,
                                        validation_methods: Optional[List[str]] = None) -> PhoneValidationResult:
        """Perform comprehensive validation using all available methods"""
        
        results = {}
        used_methods = []
        
        # Basic validation
        if self.config.validation.enable_basic_validation:
            if not validation_methods or 'basic' in validation_methods:
                try:
                    basic_result = self.basic_validator.validate(phone_number, country_code)
                    results['basic'] = basic_result
                    used_methods.append('basic')
                except Exception as e:
                    self.logger.error(f"Basic validation failed: {e}")
        
        # Pattern analysis
        if self.config.validation.enable_pattern_analysis:
            if not validation_methods or 'pattern' in validation_methods:
                try:
                    pattern_result = self.pattern_analyzer.analyze(phone_number)
                    results['pattern'] = pattern_result
                    used_methods.append('pattern')
                except Exception as e:
                    self.logger.error(f"Pattern analysis failed: {e}")
        
        # Algorithm validation
        if self.config.validation.enable_algorithm_validation:
            if not validation_methods or 'algorithm' in validation_methods:
                try:
                    algorithm_result = self.algorithm_validator.validate(phone_number)
                    results['algorithm'] = algorithm_result
                    used_methods.append('algorithm')
                except Exception as e:
                    self.logger.error(f"Algorithm validation failed: {e}")
        
        # ML validation
        if self.ml_classifier and self.config.validation.enable_ml_validation:
            if not validation_methods or 'ml' in validation_methods:
                try:
                    ml_result = self.ml_classifier.predict(phone_number)
                    results['ml'] = ml_result
                    used_methods.append('ml')
                    self.stats['ml_predictions'] += 1
                except Exception as e:
                    self.logger.error(f"ML validation failed: {e}")
        
        # AI validation
        if self.ai_validator and self.config.validation.enable_ai_validation:
            if not validation_methods or 'ai' in validation_methods:
                try:
                    ai_result = self.ai_validator.validate(phone_number)
                    results['ai'] = ai_result
                    used_methods.append('ai')
                    self.stats['ai_predictions'] += 1
                except Exception as e:
                    self.logger.error(f"AI validation failed: {e}")
        
        # API validation
        if self.api_validator and self.config.validation.enable_api_validation:
            if not validation_methods or 'api' in validation_methods:
                try:
                    api_result = asyncio.run(self.api_validator.validate_phone(phone_number))
                    results['api'] = api_result
                    used_methods.append('api')
                    self.stats['api_calls'] += 1
                except Exception as e:
                    self.logger.error(f"API validation failed: {e}")
        
        # Combine results
        return self._combine_validation_results(phone_number, results, used_methods)
    
    def _combine_validation_results(self, 
                                   phone_number: str, 
                                   results: Dict[str, Any], 
                                   used_methods: List[str]) -> PhoneValidationResult:
        """Combine results from all validation methods"""
        
        # Initialize result
        combined_result = PhoneValidationResult(
            phone_number=phone_number,
            validation_methods=used_methods
        )
        
        # Extract basic info
        basic_result = results.get('basic', {})
        combined_result.is_valid_format = basic_result.get('is_valid_format', False)
        combined_result.is_possible = basic_result.get('is_possible', False)
        combined_result.country = basic_result.get('country', 'Unknown')
        combined_result.location = basic_result.get('location', 'Unknown')
        combined_result.carrier_name = basic_result.get('carrier_name', 'Unknown')
        combined_result.line_type = basic_result.get('line_type', 'Unknown')
        
        # Calculate weighted confidence and risk scores
        total_weight = 0
        weighted_confidence = 0
        weighted_risk = 0
        
        weights = {
            'basic': self.config.validation.basic_weight,
            'pattern': self.config.validation.pattern_weight,
            'algorithm': self.config.validation.algorithm_weight,
            'ml': self.config.validation.ml_weight,
            'ai': self.config.validation.ai_weight,
            'api': self.config.validation.api_weight
        }
        
        for method, result in results.items():
            if method in weights:
                weight = weights[method]
                confidence = result.get('confidence', 0.5)
                risk = result.get('risk', 0.5)
                
                weighted_confidence += confidence * weight
                weighted_risk += risk * weight
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            combined_result.confidence_score = weighted_confidence / total_weight
            combined_result.risk_score = weighted_risk / total_weight
        else:
            combined_result.confidence_score = 0.5
            combined_result.risk_score = 0.5
        
        # Determine status
        combined_result.status = self._determine_status(
            combined_result.confidence_score,
            combined_result.risk_score,
            results
        )
        
        # Add detailed results
        combined_result.ml_predictions = results.get('ml', {})
        combined_result.ai_analysis = results.get('ai', {})
        combined_result.pattern_analysis = results.get('pattern', {})
        combined_result.algorithm_analysis = results.get('algorithm', {})
        combined_result.api_responses = results.get('api', {})
        
        # Calculate fraud probability
        combined_result.fraud_probability = self._calculate_fraud_probability(results)
        
        # Set AI confidence
        ai_result = results.get('ai', {})
        combined_result.ai_confidence = ai_result.get('confidence', 0.5)
        
        return combined_result
    
    def _determine_status(self, 
                         confidence_score: float, 
                         risk_score: float, 
                         results: Dict[str, Any]) -> PhoneStatus:
        """Determine overall phone status"""
        
        # Check basic format first
        basic_result = results.get('basic', {})
        if not basic_result.get('is_valid_format', False):
            return PhoneStatus.INVALID
        
        # Check fraud probability
        fraud_prob = self._calculate_fraud_probability(results)
        if fraud_prob > self.config.validation.fraud_threshold:
            return PhoneStatus.SUSPICIOUS
        
        # Check confidence and risk thresholds
        if confidence_score >= self.config.validation.confidence_threshold:
            if risk_score <= self.config.validation.risk_threshold:
                return PhoneStatus.VALID
            else:
                return PhoneStatus.SUSPICIOUS
        else:
            if risk_score >= self.config.validation.risk_threshold:
                return PhoneStatus.INVALID
            else:
                return PhoneStatus.UNKNOWN
    
    def _calculate_fraud_probability(self, results: Dict[str, Any]) -> float:
        """Calculate fraud probability from all results"""
        fraud_indicators = []
        
        # Pattern analysis indicators
        pattern_result = results.get('pattern', {})
        if pattern_result.get('is_suspicious', False):
            fraud_indicators.append(0.8)
        
        # Algorithm indicators
        algorithm_result = results.get('algorithm', {})
        if algorithm_result.get('luhn_check', False) == False:
            fraud_indicators.append(0.6)
        
        # ML indicators
        ml_result = results.get('ml', {})
        if ml_result.get('fraud_probability', 0) > 0.5:
            fraud_indicators.append(ml_result.get('fraud_probability', 0))
        
        # AI indicators
        ai_result = results.get('ai', {})
        if ai_result.get('fraud_probability', 0) > 0.5:
            fraud_indicators.append(ai_result.get('fraud_probability', 0))
        
        # API indicators
        api_result = results.get('api', {})
        if api_result.get('consensus_fraud_probability', 0) > 0.5:
            fraud_indicators.append(api_result.get('consensus_fraud_probability', 0))
        
        # Calculate weighted average
        if fraud_indicators:
            return sum(fraud_indicators) / len(fraud_indicators)
        else:
            return 0.0
    
    def validate_batch(self, 
                      phone_numbers: List[str], 
                      country_code: Optional[str] = None,
                      use_cache: bool = True,
                      parallel: bool = True,
                      max_workers: Optional[int] = None) -> List[PhoneValidationResult]:
        """
        Validate multiple phone numbers
        
        Args:
            phone_numbers: List of phone numbers to validate
            country_code: Default country code
            use_cache: Whether to use cache
            parallel: Whether to process in parallel
            max_workers: Maximum number of worker threads
            
        Returns:
            List of PhoneValidationResult objects
        """
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}"
        
        self.batch_logger.log_batch_start(len(phone_numbers), batch_id)
        
        try:
            if parallel and self.config.parallel_processing:
                max_workers = max_workers or self.config.max_workers
                results = self._validate_batch_parallel(
                    phone_numbers, 
                    country_code, 
                    use_cache, 
                    max_workers,
                    batch_id
                )
            else:
                results = self._validate_batch_sequential(
                    phone_numbers, 
                    country_code, 
                    use_cache,
                    batch_id
                )
            
            # Log completion
            total_time = time.time() - start_time
            batch_stats = self._calculate_batch_stats(results)
            
            self.batch_logger.log_batch_complete(batch_id, batch_stats, total_time)
            
            return results
            
        except Exception as e:
            self.batch_logger.log_batch_error(batch_id, e)
            raise
    
    def _validate_batch_parallel(self, 
                                phone_numbers: List[str], 
                                country_code: Optional[str], 
                                use_cache: bool, 
                                max_workers: int,
                                batch_id: str) -> List[PhoneValidationResult]:
        """Validate batch in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_phone = {
                executor.submit(self.validate, phone, country_code, use_cache): phone
                for phone in phone_numbers
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_phone):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Log progress
                    if completed % 10 == 0 or completed == len(phone_numbers):
                        self.batch_logger.log_batch_progress(batch_id, completed, len(phone_numbers))
                        
                except Exception as e:
                    phone = future_to_phone[future]
                    self.logger.error(f"Error validating {phone}: {e}")
                    
                    # Add error result
                    error_result = PhoneValidationResult(
                        phone_number=phone,
                        status=PhoneStatus.UNKNOWN,
                        confidence_score=0.0,
                        risk_score=1.0,
                        error=str(e),
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
                    completed += 1
        
        return results
    
    def _validate_batch_sequential(self, 
                                  phone_numbers: List[str], 
                                  country_code: Optional[str], 
                                  use_cache: bool,
                                  batch_id: str) -> List[PhoneValidationResult]:
        """Validate batch sequentially"""
        results = []
        
        for i, phone in enumerate(phone_numbers):
            try:
                result = self.validate(phone, country_code, use_cache)
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == len(phone_numbers):
                    self.batch_logger.log_batch_progress(batch_id, i + 1, len(phone_numbers))
                    
            except Exception as e:
                self.logger.error(f"Error validating {phone}: {e}")
                
                # Add error result
                error_result = PhoneValidationResult(
                    phone_number=phone,
                    status=PhoneStatus.UNKNOWN,
                    confidence_score=0.0,
                    risk_score=1.0,
                    error=str(e),
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    def _calculate_batch_stats(self, results: List[PhoneValidationResult]) -> Dict[str, Any]:
        """Calculate batch statistics"""
        if not results:
            return {}
        
        status_counts = {}
        total_confidence = 0
        total_risk = 0
        total_processing_time = 0
        
        for result in results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_confidence += result.confidence_score
            total_risk += result.risk_score
            total_processing_time += result.processing_time or 0
        
        total_count = len(results)
        
        return {
            'total_count': total_count,
            'status_counts': status_counts,
            'avg_confidence': total_confidence / total_count,
            'avg_risk': total_risk / total_count,
            'avg_processing_time': total_processing_time / total_count,
            'valid_percentage': status_counts.get('valid', 0) / total_count * 100,
            'invalid_percentage': status_counts.get('invalid', 0) / total_count * 100,
            'suspicious_percentage': status_counts.get('suspicious', 0) / total_count * 100
        }
    
    def generate_report(self, 
                       results: Union[PhoneValidationResult, List[PhoneValidationResult]], 
                       report_type: str = 'single') -> str:
        """
        Generate validation report
        
        Args:
            results: Validation result(s)
            report_type: Type of report ('single', 'batch', 'analytics')
            
        Returns:
            Formatted report string
        """
        if report_type == 'single':
            if isinstance(results, list):
                results = results[0]
            return self.report_generator.generate_single_report(results.__dict__)
        elif report_type == 'batch':
            if not isinstance(results, list):
                results = [results]
            results_dict = [r.__dict__ for r in results]
            return self.report_generator.generate_batch_report(results_dict)
        elif report_type == 'analytics':
            if not isinstance(results, list):
                results = [results]
            results_dict = [r.__dict__ for r in results]
            return self.report_generator.generate_analytics_report(results_dict)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def export_results(self, 
                      results: List[PhoneValidationResult], 
                      filename: str, 
                      format: str = 'csv'):
        """
        Export validation results
        
        Args:
            results: List of validation results
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
        """
        results_dict = [r.__dict__ for r in results]
        
        if format == 'csv':
            self.report_generator.export_to_csv(results_dict, filename)
        elif format == 'excel':
            self.report_generator.export_to_excel(results_dict, filename)
        elif format == 'json':
            self.report_generator.export_to_json(results_dict, filename)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        cache_stats = self.cache.get_cache_stats() if self.cache else {}
        
        return {
            'validation_stats': self.stats,
            'cache_stats': cache_stats,
            'database_stats': self.database.get_statistics() if self.database else {}
        }
    
    def clear_cache(self):
        """Clear validation cache"""
        if self.cache:
            self.cache.cache_manager.clear()
            self.logger.info("Cache cleared")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.database:
            self.database.close()
        
        if self.cache:
            self.cache.cache_manager.cleanup_expired()
        
        self.logger.info("Validator cleanup completed")

# Convenience functions
def validate_phone(phone_number: str, 
                  country_code: Optional[str] = None,
                  config: Optional[AppConfig] = None) -> PhoneValidationResult:
    """Validate a single phone number (convenience function)"""
    validator = AdvancedPhoneValidator(config)
    return validator.validate(phone_number, country_code)

def validate_phones(phone_numbers: List[str], 
                   country_code: Optional[str] = None,
                   config: Optional[AppConfig] = None) -> List[PhoneValidationResult]:
    """Validate multiple phone numbers (convenience function)"""
    validator = AdvancedPhoneValidator(config)
    return validator.validate_batch(phone_numbers, country_code)

# Main execution
if __name__ == "__main__":
    # Example usage
    validator = AdvancedPhoneValidator()
    
    # Single validation
    result = validator.validate("0123456789", "VN")
    print(validator.generate_report(result, "single"))
    
    # Batch validation
    phones = ["0123456789", "0987654321", "0111222333"]
    results = validator.validate_batch(phones, "VN")
    print(validator.generate_report(results, "batch"))
    
    # Export results
    validator.export_results(results, "validation_results.csv", "csv")
    
    # Show statistics
    stats = validator.get_statistics()
    print(f"Statistics: {stats}")
    
    # Cleanup
    validator.cleanup()
