#!/usr/bin/env python3
"""
Logging utilities for phone validator
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'phone_number'):
            log_entry['phone_number'] = record.phone_number
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'validation_method'):
            log_entry['validation_method'] = record.validation_method
        
        return json.dumps(log_entry, ensure_ascii=False)

class PhoneValidatorLogger:
    """Custom logger for phone validation"""
    
    def __init__(self, name: str = "phone_validator"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.configured = False
    
    def configure(self, 
                  level: str = "INFO",
                  log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                  file_path: Optional[str] = None,
                  max_bytes: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5,
                  console_logging: bool = True,
                  colored_console: bool = True,
                  json_logging: bool = False):
        """Configure logger"""
        
        if self.configured:
            return
        
        # Set level
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            if colored_console:
                console_formatter = ColoredFormatter(log_format)
            else:
                console_formatter = logging.Formatter(log_format)
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_path:
            # Ensure directory exists
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            
            if json_logging:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(log_format)
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.configured = True
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        if not self.configured:
            self.configure()
        
        # Create log record with extra fields
        extra = {}
        for key, value in kwargs.items():
            if not key.startswith('_'):
                extra[key] = value
        
        self.logger.log(level, message, extra=extra)

class ValidationLogger:
    """Specialized logger for validation operations"""
    
    def __init__(self, logger: PhoneValidatorLogger):
        self.logger = logger
    
    def log_validation_start(self, phone_number: str, validation_method: str):
        """Log validation start"""
        self.logger.info(
            f"Starting validation for {phone_number} using {validation_method}",
            phone_number=phone_number,
            validation_method=validation_method,
            operation="validation_start"
        )
    
    def log_validation_result(self, phone_number: str, result: Dict[str, Any], processing_time: float):
        """Log validation result"""
        self.logger.info(
            f"Validation completed for {phone_number}: {result.get('status', 'unknown')}",
            phone_number=phone_number,
            result=result,
            processing_time=processing_time,
            operation="validation_complete"
        )
    
    def log_validation_error(self, phone_number: str, error: Exception, validation_method: str):
        """Log validation error"""
        self.logger.error(
            f"Validation failed for {phone_number} using {validation_method}: {str(error)}",
            phone_number=phone_number,
            validation_method=validation_method,
            error=str(error),
            operation="validation_error",
            exc_info=True
        )
    
    def log_ml_prediction(self, phone_number: str, model_name: str, prediction: Dict[str, Any]):
        """Log ML prediction"""
        self.logger.debug(
            f"ML prediction for {phone_number} using {model_name}: {prediction}",
            phone_number=phone_number,
            model_name=model_name,
            prediction=prediction,
            operation="ml_prediction"
        )
    
    def log_api_call(self, phone_number: str, api_provider: str, success: bool, response_time: float):
        """Log API call"""
        status = "success" if success else "failed"
        self.logger.info(
            f"API call to {api_provider} for {phone_number}: {status} ({response_time:.3f}s)",
            phone_number=phone_number,
            api_provider=api_provider,
            success=success,
            response_time=response_time,
            operation="api_call"
        )
    
    def log_cache_hit(self, phone_number: str, cache_type: str):
        """Log cache hit"""
        self.logger.debug(
            f"Cache hit for {phone_number} in {cache_type}",
            phone_number=phone_number,
            cache_type=cache_type,
            operation="cache_hit"
        )
    
    def log_cache_miss(self, phone_number: str, cache_type: str):
        """Log cache miss"""
        self.logger.debug(
            f"Cache miss for {phone_number} in {cache_type}",
            phone_number=phone_number,
            cache_type=cache_type,
            operation="cache_miss"
        )

class BatchValidationLogger:
    """Logger for batch validation operations"""
    
    def __init__(self, logger: PhoneValidatorLogger):
        self.logger = logger
    
    def log_batch_start(self, batch_size: int, batch_id: str):
        """Log batch validation start"""
        self.logger.info(
            f"Starting batch validation: {batch_size} numbers (batch_id: {batch_id})",
            batch_size=batch_size,
            batch_id=batch_id,
            operation="batch_start"
        )
    
    def log_batch_progress(self, batch_id: str, completed: int, total: int):
        """Log batch progress"""
        progress = (completed / total) * 100
        self.logger.info(
            f"Batch {batch_id} progress: {completed}/{total} ({progress:.1f}%)",
            batch_id=batch_id,
            completed=completed,
            total=total,
            progress=progress,
            operation="batch_progress"
        )
    
    def log_batch_complete(self, batch_id: str, results: Dict[str, Any], total_time: float):
        """Log batch completion"""
        self.logger.info(
            f"Batch {batch_id} completed in {total_time:.2f}s: {results}",
            batch_id=batch_id,
            results=results,
            total_time=total_time,
            operation="batch_complete"
        )
    
    def log_batch_error(self, batch_id: str, error: Exception):
        """Log batch error"""
        self.logger.error(
            f"Batch {batch_id} failed: {str(error)}",
            batch_id=batch_id,
            error=str(error),
            operation="batch_error",
            exc_info=True
        )

class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, logger: PhoneValidatorLogger):
        self.logger = logger
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.logger.info(
            f"Performance metrics: {metrics}",
            metrics=metrics,
            operation="performance_metrics"
        )
    
    def log_slow_operation(self, operation: str, duration: float, threshold: float = 5.0):
        """Log slow operation"""
        if duration > threshold:
            self.logger.warning(
                f"Slow operation detected: {operation} took {duration:.2f}s (threshold: {threshold}s)",
                operation=operation,
                duration=duration,
                threshold=threshold,
                operation_type="slow_operation"
            )
    
    def log_memory_usage(self, memory_mb: float, operation: str):
        """Log memory usage"""
        self.logger.debug(
            f"Memory usage during {operation}: {memory_mb:.2f}MB",
            memory_mb=memory_mb,
            operation=operation,
            operation_type="memory_usage"
        )
    
    def log_cache_performance(self, cache_stats: Dict[str, Any]):
        """Log cache performance"""
        self.logger.info(
            f"Cache performance: {cache_stats}",
            cache_stats=cache_stats,
            operation="cache_performance"
        )

class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, logger: PhoneValidatorLogger):
        self.logger = logger
    
    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activity"""
        self.logger.warning(
            f"Suspicious activity detected: {activity}",
            activity=activity,
            details=details,
            operation="suspicious_activity"
        )
    
    def log_rate_limit_exceeded(self, source: str, limit: int, actual: int):
        """Log rate limit exceeded"""
        self.logger.warning(
            f"Rate limit exceeded for {source}: {actual}/{limit}",
            source=source,
            limit=limit,
            actual=actual,
            operation="rate_limit_exceeded"
        )
    
    def log_api_key_usage(self, api_provider: str, key_hash: str, usage_count: int):
        """Log API key usage"""
        self.logger.info(
            f"API key usage for {api_provider}: {usage_count} calls",
            api_provider=api_provider,
            key_hash=key_hash,
            usage_count=usage_count,
            operation="api_key_usage"
        )
    
    def log_data_access(self, data_type: str, access_count: int, user_id: Optional[str] = None):
        """Log data access"""
        self.logger.info(
            f"Data access: {data_type} accessed {access_count} times",
            data_type=data_type,
            access_count=access_count,
            user_id=user_id,
            operation="data_access"
        )

def setup_logging(config: Dict[str, Any]) -> PhoneValidatorLogger:
    """Setup logging with configuration"""
    logger = PhoneValidatorLogger()
    
    logger.configure(
        level=config.get('level', 'INFO'),
        log_format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        file_path=config.get('file_path'),
        max_bytes=config.get('max_bytes', 10 * 1024 * 1024),
        backup_count=config.get('backup_count', 5),
        console_logging=config.get('console_logging', True),
        colored_console=config.get('colored_console', True),
        json_logging=config.get('json_logging', False)
    )
    
    return logger

def get_logger(name: str = "phone_validator") -> PhoneValidatorLogger:
    """Get or create logger"""
    return PhoneValidatorLogger(name)

# Global logger instance
main_logger = PhoneValidatorLogger()
validation_logger = ValidationLogger(main_logger)
batch_logger = BatchValidationLogger(main_logger)
performance_logger = PerformanceLogger(main_logger)
security_logger = SecurityLogger(main_logger)
