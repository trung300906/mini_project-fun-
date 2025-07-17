#!/usr/bin/env python3
"""
Configuration management for phone validator
"""

import os
import json
try:
    import yaml
except ImportError:
    yaml = None
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "phone_validation.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    max_backup_files: int = 5
    connection_timeout: int = 30
    
@dataclass
class APIConfig:
    """API configuration"""
    timeout: int = 10
    max_retries: int = 3
    retry_delay: int = 1
    rate_limit: int = 100  # requests per minute
    
    # API Keys
    numverify_api_key: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    ipqualityscore_api_key: Optional[str] = None
    
    # API URLs
    numverify_url: str = "http://apilayer.net/api/validate"
    twilio_url: str = "https://lookups.twilio.com/v1/PhoneNumbers"
    ipqualityscore_url: str = "https://ipqualityscore.com/api/json/phone"

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_path: str = "models/"
    training_data_path: str = "training_data/"
    feature_extraction_enabled: bool = True
    ensemble_enabled: bool = True
    
    # Model parameters
    random_forest_n_estimators: int = 100
    random_forest_max_depth: int = 10
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    neural_network_hidden_layers: list = field(default_factory=lambda: [100, 50])
    neural_network_max_iter: int = 1000
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5

@dataclass
class AIConfig:
    """AI/Deep Learning configuration"""
    model_path: str = "ai_models/"
    training_enabled: bool = False
    
    # Transformer config
    transformer_vocab_size: int = 10000
    transformer_embedding_dim: int = 128
    transformer_num_heads: int = 8
    transformer_num_layers: int = 6
    transformer_dropout_rate: float = 0.1
    
    # LSTM config
    lstm_units: int = 64
    lstm_dropout_rate: float = 0.2
    lstm_recurrent_dropout_rate: float = 0.2
    
    # CNN config
    cnn_filters: list = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2
    cnn_dropout_rate: float = 0.25
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

@dataclass
class ValidationConfig:
    """Validation configuration"""
    default_country: str = "VN"
    enable_basic_validation: bool = True
    enable_pattern_analysis: bool = True
    enable_algorithm_validation: bool = True
    enable_ml_validation: bool = True
    enable_ai_validation: bool = True
    enable_api_validation: bool = True
    
    # Thresholds
    confidence_threshold: float = 0.5
    risk_threshold: float = 0.7
    fraud_threshold: float = 0.6
    
    # Validation weights
    basic_weight: float = 0.2
    pattern_weight: float = 0.15
    algorithm_weight: float = 0.15
    ml_weight: float = 0.2
    ai_weight: float = 0.2
    api_weight: float = 0.1

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "phone_validator.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_logging: bool = True

@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    debug: bool = False
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    parallel_processing: bool = True
    max_workers: int = 4

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = AppConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if (self.config_path.endswith('.yaml') or self.config_path.endswith('.yml')) and yaml:
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self._update_config_from_dict(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info(f"Configuration file {self.config_path} not found, using defaults")
                self.save_config()  # Save default config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        try:
            # Update database config
            if 'database' in config_data:
                db_config = config_data['database']
                for key, value in db_config.items():
                    if hasattr(self.config.database, key):
                        setattr(self.config.database, key, value)
            
            # Update API config
            if 'api' in config_data:
                api_config = config_data['api']
                for key, value in api_config.items():
                    if hasattr(self.config.api, key):
                        setattr(self.config.api, key, value)
            
            # Update ML config
            if 'ml' in config_data:
                ml_config = config_data['ml']
                for key, value in ml_config.items():
                    if hasattr(self.config.ml, key):
                        setattr(self.config.ml, key, value)
            
            # Update AI config
            if 'ai' in config_data:
                ai_config = config_data['ai']
                for key, value in ai_config.items():
                    if hasattr(self.config.ai, key):
                        setattr(self.config.ai, key, value)
            
            # Update validation config
            if 'validation' in config_data:
                validation_config = config_data['validation']
                for key, value in validation_config.items():
                    if hasattr(self.config.validation, key):
                        setattr(self.config.validation, key, value)
            
            # Update logging config
            if 'logging' in config_data:
                logging_config = config_data['logging']
                for key, value in logging_config.items():
                    if hasattr(self.config.logging, key):
                        setattr(self.config.logging, key, value)
            
            # Update general settings
            general_keys = ['debug', 'cache_enabled', 'cache_ttl', 'parallel_processing', 'max_workers']
            for key in general_keys:
                if key in config_data:
                    setattr(self.config, key, config_data[key])
                    
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_dict = self._config_to_dict()
            
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if (self.config_path.endswith('.yaml') or self.config_path.endswith('.yml')) and yaml:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': {
                'path': self.config.database.path,
                'backup_enabled': self.config.database.backup_enabled,
                'backup_interval': self.config.database.backup_interval,
                'max_backup_files': self.config.database.max_backup_files,
                'connection_timeout': self.config.database.connection_timeout
            },
            'api': {
                'timeout': self.config.api.timeout,
                'max_retries': self.config.api.max_retries,
                'retry_delay': self.config.api.retry_delay,
                'rate_limit': self.config.api.rate_limit,
                'numverify_api_key': self.config.api.numverify_api_key,
                'twilio_account_sid': self.config.api.twilio_account_sid,
                'twilio_auth_token': self.config.api.twilio_auth_token,
                'ipqualityscore_api_key': self.config.api.ipqualityscore_api_key,
                'numverify_url': self.config.api.numverify_url,
                'twilio_url': self.config.api.twilio_url,
                'ipqualityscore_url': self.config.api.ipqualityscore_url
            },
            'ml': {
                'model_path': self.config.ml.model_path,
                'training_data_path': self.config.ml.training_data_path,
                'feature_extraction_enabled': self.config.ml.feature_extraction_enabled,
                'ensemble_enabled': self.config.ml.ensemble_enabled,
                'random_forest_n_estimators': self.config.ml.random_forest_n_estimators,
                'random_forest_max_depth': self.config.ml.random_forest_max_depth,
                'svm_kernel': self.config.ml.svm_kernel,
                'svm_c': self.config.ml.svm_c,
                'neural_network_hidden_layers': self.config.ml.neural_network_hidden_layers,
                'neural_network_max_iter': self.config.ml.neural_network_max_iter,
                'test_size': self.config.ml.test_size,
                'random_state': self.config.ml.random_state,
                'cross_validation_folds': self.config.ml.cross_validation_folds
            },
            'ai': {
                'model_path': self.config.ai.model_path,
                'training_enabled': self.config.ai.training_enabled,
                'transformer_vocab_size': self.config.ai.transformer_vocab_size,
                'transformer_embedding_dim': self.config.ai.transformer_embedding_dim,
                'transformer_num_heads': self.config.ai.transformer_num_heads,
                'transformer_num_layers': self.config.ai.transformer_num_layers,
                'transformer_dropout_rate': self.config.ai.transformer_dropout_rate,
                'lstm_units': self.config.ai.lstm_units,
                'lstm_dropout_rate': self.config.ai.lstm_dropout_rate,
                'lstm_recurrent_dropout_rate': self.config.ai.lstm_recurrent_dropout_rate,
                'cnn_filters': self.config.ai.cnn_filters,
                'cnn_kernel_size': self.config.ai.cnn_kernel_size,
                'cnn_pool_size': self.config.ai.cnn_pool_size,
                'cnn_dropout_rate': self.config.ai.cnn_dropout_rate,
                'batch_size': self.config.ai.batch_size,
                'epochs': self.config.ai.epochs,
                'learning_rate': self.config.ai.learning_rate,
                'early_stopping_patience': self.config.ai.early_stopping_patience
            },
            'validation': {
                'default_country': self.config.validation.default_country,
                'enable_basic_validation': self.config.validation.enable_basic_validation,
                'enable_pattern_analysis': self.config.validation.enable_pattern_analysis,
                'enable_algorithm_validation': self.config.validation.enable_algorithm_validation,
                'enable_ml_validation': self.config.validation.enable_ml_validation,
                'enable_ai_validation': self.config.validation.enable_ai_validation,
                'enable_api_validation': self.config.validation.enable_api_validation,
                'confidence_threshold': self.config.validation.confidence_threshold,
                'risk_threshold': self.config.validation.risk_threshold,
                'fraud_threshold': self.config.validation.fraud_threshold,
                'basic_weight': self.config.validation.basic_weight,
                'pattern_weight': self.config.validation.pattern_weight,
                'algorithm_weight': self.config.validation.algorithm_weight,
                'ml_weight': self.config.validation.ml_weight,
                'ai_weight': self.config.validation.ai_weight,
                'api_weight': self.config.validation.api_weight
            },
            'logging': {
                'level': self.config.logging.level,
                'format': self.config.logging.format,
                'file_path': self.config.logging.file_path,
                'max_bytes': self.config.logging.max_bytes,
                'backup_count': self.config.logging.backup_count,
                'console_logging': self.config.logging.console_logging
            },
            'debug': self.config.debug,
            'cache_enabled': self.config.cache_enabled,
            'cache_ttl': self.config.cache_ttl,
            'parallel_processing': self.config.parallel_processing,
            'max_workers': self.config.max_workers
        }
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # API Keys
        if os.getenv('NUMVERIFY_API_KEY'):
            self.config.api.numverify_api_key = os.getenv('NUMVERIFY_API_KEY')
        
        if os.getenv('TWILIO_ACCOUNT_SID'):
            self.config.api.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        
        if os.getenv('TWILIO_AUTH_TOKEN'):
            self.config.api.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        if os.getenv('IPQUALITYSCORE_API_KEY'):
            self.config.api.ipqualityscore_api_key = os.getenv('IPQUALITYSCORE_API_KEY')
        
        # Database
        db_path = os.getenv('DATABASE_PATH')
        if db_path:
            self.config.database.path = db_path
        
        # Debug mode
        debug_env = os.getenv('DEBUG')
        if debug_env:
            self.config.debug = debug_env.lower() in ('true', '1', 'yes')
        
        # Logging level
        log_level = os.getenv('LOG_LEVEL')
        if log_level:
            self.config.logging.level = log_level
        
        logger.info("Configuration loaded from environment variables")
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate database path
        try:
            db_dir = os.path.dirname(self.config.database.path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
        except Exception as e:
            errors.append(f"Database path error: {e}")
        
        # Validate model paths
        try:
            if not os.path.exists(self.config.ml.model_path):
                os.makedirs(self.config.ml.model_path)
            if not os.path.exists(self.config.ai.model_path):
                os.makedirs(self.config.ai.model_path)
        except Exception as e:
            errors.append(f"Model path error: {e}")
        
        # Validate thresholds
        if not (0 <= self.config.validation.confidence_threshold <= 1):
            errors.append("Confidence threshold must be between 0 and 1")
        
        if not (0 <= self.config.validation.risk_threshold <= 1):
            errors.append("Risk threshold must be between 0 and 1")
        
        if not (0 <= self.config.validation.fraud_threshold <= 1):
            errors.append("Fraud threshold must be between 0 and 1")
        
        # Validate weights
        total_weight = (self.config.validation.basic_weight + 
                       self.config.validation.pattern_weight + 
                       self.config.validation.algorithm_weight + 
                       self.config.validation.ml_weight + 
                       self.config.validation.ai_weight + 
                       self.config.validation.api_weight)
        
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Validation weights sum to {total_weight}, should be 1.0")
        
        # Validate ML parameters
        if self.config.ml.random_forest_n_estimators < 1:
            errors.append("Random forest n_estimators must be > 0")
        
        if self.config.ml.test_size <= 0 or self.config.ml.test_size >= 1:
            errors.append("Test size must be between 0 and 1")
        
        # Validate AI parameters
        if self.config.ai.transformer_vocab_size < 1:
            errors.append("Transformer vocab size must be > 0")
        
        if self.config.ai.batch_size < 1:
            errors.append("Batch size must be > 0")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True

# Global configuration instance
config_manager = ConfigManager()
