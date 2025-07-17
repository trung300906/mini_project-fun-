# Advanced Phone Validator

A comprehensive Python package for phone number validation with AI, Machine Learning, and multi-method validation approaches.

## 🚀 Features

### Core Validation Methods
- **Basic Validation**: Format checking, country detection, carrier identification
- **Pattern Analysis**: Suspicious pattern detection, entropy analysis, mathematical properties
- **Algorithm Validation**: Luhn algorithm, Bayesian analysis, Monte Carlo validation
- **Machine Learning**: Ensemble learning with Random Forest, SVM, Neural Networks
- **Deep Learning**: Transformer, LSTM, CNN architectures for advanced analysis
- **API Integration**: External validation services with consensus analysis

### Advanced Features
- **Multi-threaded Processing**: Parallel validation for batch operations
- **Intelligent Caching**: Memory and file-based caching with TTL
- **Comprehensive Reporting**: Single, batch, and analytics reports
- **Database Integration**: SQLite-based persistence with full history
- **Fraud Detection**: AI-powered fraud probability calculation
- **Export Capabilities**: CSV, Excel, JSON export formats
- **Configurable**: YAML/JSON configuration with environment variable support

## 📦 Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for full dependencies

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies
```bash
# For deep learning features
pip install tensorflow torch transformers

# For advanced visualization
pip install matplotlib seaborn

# For Excel export
pip install openpyxl

# For YAML configuration
pip install PyYAML
```

## 🎯 Quick Start

### Simple Usage
```python
from phone_validator import quick_validate, batch_validate

# Single phone validation
result = quick_validate("0123456789", "VN")
print(f"Valid: {result['is_valid']}, Status: {result['status']}")

# Batch validation
phones = ["0123456789", "0987654321", "0111222333"]
results = batch_validate(phones, "VN")
for result in results:
    print(f"{result['phone_number']}: {result['status']}")
```

### Advanced Usage
```python
from phone_validator import AdvancedPhoneValidator

# Initialize validator
validator = AdvancedPhoneValidator()

# Single validation with full features
result = validator.validate("0123456789", "VN")
print(f"Status: {result.status}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Risk: {result.risk_score:.2f}")
print(f"Fraud Probability: {result.fraud_probability:.2f}")

# Generate detailed report
report = validator.generate_report(result, "single")
print(report)

# Batch validation
phones = ["0123456789", "0987654321", "0111222333"]
results = validator.validate_batch(phones, "VN")

# Export results
validator.export_results(results, "results.csv", "csv")
validator.export_results(results, "results.xlsx", "excel")
validator.export_results(results, "results.json", "json")
```

## 🔧 Configuration

### Basic Configuration
```python
from phone_validator.utils.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.get_config()

# Update configuration
config_manager.update_config(
    debug=True,
    cache_enabled=True,
    parallel_processing=True
)
```

### YAML Configuration File
```yaml
# config.yaml
database:
  path: "phone_validation.db"
  backup_enabled: true

validation:
  default_country: "VN"
  enable_ml_validation: true
  enable_ai_validation: true
  confidence_threshold: 0.7
  risk_threshold: 0.6

api:
  timeout: 10
  max_retries: 3
  numverify_api_key: "your_api_key"
  twilio_account_sid: "your_sid"
  twilio_auth_token: "your_token"

ml:
  model_path: "models/"
  ensemble_enabled: true
  random_forest_n_estimators: 100

ai:
  model_path: "ai_models/"
  transformer_embedding_dim: 128
  lstm_units: 64

logging:
  level: "INFO"
  file_path: "phone_validator.log"
  console_logging: true
```

### Environment Variables
```bash
export NUMVERIFY_API_KEY="your_api_key"
export TWILIO_ACCOUNT_SID="your_sid"
export TWILIO_AUTH_TOKEN="your_token"
export DATABASE_PATH="custom_db.db"
export DEBUG="true"
export LOG_LEVEL="DEBUG"
```

## 📊 Validation Methods

### 1. Basic Validation
- Format validation using `phonenumbers` library
- Country and region detection
- Carrier identification
- Line type detection (mobile, landline, etc.)

### 2. Pattern Analysis
- Digit distribution analysis
- Entropy calculations
- Mathematical properties (sum, product, etc.)
- Suspicious pattern detection
- Complexity scoring

### 3. Algorithm Validation
- Luhn algorithm verification
- Bayesian spam probability
- Monte Carlo validation
- Statistical analysis
- Checksum validation

### 4. Machine Learning
- Feature extraction (20+ features)
- Multiple ML models:
  - Random Forest
  - Support Vector Machine
  - Neural Networks
  - Gradient Boosting
- Ensemble learning with voting
- Cross-validation and hyperparameter tuning

### 5. Deep Learning
- Transformer architecture for sequence analysis
- LSTM networks for temporal patterns
- CNN for pattern recognition
- Attention mechanisms
- Transfer learning capabilities

### 6. API Integration
- Multiple external API providers:
  - Numverify
  - Twilio Lookup
  - IPQualityScore
- Consensus analysis across providers
- Rate limiting and error handling
- Asynchronous processing

## 📈 Reporting

### Single Phone Report
```python
validator = AdvancedPhoneValidator()
result = validator.validate("0123456789", "VN")
report = validator.generate_report(result, "single")
print(report)
```

### Batch Report
```python
results = validator.validate_batch(phones, "VN")
batch_report = validator.generate_report(results, "batch")
print(batch_report)
```

### Analytics Report
```python
analytics_report = validator.generate_report(results, "analytics")
print(analytics_report)
```

## 🗄️ Database Schema

The system uses SQLite with the following tables:

- **validation_results**: Main validation results
- **ml_features**: Machine learning features
- **ai_predictions**: AI model predictions
- **api_responses**: External API responses
- **validation_history**: Historical validation data
- **statistics**: System statistics

## 🔍 Fraud Detection

The system includes comprehensive fraud detection:

- **Pattern-based**: Suspicious number patterns
- **ML-based**: Machine learning fraud models
- **AI-based**: Deep learning fraud detection
- **Consensus**: Multiple method consensus
- **Risk scoring**: Comprehensive risk assessment

## 🚀 Performance

### Optimization Features
- **Parallel Processing**: Multi-threaded batch validation
- **Intelligent Caching**: Memory and file-based caching
- **Asynchronous API**: Non-blocking API calls
- **Database Indexing**: Optimized database queries
- **Resource Management**: Efficient memory usage

### Benchmarks
- Single validation: ~0.1-0.5 seconds
- Batch validation: ~50-200 phones/second
- Cache hit ratio: >90% for repeated validations
- Memory usage: <100MB for typical workloads

## 📝 Examples

See `examples/usage_example.py` for comprehensive usage examples:

```bash
python examples/usage_example.py
```

## 🔒 Security Features

- **API Key Protection**: Secure API key management
- **Rate Limiting**: Prevent abuse and overuse
- **Data Encryption**: Sensitive data protection
- **Audit Logging**: Comprehensive activity logging
- **Input Validation**: Prevent injection attacks

## 🌍 Supported Countries

The system supports international phone numbers with special optimizations for:
- Vietnam (VN) - Primary focus
- United States (US)
- United Kingdom (GB)
- Germany (DE)
- France (FR)
- And 200+ more countries via phonenumbers library

## 📊 Statistics and Monitoring

```python
# Get system statistics
stats = validator.get_statistics()
print(f"Total validations: {stats['validation_stats']['total_validations']}")
print(f"Cache hit rate: {stats['cache_stats']['memory']['hit_rate']:.1f}%")
print(f"ML predictions: {stats['validation_stats']['ml_predictions']}")
```

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v --cov=phone_validator
```

### Code Formatting
```bash
black phone_validator/
flake8 phone_validator/
mypy phone_validator/
```

### Building Package
```bash
python setup.py sdist bdist_wheel
```

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support, please contact:
- Email: support@phonevalidator.com
- Issues: GitHub Issues
- Documentation: [ReadTheDocs](https://advanced-phone-validator.readthedocs.io/)

## 🙏 Acknowledgments

- `phonenumbers` library for international phone number support
- `scikit-learn` for machine learning capabilities
- `TensorFlow` and `PyTorch` for deep learning features
- Various API providers for external validation services

---

**Advanced Phone Validator** - Making phone validation intelligent, comprehensive, and reliable.
