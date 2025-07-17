#!/usr/bin/env python3
"""
Example usage of Advanced Phone Validator
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phone_validator import AdvancedPhoneValidator, quick_validate, batch_validate
from phone_validator.utils.config import ConfigManager

def example_single_validation():
    """Example: Single phone validation"""
    print("=== SINGLE PHONE VALIDATION ===")
    
    # Method 1: Using quick_validate (simple)
    result = quick_validate("0123456789", "VN")
    print(f"Quick validation result: {result}")
    
    # Method 2: Using AdvancedPhoneValidator (full features)
    validator = AdvancedPhoneValidator()
    full_result = validator.validate("0123456789", "VN")
    
    # Generate report
    report = validator.generate_report(full_result, "single")
    print(f"Full validation report:\n{report}")
    
    print("\n" + "="*50 + "\n")

def example_batch_validation():
    """Example: Batch phone validation"""
    print("=== BATCH PHONE VALIDATION ===")
    
    phone_numbers = [
        "0123456789",
        "0987654321",
        "0111222333",
        "0444555666",
        "0777888999",
        "invalid_phone",
        "84123456789"
    ]
    
    # Method 1: Using batch_validate (simple)
    simple_results = batch_validate(phone_numbers, "VN")
    print(f"Simple batch results: {len(simple_results)} phones validated")
    
    # Method 2: Using AdvancedPhoneValidator (full features)
    validator = AdvancedPhoneValidator()
    full_results = validator.validate_batch(phone_numbers, "VN")
    
    # Generate batch report
    batch_report = validator.generate_report(full_results, "batch")
    print(f"Batch validation report:\n{batch_report}")
    
    # Export results
    validator.export_results(full_results, "batch_results.csv", "csv")
    print("Results exported to batch_results.csv")
    
    print("\n" + "="*50 + "\n")

def example_configuration():
    """Example: Configuration management"""
    print("=== CONFIGURATION MANAGEMENT ===")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Current configuration:")
    print(f"- Database path: {config.database.path}")
    print(f"- ML enabled: {config.validation.enable_ml_validation}")
    print(f"- AI enabled: {config.validation.enable_ai_validation}")
    print(f"- Cache enabled: {config.cache_enabled}")
    print(f"- Debug mode: {config.debug}")
    
    # Update configuration
    config_manager.update_config(debug=True)
    print("Debug mode enabled")
    
    # Load from environment variables
    config_manager.load_from_env()
    print("Configuration loaded from environment")
    
    print("\n" + "="*50 + "\n")

def example_advanced_features():
    """Example: Advanced features"""
    print("=== ADVANCED FEATURES ===")
    
    validator = AdvancedPhoneValidator()
    
    # Validate with specific methods only
    result1 = validator.validate(
        "0123456789", 
        "VN", 
        validation_methods=['basic', 'pattern', 'algorithm']
    )
    print(f"Validation with basic methods: {result1.status}")
    
    # Validate without cache
    result2 = validator.validate("0123456789", "VN", use_cache=False)
    print(f"Validation without cache: {result2.status}")
    
    # Get statistics
    stats = validator.get_statistics()
    print(f"Validator statistics: {stats}")
    
    # Clear cache
    validator.clear_cache()
    print("Cache cleared")
    
    print("\n" + "="*50 + "\n")

def example_error_handling():
    """Example: Error handling"""
    print("=== ERROR HANDLING ===")
    
    validator = AdvancedPhoneValidator()
    
    # Test with invalid phone numbers
    invalid_phones = [
        "",
        "abc123",
        "123",
        "+++++",
        "0000000000000000000000",
        None
    ]
    
    for phone in invalid_phones:
        try:
            if phone is not None:
                result = validator.validate(phone, "VN")
                print(f"Phone: {phone} -> Status: {result.status}, Error: {result.error}")
            else:
                print(f"Phone: {phone} -> Skipped (None)")
        except Exception as e:
            print(f"Phone: {phone} -> Exception: {str(e)}")
    
    print("\n" + "="*50 + "\n")

def example_reporting():
    """Example: Reporting and analytics"""
    print("=== REPORTING AND ANALYTICS ===")
    
    validator = AdvancedPhoneValidator()
    
    # Generate test data
    test_phones = [
        "0123456789", "0987654321", "0111222333",
        "0444555666", "0777888999", "invalid_phone"
    ]
    
    results = validator.validate_batch(test_phones, "VN")
    
    # Generate different types of reports
    single_report = validator.generate_report(results[0], "single")
    batch_report = validator.generate_report(results, "batch")
    analytics_report = validator.generate_report(results, "analytics")
    
    print("Single report generated")
    print("Batch report generated")
    print("Analytics report generated")
    
    # Export in different formats
    validator.export_results(results, "results.csv", "csv")
    validator.export_results(results, "results.json", "json")
    # validator.export_results(results, "results.xlsx", "excel")  # Requires openpyxl
    
    print("Results exported in multiple formats")
    
    print("\n" + "="*50 + "\n")

def main():
    """Main function to run all examples"""
    print("ADVANCED PHONE VALIDATOR - EXAMPLES")
    print("=" * 60)
    
    try:
        example_single_validation()
        example_batch_validation()
        example_configuration()
        example_advanced_features()
        example_error_handling()
        example_reporting()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
