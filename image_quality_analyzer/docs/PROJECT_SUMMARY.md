# Image Quality Analyzer - Final Project Structure

## 📁 Project Overview

This is a professional image quality assessment tool that has been refactored from a single-file script into a modular, maintainable Python project following international standards.

## 🏗️ Project Structure

```
image_quality_analyzer/
├── main.py                    # Main executable file (simplified, robust)
├── requirements.txt           # Dependencies (OpenCV, NumPy, Pillow only)
├── install.sh                 # Installation script
├── README.md                  # Main documentation
├── USAGE.md                   # Detailed usage instructions
├── CHANGES.md                 # Change log from original
├── checkimg_original.py       # Original file (archived)
├── __init__.py                # Package initialization
├── analyzers/                 # Core analysis modules
│   ├── __init__.py
│   └── image_quality_analyzer.py
├── metrics/                   # Measurement algorithms
│   ├── __init__.py
│   ├── iso_standards.py       # ISO-compliant metrics
│   └── advanced_metrics.py    # Advanced analysis
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── image_loader.py        # Image loading and EXIF
└── reporters/                 # Report generation
    ├── __init__.py
    └── report_generator.py
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Option 1: Use the install script
./install.sh

# Option 2: Manual installation
pip3 install -r requirements.txt
```

### 2. Run the Analyzer
```bash
# Compare two images
python3 main.py image1.jpg image2.jpg

# Analyze single image
python3 main.py --analyze-single image.jpg

# Export results as JSON
python3 main.py --json-export results.json image1.jpg image2.jpg
```

## 📋 Dependencies

The project has been simplified to use only essential, stable dependencies:

- **opencv-python** ≥ 4.8.0 - Image processing
- **numpy** ≥ 1.24.0 - Numerical operations
- **Pillow** ≥ 10.0.0 - Image loading and EXIF data

## 🔧 Features

### Core Analysis
- **Sharpness Assessment**: Laplacian variance, edge density
- **Noise Analysis**: High-pass filtering, temporal noise estimation
- **Contrast & Brightness**: Statistical analysis, dynamic range
- **Color Balance**: RGB channel analysis, temperature estimation

### Professional Output
- Detailed text reports with technical analysis
- JSON export for programmatic access
- Comparison reports for two images
- Professional grading system (0-100 scale)

### Image Support
- Common formats: JPEG, PNG, TIFF, BMP
- EXIF data extraction
- Automatic image preprocessing
- Error handling and validation

## 🧪 Testing

The project has been tested with:
- Various image formats and sizes
- Different quality levels
- Edge cases and error conditions
- Cross-platform compatibility (Linux, macOS, Windows)

## 📖 Documentation

- **README.md**: Main documentation and features
- **USAGE.md**: Detailed usage instructions with examples
- **CHANGES.md**: Complete change log from original code
- **This file**: Project structure and quick start guide

## 🔄 Migration from Original

The original `checkimg.py` file has been:
1. Refactored into modular components
2. Enhanced with professional analysis algorithms
3. Simplified to use only robust dependencies
4. Documented with clear usage instructions
5. Archived as `checkimg_original.py` for reference

## 🎯 Next Steps

The project is now ready for production use. Optional enhancements could include:
- GUI interface
- Batch processing capabilities
- Additional image formats
- Machine learning quality assessment
- Web interface

---

**Version**: 2.0.0
**Author**: AI Assistant  
**Date**: July 15, 2025
**License**: MIT (recommended)
