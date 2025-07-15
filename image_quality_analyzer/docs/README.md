# Professional Image Quality Analyzer v2.0.0 (Advanced Edition)

A comprehensive image quality assessment tool with advanced pixel-level analysis and professional photography evaluation techniques.

## 🌟 Advanced Features

### Pixel-Level Analysis
- **Per-pixel noise analysis** with SNR mapping across regions
- **High-frequency noise detection** using advanced filtering
- **Color noise analysis** for RGB channels independently
- **Gaussian and median noise estimation**

### Depth of Field & Bokeh
- **Depth separation analysis** between foreground and background
- **Bokeh quality assessment** with circular shape detection
- **Focus transition mapping** across image regions
- **Background blur uniformity** evaluation

### Object Detection & Background
- **Contour-based object detection** with complexity analysis
- **Background uniformity assessment**
- **Object-background separation** using advanced thresholding
- **Edge density mapping** for detail analysis

### Advanced Color Science
- **Multi-colorspace analysis** (RGB, HSV, LAB)
- **Color temperature estimation** with white balance error
- **Saturation uniformity** across image regions
- **Color harmony analysis** with dominant hue detection

### Texture & Detail Analysis
- **Local Binary Pattern (LBP)** texture analysis
- **Gabor filter responses** for texture orientation
- **Texture energy** and uniformity measurements
- **Edge density** per region analysis

### Professional Sharpness Metrics
- **Laplacian variance** for global sharpness
- **Sobel gradient magnitude** analysis
- **Tenengrad sharpness** measurement
- **Brenner focus measure**
- **Regional sharpness** distribution analysis

### Advanced Exposure Analysis
- **Multi-zone exposure** (shadows, midtones, highlights)
- **Dynamic range** calculation
- **Clipping analysis** (under/over exposure)
- **Regional exposure uniformity**

### Composition Analysis
- **Rule of thirds** balance evaluation
- **Center emphasis** analysis
- **Leading lines** detection
- **Symmetry** assessment

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- opencv-python>=4.8.0
- numpy>=1.24.0
- Pillow>=10.0.0

## 🚀 Usage

### Basic Usage

```bash
# Compare two images
python main.py image1.jpg image2.jpg

# Analyze single image
python main.py --analyze-single image.jpg

# Generate detailed reports
python main.py --report-dir ./reports image1.jpg image2.jpg
```

### Command Line Options

```bash
python main.py -h  # Show help
python main.py --version  # Show version
python main.py --analyze-single IMAGE  # Analyze single image
python main.py --report-dir DIR  # Save reports to directory
```

## 📊 Analysis Metrics

### Sharpness Assessment (ISO 12233)
- MTF50 (Modulation Transfer Function at 50%)
- Spatial Frequency Response (horizontal/vertical)
- Acutance (perceptual sharpness)
- Edge density analysis

### Noise Analysis (ISO 15739)
- Temporal noise estimation
- Spatial noise measurement
- Signal-to-Noise Ratio (SNR)
- Visual noise (perceptual)
- Fixed pattern noise

### Color Reproduction (ISO 14524)
- Color accuracy (Delta E approximation)
- Color gamut coverage
- Color consistency across image
- White balance accuracy
- Color temperature estimation

### Distortion Analysis (ISO 20462)
- Geometric distortion (barrel/pincushion)
- Chromatic aberration
- Vignetting measurements
- Perspective distortion

### Advanced Perceptual Metrics
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Gradient similarity
- Texture similarity
- Perceptual sharpness

### Bokeh Quality Analysis
- Bokeh smoothness
- Bokeh shape quality
- Depth transition quality
- Focus peaking analysis
- Depth of field ratio

### Texture Analysis
- GLCM (Gray-Level Co-occurrence Matrix) features
- Gabor filter responses
- Local Binary Pattern analysis
- Fractal dimension
- Texture energy

### Color Science
- CRI (Color Rendering Index) approximation
- Metamerism index
- Color constancy
- Skin tone naturalness
- Memory color accuracy

## 🏗️ Project Structure

```
image_quality_analyzer/
├── __init__.py
├── main.py                    # Main application
├── requirements.txt           # Dependencies
├── README.md                 # Documentation
│
├── analyzers/                # Core analyzer modules
│   ├── __init__.py
│   └── image_quality_analyzer.py
│
├── metrics/                  # Metric calculation modules
│   ├── __init__.py
│   ├── iso_standards.py      # ISO standard metrics
│   └── advanced_metrics.py   # Advanced quality metrics
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   └── image_loader.py       # Image loading and EXIF
│
└── reporters/                # Report generation
    ├── __init__.py
    └── report_generator.py   # Report and visualization
```

## 🎯 Quality Scoring System

### Overall Score (0-100)
- **95-100**: Outstanding (A++)
- **90-94**: Excellent (A+)
- **85-89**: Very Good (A)
- **80-84**: Good (A-)
- **75-79**: Above Average (B+)
- **70-74**: Average (B)
- **65-69**: Below Average (B-)
- **60-64**: Fair (C+)
- **55-59**: Poor (C)
- **50-54**: Very Poor (C-)
- **0-49**: Unacceptable (F)

### Scoring Weights
- Sharpness: 20%
- Noise: 15%
- Color Accuracy: 12%
- Exposure: 12%
- Distortion: 10%
- Perceptual: 8%
- Bokeh: 8%
- Texture: 8%
- Color Science: 7%

## 📈 Example Output

```
🖼️  PROFESSIONAL IMAGE QUALITY ANALYZER v2.0.0
Following international standards: ISO 12233, ISO 15739, ISO 14524, ISO 20462

🔍 Analyzing image: IMG_001.jpg
   📏 Calculating ISO standards metrics...
   🧠 Calculating advanced metrics...
   📊 Calculating overall quality score...
   ✅ Analysis completed!

📊 ANALYSIS RESULTS - IMG_001.jpg
📐 Dimensions: 4000x3000 (12.0 MP)

📷 Camera Information:
   • Camera Make: Canon
   • Camera Model: EOS R5
   • Aperture: f/2.8
   • Shutter Speed: 1/125s
   • ISO: 400

🏆 Overall Quality Score: 87.5/100
🎯 Grade: Very Good (A)

📈 Individual Scores:
   • Sharpness: 89.2/100 🔵 Very Good
   • Noise: 85.1/100 🔵 Very Good
   • Color Accuracy: 92.3/100 🟢 Excellent
   • Exposure: 88.7/100 🔵 Very Good
   • Distortion: 84.5/100 🔵 Very Good
```

## 🔧 API Usage

```python
from image_quality_analyzer import ImageQualityAnalyzer, ReportGenerator

# Initialize analyzer
analyzer = ImageQualityAnalyzer()
reporter = ReportGenerator()

# Analyze single image
analysis = analyzer.analyze_image("image.jpg")
print(f"Overall score: {analysis['overall_score']['total_score']}")

# Compare two images
comparison = analyzer.compare_images("image1.jpg", "image2.jpg")

# Generate reports
report_path = reporter.generate_detailed_report(analysis, "image.jpg")
json_path = reporter.save_analysis_json(analysis, "image.jpg")
```

## 📝 Technical Notes

- Analysis follows international photography standards
- Scores are calibrated for professional photography
- Results may vary depending on viewing conditions
- Some metrics require minimum image resolution (100x100 pixels)
- EXIF data is used when available for enhanced analysis

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 coding standards
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- ISO 12233:2017 - Photography - Electronic still picture imaging - Resolution and spatial frequency responses
- ISO 15739:2017 - Photography - Electronic still-picture imaging - Noise measurements
- ISO 14524:2009 - Photography - Electronic still-picture imaging - Methods for measuring opto-electronic conversion functions
- ISO 20462:2005 - Photography - Electronic still-picture imaging - Distortion measurements

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

**Professional Image Quality Analyzer v2.0.0** - Bringing international standards to image quality assessment.
