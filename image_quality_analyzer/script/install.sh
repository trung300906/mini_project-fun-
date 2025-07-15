#!/bin/bash

# Installation script for Image Quality Analyzer
# This script sets up the environment and installs dependencies

echo "🔧 Installing Image Quality Analyzer..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed. Please install pip3."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Make main.py executable
chmod +x main.py

echo "✅ Installation complete!"
echo ""
echo "🚀 Usage:"
echo "  python3 main.py image1.jpg image2.jpg    # Compare two images"
echo "  python3 main.py --analyze-single image.jpg  # Analyze single image"
echo ""
echo "📖 For detailed usage instructions, see USAGE.md"
