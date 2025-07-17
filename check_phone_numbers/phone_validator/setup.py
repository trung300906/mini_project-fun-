#!/usr/bin/env python3
"""
Setup script for Advanced Phone Validator package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README file"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Advanced Phone Validator - A comprehensive phone number validation system with AI and ML"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "phonenumbers>=8.13.0",
            "requests>=2.31.0",
            "aiohttp>=3.8.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "PyYAML>=6.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jinja2>=3.1.0",
            "openpyxl>=3.1.0"
        ]

# Package metadata
NAME = "advanced-phone-validator"
VERSION = "1.0.0"
DESCRIPTION = "Advanced phone number validation with AI and ML"
LONG_DESCRIPTION = read_readme()
AUTHOR = "Phone Validator Team"
AUTHOR_EMAIL = "support@phonevalidator.com"
URL = "https://github.com/your-org/advanced-phone-validator"
LICENSE = "MIT"

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Package requirements
INSTALL_REQUIRES = read_requirements()

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0"
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0"
    ],
    "deep-learning": [
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "transformers>=4.30.0"
    ],
    "redis": [
        "redis>=4.6.0"
    ],
    "postgresql": [
        "psycopg2-binary>=2.9.0"
    ],
    "mongodb": [
        "pymongo>=4.4.0"
    ]
}

# All extras
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Telecommunications Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Communications :: Telephony",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Utilities"
]

# Keywords
KEYWORDS = [
    "phone", "validation", "telephone", "number", "verification",
    "ai", "machine-learning", "deep-learning", "fraud-detection",
    "telecommunications", "carrier", "vietnam", "international"
]

# Entry points
ENTRY_POINTS = {
    "console_scripts": [
        "phone-validator=phone_validator.cli:main",
    ]
}

# Package data
PACKAGE_DATA = {
    "phone_validator": [
        "data/*.json",
        "data/*.csv",
        "models/*.pkl",
        "ai_models/*.h5",
        "templates/*.html",
        "config/*.yaml"
    ]
}

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    
    # Requirements
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Additional options
    zip_safe=False,
    platforms=["any"],
    
    # Project URLs
    project_urls={
        "Documentation": "https://advanced-phone-validator.readthedocs.io/",
        "Source": "https://github.com/your-org/advanced-phone-validator",
        "Tracker": "https://github.com/your-org/advanced-phone-validator/issues",
        "Funding": "https://github.com/sponsors/your-org"
    }
)
