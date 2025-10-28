"""
Setup script for Physics-Informed Conformal Prediction (PICP) package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Physics-Informed Conformal Prediction for capacitor lifetime forecasting"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "PyQt5>=5.15.0",
        "psutil>=5.8.0"
    ]

setup(
    name="picp",
    version="1.0.0",
    author="PICP Development Team",
    author_email="picp@example.com",
    description="Physics-Informed Conformal Prediction for capacitor lifetime forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/picp",
    
    packages=find_packages(),
    py_modules=[
        "core",
        "excel_reader",
        "picp_core",
        "picp_ui", 
        "picp_plotting",
        "picp_logger",
        "main"
    ],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.7",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0"
        ],
        "build": [
            "pyinstaller>=4.5.0"
        ]
    },
    
    entry_points={
        "console_scripts": [
            "picp=main:main",
            "picp-gui=main:launch_gui", 
            "picp-cli=main:run_cli",
            "picp-test=main:run_tests"
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
    
    keywords=[
        "conformal prediction",
        "bayesian inference", 
        "physics-informed machine learning",
        "uncertainty quantification",
        "capacitor lifetime",
        "reliability analysis",
        "time series forecasting"
    ],
    
    project_urls={
        "Documentation": "https://picp.readthedocs.io/",
        "Source": "https://github.com/example/picp",
        "Tracker": "https://github.com/example/picp/issues",
    }
)