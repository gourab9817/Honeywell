"""
Setup configuration for F&B Anomaly Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fnb-anomaly-detection",
    version="1.0.0",
    author="Your Team Name",
    author_email="team@fnb-anomaly.com",
    description="AI-powered anomaly detection system for F&B manufacturing processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fnb-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Proprietary",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.24.0",
        ],
        "visualization": [
            "plotly>=5.16.0",
            "dash>=2.11.0",
            "bokeh>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fnb-anomaly=app.app:main",
            "fnb-train=src.model_trainer:main",
            "fnb-predict=src.predictor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.css", "*.js", "*.png", "*.jpg", "*.json"],
    },
    zip_safe=False,
)