#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="orion-heir-translator",
    version="0.1.0",
    author="FHE Research Team",
    author_email="fhe-research@example.com",
    description="Standalone translator from Orion FHE to HEIR MLIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/orion-heir-translator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "orion": ["orion-fhe"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "orion-heir-translate=orion_heir.tools.orion_heir_driver:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
