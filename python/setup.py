#!/usr/bin/env python3
import os
from setuptools import setup, find_packages

# Check for VERSION in the current directory first (sdist/wheel)
version_file = os.path.join(os.path.dirname(__file__), "VERSION")
if not os.path.exists(version_file):
    # Fallback to root VERSION file (dev)
    version_file = os.path.join(os.path.dirname(__file__), "..", "VERSION")

with open(version_file) as f:
    version = f.read().strip()

setup(
    name="docc",
    version=version,
    description="A JIT compiler for Numpy-based Python programs targeting various hardware backends.",
    author="Daisytuner",
    python_requires=">=3.10, <3.13",
    packages=find_packages(),
    package_data={"docc": ["*.so", "*.pyd", "*.a", "include/**/*"]},
    include_package_data=True,
    has_ext_modules=lambda: True,
    install_requires=[
        "numpy>=1.19.0",
    ],
)
