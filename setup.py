#!/usr/bin/env python

import os

from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "matplotlib",
    "numpy",
    "scipy",
    "scikit-learn",
    "torchdyn>=1.0.6",
    "pot",
    "torchdiffeq",
    "absl-py",
]

version_py = os.path.join(os.path.dirname(__file__), "torchcfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read() if os.path.exists("README.md") else ""

setup(
    name="torchcfm",
    version=version,
    description="FMS²: Unified Flow Matching for Segmentation and Synthesis of Thin Structures — ECCV 2026.",
    url="https://github.com/BabakAsadi94/FMS2",
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
