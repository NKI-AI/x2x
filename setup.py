# setup.py
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="x2x",
    version="0.1.0",
    author="Yoni Schirris",
    description="X2X: Explainable AI to Explained AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="LGPL-2.1+",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        "flask",
        "requests",
        "openai==1.60.2",
        "openslide-bin",
        "dlup==0.3.38",
        "pandas",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "numpy==1.25.2",
        "grad-cam==1.5.4",
    ],
)
