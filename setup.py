from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="battery-impedance-analyzer",
    version="0.1.0",
    author="Ucaretron Inc.",
    author_email="info@ucaretron.com",
    description="Battery Impedance Analysis System for Electric Vehicle Battery Diagnostics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/battery-impedance-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
