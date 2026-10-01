from setuptools import setup, find_packages

setup(
    name="ab-test-predictor",
    version="1.0.0",
    description="Predict A/B test outcomes faster using Bayesian inference",
    author="Bayesian Stats Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "ab-test-predictor=ab_test_predictor.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
