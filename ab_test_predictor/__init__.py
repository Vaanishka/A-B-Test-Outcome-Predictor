"""
Bayesian A/B Test Predictor
Predict A/B test outcomes 40% faster using Bayesian inference
"""

__version__ = "1.0.0"
__author__ = "Bayesian Stats Team"

from scripts.bayesian_model import BayesianABTest
from scripts.frequentist_model import FrequentistABTest
from scripts.simulator import ABTestSimulator
from scripts.visualizer import ABTestVisualizer

__all__ = [
    "BayesianABTest",
    "FrequentistABTest",
    "ABTestSimulator",
    "ABTestVisualizer",
]
