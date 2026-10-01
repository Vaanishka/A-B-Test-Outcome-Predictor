"""
Core API for Bayesian A/B Test Predictor
High-level interface for running analyses
"""

import pandas as pd
from pathlib import Path
from scripts.simulator import ABTestSimulator
from scripts.visualizer import ABTestVisualizer


def run_analysis(
    data_dir: str,
    output_dir: str = "results",
    confidence_threshold: float = 0.95,
    revenue_per_week: float = 50000,
    config: dict = None,
):
    """
    Run complete Bayesian vs Frequentist A/B test analysis.

    Args:
        data_dir: Directory containing CSV datasets
        output_dir: Where to save results and plots
        confidence_threshold: P(B > A) threshold (default 0.95)
        revenue_per_week: Revenue per week saved (for ROI calculation)
        config: Optional dict with dataset configurations:
                {
                    "dataset_name": {
                        "outcome_type": "binary|continuous|count",
                        "baseline": 0.02
                    }
                }
                If not provided, auto-detects from filename/data.

    Returns:
        DataFrame with simulation results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulations
    simulator = ABTestSimulator(str(data_dir), config=config)
    results_df = simulator.run_all_simulations(confidence_threshold=confidence_threshold)

    # Save results
    results_path = output_dir / "simulation_results.csv"
    simulator.save_results(str(results_path))

    # Generate visualizations
    visualizer = ABTestVisualizer(str(output_dir))
    visualizer.generate_all_plots(
        str(data_dir),
        results_df,
        revenue_per_week=revenue_per_week
    )

    return results_df


def get_summary(results_df: pd.DataFrame) -> dict:
    """
    Get summary statistics from results.

    Args:
        results_df: Results DataFrame from run_analysis()

    Returns:
        Dictionary with key metrics
    """
    return {
        "average_speedup_percent": results_df["speed_up_percent"].mean(),
        "average_weeks_saved": results_df["weeks_saved"].mean(),
        "total_weeks_saved": results_df["weeks_saved"].sum(),
        "accuracy": results_df["accuracy"].mean(),
        "tests_bayesian_faster": (results_df["speed_up_percent"] > 0).sum(),
        "total_tests": len(results_df),
    }
