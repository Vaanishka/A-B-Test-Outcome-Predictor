"""
Visualization module for Bayesian vs Frequentist A/B test results
Generates 4 portfolio-ready plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from bayesian_model import BayesianABTest
from frequentist_model import FrequentistABTest


class ABTestVisualizer:
    """Generate publication-ready visualizations for A/B test analysis."""

    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots (default: current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (14, 8)
        plt.rcParams["font.size"] = 10

    def plot_posterior_evolution(self, csv_path: str, outcome_type: str, baseline: float,
                                 save_path: str = None):
        """
        Plot 1: Posterior Evolution Over Time
        Shows how Bayesian belief changes across 30 days.

        Args:
            csv_path: Path to dataset CSV
            outcome_type: "binary", "continuous", or "count"
            baseline: Baseline metric value
            save_path: Where to save plot
        """
        dataset_name = Path(csv_path).stem

        # Load data
        df = pd.read_csv(csv_path)

        # Initialize test
        bayes_test = BayesianABTest(outcome_type=outcome_type, baseline=baseline)

        # Track posteriors day by day
        days = []
        prob_b_better_list = []
        ci_a_lower = []
        ci_a_upper = []
        ci_b_lower = []
        ci_b_upper = []

        for day in range(1, 31):
            day_data = df[df["day"] == day]

            if len(day_data) == 0:
                continue

            data_a = day_data[day_data["variant"] == "A"]["outcome"].values
            data_b = day_data[day_data["variant"] == "B"]["outcome"].values

            if outcome_type == "binary":
                successes_a = int(np.sum(data_a))
                successes_b = int(np.sum(data_b))
                bayes_test.add_binary_data("A", successes_a, len(data_a))
                bayes_test.add_binary_data("B", successes_b, len(data_b))

            elif outcome_type == "continuous":
                bayes_test.add_continuous_data("A", list(data_a))
                bayes_test.add_continuous_data("B", list(data_b))

            elif outcome_type == "count":
                bayes_test.add_count_data("A", list(data_a.astype(int)))
                bayes_test.add_count_data("B", list(data_b.astype(int)))

            bayes_test.update_and_check(day)
            posterior = bayes_test.get_latest_posterior()

            days.append(day)
            prob_b_better_list.append(posterior.prob_b_better)
            ci_a_lower.append(posterior.credible_interval_a[0])
            ci_a_upper.append(posterior.credible_interval_a[1])
            ci_b_lower.append(posterior.credible_interval_b[0])
            ci_b_upper.append(posterior.credible_interval_b[1])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: P(B > A) over time
        ax1.plot(days, prob_b_better_list, "o-", linewidth=2.5, markersize=6, color="#2E86AB", label="P(B > A)")
        ax1.axhline(y=0.95, color="red", linestyle="--", linewidth=2, label="Decision Threshold (95%)")
        ax1.fill_between(days, 0, prob_b_better_list, alpha=0.2, color="#2E86AB")

        if bayes_test.decision_day:
            ax1.axvline(x=bayes_test.decision_day, color="green", linestyle=":", linewidth=2.5,
                       label=f"Decision Day: {bayes_test.decision_day}")

        ax1.set_xlabel("Day", fontsize=12, fontweight="bold")
        ax1.set_ylabel("P(B > A)", fontsize=12, fontweight="bold")
        ax1.set_title(f"Posterior Evolution: {dataset_name}\nP(Variant B is Better than A)",
                     fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc="lower right")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Plot 2: Credible intervals shrinking
        ax2.fill_between(days, ci_a_lower, ci_a_upper, alpha=0.4, color="#A23B72", label="Variant A (95% CI)")
        ax2.fill_between(days, ci_b_lower, ci_b_upper, alpha=0.4, color="#F18F01", label="Variant B (95% CI)")
        ax2.plot(days, [bayes_test.posteriors[i].variant_a_mean for i in range(len(days))],
                color="#A23B72", linewidth=2.5, marker="o", markersize=4, label="Variant A Mean")
        ax2.plot(days, [bayes_test.posteriors[i].variant_b_mean for i in range(len(days))],
                color="#F18F01", linewidth=2.5, marker="s", markersize=4, label="Variant B Mean")

        ax2.set_xlabel("Day", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
        ax2.set_title("Credible Intervals: Uncertainty Shrinking Over Time", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"plot_1_posterior_evolution_{dataset_name}.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_decision_timeline_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot 2: Decision Timeline Comparison (All 12 Tests)
        Side-by-side Bayesian vs Frequentist decision days.

        Args:
            results_df: Results DataFrame from simulator
            save_path: Where to save plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        datasets = results_df["dataset"].values
        bayes_days = results_df["bayesian_decision_day"].fillna(30).values
        freq_days = results_df["frequentist_decision_day"].fillna(30).values

        x = np.arange(len(datasets))
        width = 0.35

        bars1 = ax.bar(x - width/2, bayes_days, width, label="Bayesian", color="#2E86AB", alpha=0.8)
        bars2 = ax.bar(x + width/2, freq_days, width, label="Frequentist", color="#A23B72", alpha=0.8)

        ax.axhline(y=30, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Experiment End")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{int(height)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{int(height)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel("Days to Decision", fontsize=12, fontweight="bold")
        ax.set_title("Decision Timeline: Bayesian vs Frequentist\n(Lower is Better)",
                    fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 35])

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "plot_2_decision_timeline.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_speed_up_heatmap(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot 3: Speed-up Heatmap
        Shows speed-up % for each test, organized by type.

        Args:
            results_df: Results DataFrame from simulator
            save_path: Where to save plot
        """
        # Prepare data
        data = results_df[["dataset", "outcome_type", "speed_up_percent"]].copy()
        data = data.sort_values("outcome_type")

        # Create pivot for heatmap
        heatmap_data = []
        labels = []
        colors_list = []

        outcome_colors = {
            "binary": "#2E86AB",
            "continuous": "#A23B72",
            "count": "#F18F01"
        }

        for _, row in data.iterrows():
            heatmap_data.append(row["speed_up_percent"])
            labels.append(row["dataset"])
            colors_list.append(outcome_colors.get(row["outcome_type"], "#999999"))

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.barh(labels, heatmap_data, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, heatmap_data)):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.1f}%",
                   va="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Speed-up (%)", fontsize=12, fontweight="bold")
        ax.set_title("Speed-up Percentage: Bayesian vs Frequentist\n(Higher is Better)",
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add legend for outcome types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2E86AB", label="Binary", alpha=0.8),
            Patch(facecolor="#A23B72", label="Continuous", alpha=0.8),
            Patch(facecolor="#F18F01", label="Count", alpha=0.8),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "plot_3_speedup_heatmap.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_roi_calculation(self, results_df: pd.DataFrame, revenue_per_week: float = 50000,
                            save_path: str = None):
        """
        Plot 4: ROI Calculation
        Business impact: weeks saved and estimated revenue.

        Args:
            results_df: Results DataFrame from simulator
            revenue_per_week: Assumed revenue per week saved (default $50k)
            save_path: Where to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Weeks saved per test
        weeks_saved = results_df["weeks_saved"].values
        datasets = results_df["dataset"].values

        bars1 = ax1.bar(datasets, weeks_saved, color="#2E86AB", alpha=0.8, edgecolor="black", linewidth=1.5)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax1.set_ylabel("Weeks Saved", fontsize=12, fontweight="bold")
        ax1.set_title("Weeks Saved per Test", fontsize=13, fontweight="bold")
        ax1.set_xticklabels(datasets, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Cumulative ROI
        total_weeks_saved = weeks_saved.sum()
        total_revenue_saved = total_weeks_saved * revenue_per_week

        categories = ["Total Weeks\nSaved", "Revenue\nImpact"]
        values = [total_weeks_saved, total_revenue_saved / 1000]  # Convert to thousands
        colors = ["#2E86AB", "#F18F01"]

        bars2 = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2, width=0.6)

        # Add value labels with proper formatting
        for bar, value, is_revenue in zip(bars2, values, [False, True]):
            height = bar.get_height()
            if is_revenue:
                label_text = f"${value:.0f}k"
            else:
                label_text = f"{value:.1f} weeks"
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label_text, ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax2.set_ylabel("Value", fontsize=12, fontweight="bold")
        ax2.set_title(f"Cumulative Impact Across 12 Tests\n(Assuming ${revenue_per_week:,.0f}/week revenue)",
                     fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add statistics box
        stats_text = f"""
        Average Speed-up: {results_df['speed_up_percent'].mean():.1f}%
        Total Weeks Saved: {total_weeks_saved:.1f}
        Estimated Revenue: ${total_revenue_saved:,.0f}
        Accuracy: {results_df['accuracy'].mean() * 100:.1f}%
        """

        fig.text(0.5, 0.02, stats_text, ha="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                family="monospace", fontweight="bold")

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        if save_path is None:
            save_path = self.output_dir / "plot_4_roi_calculation.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
        plt.close()

    def generate_all_plots(self, data_dir: str, results_df: pd.DataFrame, revenue_per_week: float = 50000):
        """
        Generate all 4 plots.

        Args:
            data_dir: Directory containing CSV datasets
            results_df: Results DataFrame from simulator
            revenue_per_week: Assumed revenue per week saved
        """
        data_dir = Path(data_dir)
        csv_files = sorted(data_dir.glob("*.csv"))

        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*80}\n")

        # Plot 1: Posterior evolution (3 representative datasets)
        print("Plot 1: Posterior Evolution (3 examples)...")
        representative_datasets = [
            (csv_files[0], "binary", 0.02),      # Binary example
            (csv_files[1], "continuous", 48.0),  # Continuous example
            (csv_files[2], "count", 3.5),        # Count example
        ]

        for csv_file, outcome_type, baseline in representative_datasets:
            self.plot_posterior_evolution(str(csv_file), outcome_type, baseline)

        # Plot 2: Decision timeline comparison
        print("Plot 2: Decision Timeline Comparison...")
        self.plot_decision_timeline_comparison(results_df)

        # Plot 3: Speed-up heatmap
        print("Plot 3: Speed-up Heatmap...")
        self.plot_speed_up_heatmap(results_df)

        # Plot 4: ROI calculation
        print("Plot 4: ROI Calculation...")
        self.plot_roi_calculation(results_df, revenue_per_week=revenue_per_week)

        print(f"\n{'='*80}")
        print(f"✓ ALL VISUALIZATIONS COMPLETE")
        print(f"{'='*80}")
        print(f"\nPlots saved to: {self.output_dir}")
