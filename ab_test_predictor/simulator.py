"""
Simulator for running Bayesian vs Frequentist A/B tests on synthetic datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from bayesian_model import BayesianABTest
from frequentist_model import FrequentistABTest


class ABTestSimulator:
    """
    Runs side-by-side simulations of Bayesian vs Frequentist A/B tests.
    Supports custom datasets with optional baseline/outcome type configuration.
    """

    # Default baselines for known datasets
    DEFAULT_BASELINES = {
        "checkout_conversion": 0.02,
        "bundle_aov": 48.0,
        "cart_items": 3.5,
        "premium_adoption": 0.08,
        "engagement_hours": 4.2,
        "onboarding_speed": 14.0,
        "daily_sessions": 1.8,
        "retention_rate": 0.35,
        "video_engagement": 2.1,
        "support_resolution": 18.0,
        "sales_conversion": 0.12,
        "ltv_impact": 1200.0,
    }

    def __init__(self, data_dir: str, config: Dict = None):
        """
        Initialize simulator.

        Args:
            data_dir: Path to directory containing CSV datasets
            config: Optional dict with dataset configs. Format:
                    {
                        "dataset_name": {
                            "outcome_type": "binary|continuous|count",
                            "baseline": 0.02
                        }
                    }
                    If not provided, auto-detects from filename/data.
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.results = []

    @staticmethod
    def _infer_outcome_type(csv_path: str) -> str:
        """Infer outcome type from dataset name and sample data."""
        name = Path(csv_path).stem.lower()

        if "conversion" in name or "adoption" in name or "retention" in name:
            return "binary"
        elif "aov" in name or "hours" in name or "speed" in name or "resolution" in name or "ltv" in name:
            return "continuous"
        elif "items" in name or "sessions" in name or "engagement" in name:
            return "count"
        else:
            return "continuous"  # Default

    @staticmethod
    def _get_baseline(outcome_type: str, csv_path: str) -> float:
        """Get baseline for outcome type based on dataset name."""
        name = Path(csv_path).stem.lower()

        baselines = {
            "checkout_conversion": 0.02,
            "bundle_aov": 48.0,
            "cart_items": 3.5,
            "premium_adoption": 0.08,
            "engagement_hours": 4.2,
            "onboarding_speed": 14.0,
            "daily_sessions": 1.8,
            "retention_rate": 0.35,
            "video_engagement": 2.1,
            "support_resolution": 18.0,
            "sales_conversion": 0.12,
            "ltv_impact": 1200.0,
        }

        for key, baseline in baselines.items():
            if key in name:
                return baseline

        return 1.0  # Default

    def simulate_dataset(self, csv_path: str, confidence_threshold: float = 0.95) -> Dict:
        """
        Simulate one dataset with Bayesian and Frequentist methods.

        Args:
            csv_path: Path to CSV file
            confidence_threshold: P(B > A) threshold for Bayesian decision

        Returns:
            Dictionary with results for both methods
        """
        dataset_name = Path(csv_path).stem
        print(f"\n  Simulating: {dataset_name}...", end=" ")

        # Load data
        df = pd.read_csv(csv_path)

        # Get outcome type and baseline (from config or auto-detect)
        if dataset_name in self.config:
            config_entry = self.config[dataset_name]
            outcome_type = config_entry.get("outcome_type")
            baseline = config_entry.get("baseline")

            if not outcome_type:
                outcome_type = self._infer_outcome_type(csv_path)
            if baseline is None:
                baseline = self._get_baseline(outcome_type, csv_path)
        else:
            outcome_type = self._infer_outcome_type(csv_path)
            baseline = self._get_baseline(outcome_type, csv_path)

        # Initialize models
        bayes_test = BayesianABTest(
            outcome_type=outcome_type,
            baseline=baseline,
            confidence_threshold=confidence_threshold
        )

        freq_test = FrequentistABTest(outcome_type=outcome_type)

        # Simulate day by day
        bayes_decision_day = None
        freq_decision_day = None

        for day in range(1, 31):
            day_data = df[df["day"] == day]

            if len(day_data) == 0:
                continue

            # Separate by variant
            data_a = day_data[day_data["variant"] == "A"]["outcome"].values
            data_b = day_data[day_data["variant"] == "B"]["outcome"].values

            if outcome_type == "binary":
                # Count successes
                successes_a = int(np.sum(data_a))
                successes_b = int(np.sum(data_b))
                bayes_test.add_binary_data("A", successes_a, len(data_a))
                bayes_test.add_binary_data("B", successes_b, len(data_b))
                freq_test.add_binary_data("A", successes_a, len(data_a))
                freq_test.add_binary_data("B", successes_b, len(data_b))

            elif outcome_type == "continuous":
                bayes_test.add_continuous_data("A", list(data_a))
                bayes_test.add_continuous_data("B", list(data_b))
                freq_test.add_continuous_data("A", list(data_a))
                freq_test.add_continuous_data("B", list(data_b))

            elif outcome_type == "count":
                bayes_test.add_count_data("A", list(data_a.astype(int)))
                bayes_test.add_count_data("B", list(data_b.astype(int)))
                freq_test.add_count_data("A", list(data_a.astype(int)))
                freq_test.add_count_data("B", list(data_b.astype(int)))

            # Check if decision reached
            if bayes_decision_day is None:
                if bayes_test.update_and_check(day):
                    bayes_decision_day = day

            if freq_decision_day is None:
                if freq_test.update_and_check(day):
                    freq_decision_day = day

        # Get final results
        bayes_results = bayes_test.get_results_summary()
        freq_results = freq_test.get_results_summary()

        # Calculate speed-up
        if freq_decision_day and bayes_decision_day:
            speed_up_pct = (freq_decision_day - bayes_decision_day) / freq_decision_day * 100
            weeks_saved = (freq_decision_day - bayes_decision_day) / 7
        else:
            speed_up_pct = 0
            weeks_saved = 0

        # Determine accuracy (did both pick same winner?)
        accuracy = 1 if bayes_results.get("winner") == freq_results.get("winner") else 0

        result = {
            "dataset": dataset_name,
            "outcome_type": outcome_type,
            "bayesian_decision_day": bayes_decision_day,
            "frequentist_decision_day": freq_decision_day,
            "bayesian_winner": bayes_results.get("winner"),
            "frequentist_winner": freq_results.get("winner"),
            "speed_up_percent": speed_up_pct,
            "weeks_saved": weeks_saved,
            "accuracy": accuracy,
            "bayesian_confidence": bayes_results.get("final_prob_b_better", 0),
            "frequentist_p_value": freq_results.get("final_p_value", 1.0),
            "bayesian_effect_size": bayes_results.get("variant_b_mean") - bayes_results.get("variant_a_mean"),
            "frequentist_effect_size": freq_results.get("effect_size"),
        }

        print(f"✓ (Bayes: day {bayes_decision_day}, Freq: day {freq_decision_day})")

        return result

    def run_all_simulations(self, confidence_threshold: float = 0.95) -> pd.DataFrame:
        """
        Run simulations on all datasets in directory.

        Args:
            confidence_threshold: P(B > A) threshold for Bayesian

        Returns:
            DataFrame with results for all datasets
        """
        csv_files = sorted(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        print(f"\n{'='*80}")
        print(f"RUNNING SIMULATIONS ON {len(csv_files)} DATASETS")
        print(f"{'='*80}")

        for csv_file in csv_files:
            result = self.simulate_dataset(str(csv_file), confidence_threshold)
            self.results.append(result)

        df_results = pd.DataFrame(self.results)

        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")

        # Print summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Average speed-up: {df_results['speed_up_percent'].mean():.2f}%")
        print(f"  Average weeks saved: {df_results['weeks_saved'].mean():.2f} weeks")
        print(f"  Accuracy (both pick same winner): {df_results['accuracy'].mean() * 100:.1f}%")
        print(f"  Tests with Bayesian faster: {(df_results['speed_up_percent'] > 0).sum()}/{len(df_results)}")

        return df_results

    def save_results(self, output_path: str):
        """Save results to CSV."""
        if not self.results:
            print("No results to save. Run simulations first.")
            return

        df_results = pd.DataFrame(self.results)
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent / "data" / "raw"
    simulator = ABTestSimulator(str(data_dir))
    results_df = simulator.run_all_simulations(confidence_threshold=0.95)
    simulator.save_results(str(Path(__file__).parent / "results.csv"))
