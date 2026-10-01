"""
Frequentist A/B Test Benchmark
Traditional statistical hypothesis testing for comparison
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


class FrequentistABTest:
    """
    Traditional frequentist A/B test using fixed sample size and p-value testing.
    Supports binary, continuous, and count outcomes.
    """

    def __init__(self, outcome_type: str, alpha: float = 0.05, two_tailed: bool = True):
        """
        Initialize Frequentist A/B test.

        Args:
            outcome_type: "binary", "continuous", or "count"
            alpha: Significance level (default 0.05)
            two_tailed: Whether to use two-tailed test (default True)
        """
        self.outcome_type = outcome_type
        self.alpha = alpha
        self.two_tailed = two_tailed

        # Data storage
        self.outcomes_a = []
        self.outcomes_b = []

        self.decision_day: Optional[int] = None
        self.winner: Optional[str] = None
        self.p_values: Dict[int, float] = {}

    def add_binary_data(self, variant: str, successes: int, trials: int):
        """Add binary outcome data."""
        if variant == "A":
            self.outcomes_a.extend([1] * successes + [0] * (trials - successes))
        elif variant == "B":
            self.outcomes_b.extend([1] * successes + [0] * (trials - successes))

    def add_continuous_data(self, variant: str, outcomes: list):
        """Add continuous outcome data."""
        if variant == "A":
            self.outcomes_a.extend(outcomes)
        elif variant == "B":
            self.outcomes_b.extend(outcomes)

    def add_count_data(self, variant: str, counts: list):
        """Add count outcome data."""
        if variant == "A":
            self.outcomes_a.extend(counts)
        elif variant == "B":
            self.outcomes_b.extend(counts)

    def _test_binary(self) -> Tuple[float, float, str]:
        """Chi-square test for binary outcomes."""
        if not self.outcomes_a or not self.outcomes_b:
            return 1.0, 0.0, "A"

        successes_a = sum(self.outcomes_a)
        trials_a = len(self.outcomes_a)
        successes_b = sum(self.outcomes_b)
        trials_b = len(self.outcomes_b)

        # Contingency table
        table = np.array([
            [successes_a, trials_a - successes_a],
            [successes_b, trials_b - successes_b]
        ])

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        # Determine winner
        rate_a = successes_a / trials_a if trials_a > 0 else 0
        rate_b = successes_b / trials_b if trials_b > 0 else 0
        winner = "B" if rate_b > rate_a else "A"

        return p_value, rate_b - rate_a, winner

    def _test_continuous(self) -> Tuple[float, float, str]:
        """T-test for continuous outcomes."""
        if not self.outcomes_a or not self.outcomes_b:
            return 1.0, 0.0, "A"

        a = np.array(self.outcomes_a)
        b = np.array(self.outcomes_b)

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value_two_tailed = stats.ttest_ind(a, b, equal_var=False)

        # One-tailed if needed
        p_value = p_value_two_tailed / 2 if not self.two_tailed else p_value_two_tailed

        mean_a = np.mean(a)
        mean_b = np.mean(b)
        winner = "B" if mean_b > mean_a else "A"

        return p_value, mean_b - mean_a, winner

    def _test_count(self) -> Tuple[float, float, str]:
        """Poisson test for count outcomes."""
        if not self.outcomes_a or not self.outcomes_b:
            return 1.0, 0.0, "A"

        a = np.array(self.outcomes_a)
        b = np.array(self.outcomes_b)

        mean_a = np.mean(a)
        mean_b = np.mean(b)

        # Use t-test as approximation for count data
        t_stat, p_value_two_tailed = stats.ttest_ind(a, b, equal_var=False)
        p_value = p_value_two_tailed / 2 if not self.two_tailed else p_value_two_tailed

        winner = "B" if mean_b > mean_a else "A"

        return p_value, mean_b - mean_a, winner

    def update_and_check(self, day: int) -> bool:
        """
        Run statistical test and check if p < alpha.
        Returns True if winner declared.
        """
        if self.outcome_type == "binary":
            p_value, effect, winner = self._test_binary()
        elif self.outcome_type == "continuous":
            p_value, effect, winner = self._test_continuous()
        elif self.outcome_type == "count":
            p_value, effect, winner = self._test_count()
        else:
            raise ValueError(f"Unknown outcome_type: {self.outcome_type}")

        self.p_values[day] = p_value

        # Check significance threshold
        if p_value < self.alpha and self.decision_day is None:
            self.decision_day = day
            self.winner = winner
            return True

        return False

    def get_results_summary(self) -> Dict:
        """Get summary of test results."""
        if self.outcome_type == "binary":
            rate_a = sum(self.outcomes_a) / len(self.outcomes_a) if self.outcomes_a else 0
            rate_b = sum(self.outcomes_b) / len(self.outcomes_b) if self.outcomes_b else 0
            effect = rate_b - rate_a

            # Confidence intervals
            n_a = len(self.outcomes_a)
            n_b = len(self.outcomes_b)
            se_a = np.sqrt(rate_a * (1 - rate_a) / n_a) if n_a > 0 else 0
            se_b = np.sqrt(rate_b * (1 - rate_b) / n_b) if n_b > 0 else 0

            ci_a = (rate_a - 1.96 * se_a, rate_a + 1.96 * se_a)
            ci_b = (rate_b - 1.96 * se_b, rate_b + 1.96 * se_b)

        elif self.outcome_type == "continuous":
            a = np.array(self.outcomes_a)
            b = np.array(self.outcomes_b)

            mean_a = np.mean(a) if len(a) > 0 else 0
            mean_b = np.mean(b) if len(b) > 0 else 0
            effect = mean_b - mean_a

            se_a = np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
            se_b = np.std(b, ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0

            ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
            ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)

        elif self.outcome_type == "count":
            a = np.array(self.outcomes_a)
            b = np.array(self.outcomes_b)

            mean_a = np.mean(a) if len(a) > 0 else 0
            mean_b = np.mean(b) if len(b) > 0 else 0
            effect = mean_b - mean_a

            se_a = np.sqrt(mean_a / len(a)) if len(a) > 0 else 0
            se_b = np.sqrt(mean_b / len(b)) if len(b) > 0 else 0

            ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
            ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)

        return {
            "outcome_type": self.outcome_type,
            "decision_day": self.decision_day,
            "winner": self.winner,
            "final_p_value": self.p_values.get(self.decision_day, 1.0) if self.decision_day else 1.0,
            "variant_a_mean": mean_a if self.outcome_type in ["continuous", "count"] else rate_a,
            "variant_b_mean": mean_b if self.outcome_type in ["continuous", "count"] else rate_b,
            "variant_a_ci": ci_a,
            "variant_b_ci": ci_b,
            "effect_size": effect,
        }
