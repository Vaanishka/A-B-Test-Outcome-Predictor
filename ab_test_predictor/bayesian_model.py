"""
Bayesian A/B Test Inference Engine
Core module for sequential Bayesian analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class BayesianPosterior:
    """Represents posterior distribution at a given point in time."""
    day: int
    variant_a_mean: float
    variant_b_mean: float
    variant_a_std: float
    variant_b_std: float
    prob_b_better: float
    credible_interval_a: Tuple[float, float]
    credible_interval_b: Tuple[float, float]


class BayesianABTest:
    """
    Bayesian A/B test inference using conjugate priors.
    Supports binary, continuous, and count outcomes.
    """

    def __init__(self, outcome_type: str, baseline: float, confidence_threshold: float = 0.95):
        """
        Initialize Bayesian A/B test.

        Args:
            outcome_type: "binary", "continuous", or "count"
            baseline: Expected baseline metric value
            confidence_threshold: P(B > A) threshold for declaring winner (default 0.95)
        """
        self.outcome_type = outcome_type
        self.baseline = baseline
        self.confidence_threshold = confidence_threshold

        # Initialize sufficient statistics for conjugate priors
        if outcome_type == "binary":
            self._init_binary()
        elif outcome_type == "continuous":
            self._init_continuous()
        elif outcome_type == "count":
            self._init_count()
        else:
            raise ValueError(f"Unknown outcome_type: {outcome_type}")

        self.posteriors: List[BayesianPosterior] = []
        self.decision_day: Optional[int] = None
        self.winner: Optional[str] = None

    def _init_binary(self):
        """Initialize Beta-Binomial conjugate prior."""
        # Weak prior: Beta(1, 1)
        self.prior_alpha_a = 1.0
        self.prior_beta_a = 1.0
        self.prior_alpha_b = 1.0
        self.prior_beta_b = 1.0

        # Sufficient statistics
        self.successes_a = 0
        self.trials_a = 0
        self.successes_b = 0
        self.trials_b = 0

    def _init_continuous(self):
        """Initialize Normal-InverseGamma conjugate prior."""
        # Weak informative prior
        self.prior_mu_a = self.baseline
        self.prior_lambda_a = 1.0  # Prior precision
        self.prior_alpha_a = 1.0  # Shape
        self.prior_beta_a = 1.0  # Rate

        self.prior_mu_b = self.baseline
        self.prior_lambda_b = 1.0
        self.prior_alpha_b = 1.0
        self.prior_beta_b = 1.0

        # Sufficient statistics
        self.sum_x_a = 0.0
        self.sum_x2_a = 0.0
        self.n_a = 0

        self.sum_x_b = 0.0
        self.sum_x2_b = 0.0
        self.n_b = 0

    def _init_count(self):
        """Initialize Poisson-Gamma conjugate prior."""
        # Weak informative prior: Gamma(α, β)
        self.prior_alpha_a = self.baseline
        self.prior_beta_a = 1.0

        self.prior_alpha_b = self.baseline
        self.prior_beta_b = 1.0

        # Sufficient statistics
        self.sum_counts_a = 0
        self.n_a = 0
        self.sum_counts_b = 0
        self.n_b = 0

    def add_binary_data(self, variant: str, successes: int, trials: int):
        """Add binary outcome data."""
        if variant == "A":
            self.successes_a += successes
            self.trials_a += trials
        elif variant == "B":
            self.successes_b += successes
            self.trials_b += trials
        else:
            raise ValueError(f"Invalid variant: {variant}")

    def add_continuous_data(self, variant: str, outcomes: List[float]):
        """Add continuous outcome data."""
        outcomes = np.array(outcomes)
        if variant == "A":
            self.sum_x_a += np.sum(outcomes)
            self.sum_x2_a += np.sum(outcomes ** 2)
            self.n_a += len(outcomes)
        elif variant == "B":
            self.sum_x_b += np.sum(outcomes)
            self.sum_x2_b += np.sum(outcomes ** 2)
            self.n_b += len(outcomes)
        else:
            raise ValueError(f"Invalid variant: {variant}")

    def add_count_data(self, variant: str, counts: List[int]):
        """Add count outcome data."""
        counts = np.array(counts)
        if variant == "A":
            self.sum_counts_a += np.sum(counts)
            self.n_a += len(counts)
        elif variant == "B":
            self.sum_counts_b += np.sum(counts)
            self.n_b += len(counts)
        else:
            raise ValueError(f"Invalid variant: {variant}")

    def _get_binary_posterior(self) -> Tuple[Dict, Dict]:
        """Compute Beta posterior for binary outcomes."""
        # Posterior = Beta(prior_α + successes, prior_β + failures)
        alpha_a = self.prior_alpha_a + self.successes_a
        beta_a = self.prior_beta_a + (self.trials_a - self.successes_a)

        alpha_b = self.prior_alpha_b + self.successes_b
        beta_b = self.prior_beta_b + (self.trials_b - self.successes_b)

        # Posterior mean and std
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)

        var_a = (alpha_a * beta_a) / ((alpha_a + beta_a) ** 2 * (alpha_a + beta_a + 1))
        var_b = (alpha_b * beta_b) / ((alpha_b + beta_b) ** 2 * (alpha_b + beta_b + 1))

        # Clamp variance to prevent NaN from sqrt
        var_a = max(var_a, 0.0001)
        var_b = max(var_b, 0.0001)

        std_a = np.sqrt(var_a)
        std_b = np.sqrt(var_b)

        # Credible intervals (95%)
        ci_a = (stats.beta.ppf(0.025, alpha_a, beta_a), stats.beta.ppf(0.975, alpha_a, beta_a))
        ci_b = (stats.beta.ppf(0.025, alpha_b, beta_b), stats.beta.ppf(0.975, alpha_b, beta_b))

        return {
            "mean": mean_a,
            "std": std_a,
            "ci": ci_a,
            "alpha": alpha_a,
            "beta": beta_a,
        }, {
            "mean": mean_b,
            "std": std_b,
            "ci": ci_b,
            "alpha": alpha_b,
            "beta": beta_b,
        }

    def _get_continuous_posterior(self) -> Tuple[Dict, Dict]:
        """Compute Normal-InverseGamma posterior for continuous outcomes."""
        # Update A
        if self.n_a > 0:
            mean_a = self.sum_x_a / self.n_a
            var_a = (self.sum_x2_a - self.sum_x_a ** 2 / self.n_a) / max(self.n_a - 1, 1)
            var_a = max(var_a, 0.001)  # Avoid zero variance

            # Posterior parameters
            lambda_a_post = self.prior_lambda_a + self.n_a
            mu_a_post = (self.prior_lambda_a * self.prior_mu_a + self.sum_x_a) / lambda_a_post
            alpha_a_post = self.prior_alpha_a + self.n_a / 2
            beta_a_post = self.prior_beta_a + 0.5 * (self.sum_x2_a + self.prior_lambda_a * self.prior_mu_a ** 2 -
                                                      lambda_a_post * mu_a_post ** 2)
            beta_a_post = max(beta_a_post, 0.001)

            # Posterior mean (t-distribution for uncertainty)
            df_a = 2 * alpha_a_post
            t_std_a = np.sqrt(beta_a_post * (1 + 1 / lambda_a_post) / alpha_a_post)
            ci_a = (mu_a_post - 1.96 * t_std_a, mu_a_post + 1.96 * t_std_a)
        else:
            mean_a = self.prior_mu_a
            t_std_a = np.sqrt(self.prior_beta_a / self.prior_alpha_a)
            ci_a = (mean_a - 1.96 * t_std_a, mean_a + 1.96 * t_std_a)
            lambda_a_post = self.prior_lambda_a
            alpha_a_post = self.prior_alpha_a
            beta_a_post = self.prior_beta_a

        # Update B (same logic)
        if self.n_b > 0:
            mean_b = self.sum_x_b / self.n_b
            var_b = (self.sum_x2_b - self.sum_x_b ** 2 / self.n_b) / max(self.n_b - 1, 1)
            var_b = max(var_b, 0.001)

            lambda_b_post = self.prior_lambda_b + self.n_b
            mu_b_post = (self.prior_lambda_b * self.prior_mu_b + self.sum_x_b) / lambda_b_post
            alpha_b_post = self.prior_alpha_b + self.n_b / 2
            beta_b_post = self.prior_beta_b + 0.5 * (self.sum_x2_b + self.prior_lambda_b * self.prior_mu_b ** 2 -
                                                      lambda_b_post * mu_b_post ** 2)
            beta_b_post = max(beta_b_post, 0.001)

            t_std_b = np.sqrt(beta_b_post * (1 + 1 / lambda_b_post) / alpha_b_post)
            ci_b = (mu_b_post - 1.96 * t_std_b, mu_b_post + 1.96 * t_std_b)
        else:
            mean_b = self.prior_mu_b
            t_std_b = np.sqrt(self.prior_beta_b / self.prior_alpha_b)
            ci_b = (mean_b - 1.96 * t_std_b, mean_b + 1.96 * t_std_b)
            lambda_b_post = self.prior_lambda_b
            alpha_b_post = self.prior_alpha_b
            beta_b_post = self.prior_beta_b

        return {
            "mean": mean_a,
            "std": t_std_a if self.n_a > 0 else np.sqrt(self.prior_beta_a / self.prior_alpha_a),
            "ci": ci_a,
            "lambda": lambda_a_post,
            "alpha": alpha_a_post,
            "beta": beta_a_post,
        }, {
            "mean": mean_b,
            "std": t_std_b if self.n_b > 0 else np.sqrt(self.prior_beta_b / self.prior_alpha_b),
            "ci": ci_b,
            "lambda": lambda_b_post,
            "alpha": alpha_b_post,
            "beta": beta_b_post,
        }

    def _get_count_posterior(self) -> Tuple[Dict, Dict]:
        """Compute Gamma posterior for count outcomes."""
        # Posterior = Gamma(α + sum(counts), β + n)
        alpha_a_post = self.prior_alpha_a + self.sum_counts_a
        beta_a_post = self.prior_beta_a + self.n_a

        alpha_b_post = self.prior_alpha_b + self.sum_counts_b
        beta_b_post = self.prior_beta_b + self.n_b

        # Posterior mean and std
        mean_a = alpha_a_post / beta_a_post
        mean_b = alpha_b_post / beta_b_post

        var_a = alpha_a_post / (beta_a_post ** 2)
        var_b = alpha_b_post / (beta_b_post ** 2)

        std_a = np.sqrt(var_a)
        std_b = np.sqrt(var_b)

        # Credible intervals
        ci_a = (stats.gamma.ppf(0.025, alpha_a_post, scale=1/beta_a_post),
                stats.gamma.ppf(0.975, alpha_a_post, scale=1/beta_a_post))
        ci_b = (stats.gamma.ppf(0.025, alpha_b_post, scale=1/beta_b_post),
                stats.gamma.ppf(0.975, alpha_b_post, scale=1/beta_b_post))

        return {
            "mean": mean_a,
            "std": std_a,
            "ci": ci_a,
            "alpha": alpha_a_post,
            "beta": beta_a_post,
        }, {
            "mean": mean_b,
            "std": std_b,
            "ci": ci_b,
            "alpha": alpha_b_post,
            "beta": beta_b_post,
        }

    def _monte_carlo_prob_b_better(self, posterior_a: Dict, posterior_b: Dict, samples: int = 10000) -> float:
        """
        Estimate P(B > A) using Monte Carlo sampling from posteriors.
        """
        if self.outcome_type == "binary":
            # Clamp Beta parameters to valid range (must be > 0)
            alpha_a = max(posterior_a["alpha"], 0.5)
            beta_a = max(posterior_a["beta"], 0.5)
            alpha_b = max(posterior_b["alpha"], 0.5)
            beta_b = max(posterior_b["beta"], 0.5)

            samples_a = np.random.beta(alpha_a, beta_a, samples)
            samples_b = np.random.beta(alpha_b, beta_b, samples)
        elif self.outcome_type == "continuous":
            # Clamp std to prevent zero/negative variance
            std_a = max(posterior_a["std"], 0.001)
            std_b = max(posterior_b["std"], 0.001)
            samples_a = np.random.normal(posterior_a["mean"], std_a, samples)
            samples_b = np.random.normal(posterior_b["mean"], std_b, samples)
        elif self.outcome_type == "count":
            # Clamp Gamma parameters to valid range
            alpha_a = max(posterior_a["alpha"], 0.5)
            beta_a = max(posterior_a["beta"], 0.001)
            alpha_b = max(posterior_b["alpha"], 0.5)
            beta_b = max(posterior_b["beta"], 0.001)

            samples_a = np.random.gamma(alpha_a, 1/beta_a, samples)
            samples_b = np.random.gamma(alpha_b, 1/beta_b, samples)

        # Proportion of samples where B > A
        prob_b_better = np.mean(samples_b > samples_a)
        return prob_b_better

    def update_and_check(self, day: int) -> bool:
        """
        Update posterior and check if decision threshold reached.
        Returns True if winner declared.
        """
        if self.outcome_type == "binary":
            posterior_a, posterior_b = self._get_binary_posterior()
        elif self.outcome_type == "continuous":
            posterior_a, posterior_b = self._get_continuous_posterior()
        elif self.outcome_type == "count":
            posterior_a, posterior_b = self._get_count_posterior()

        prob_b_better = self._monte_carlo_prob_b_better(posterior_a, posterior_b)

        # Store posterior
        p = BayesianPosterior(
            day=day,
            variant_a_mean=posterior_a["mean"],
            variant_b_mean=posterior_b["mean"],
            variant_a_std=posterior_a["std"],
            variant_b_std=posterior_b["std"],
            prob_b_better=prob_b_better,
            credible_interval_a=posterior_a["ci"],
            credible_interval_b=posterior_b["ci"],
        )
        self.posteriors.append(p)

        # Check decision threshold
        if prob_b_better >= self.confidence_threshold and self.decision_day is None:
            self.decision_day = day
            self.winner = "B"
            return True
        elif prob_b_better <= (1 - self.confidence_threshold) and self.decision_day is None:
            self.decision_day = day
            self.winner = "A"
            return True

        return False

    def get_latest_posterior(self) -> BayesianPosterior:
        """Get most recent posterior."""
        if not self.posteriors:
            raise ValueError("No posteriors computed yet. Call update_and_check first.")
        return self.posteriors[-1]

    def probability_b_better(self) -> float:
        """Get current P(B > A)."""
        return self.get_latest_posterior().prob_b_better

    def get_results_summary(self) -> Dict:
        """Get summary of test results."""
        if not self.posteriors:
            return {}

        latest = self.get_latest_posterior()
        return {
            "outcome_type": self.outcome_type,
            "decision_day": self.decision_day,
            "winner": self.winner,
            "final_prob_b_better": latest.prob_b_better,
            "variant_a_mean": latest.variant_a_mean,
            "variant_b_mean": latest.variant_b_mean,
            "variant_a_ci": latest.credible_interval_a,
            "variant_b_ci": latest.credible_interval_b,
            "total_posteriors_computed": len(self.posteriors),
        }
