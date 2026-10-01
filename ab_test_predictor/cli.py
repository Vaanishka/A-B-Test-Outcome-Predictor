"""
Command-line interface for Bayesian A/B Test Predictor
"""

import argparse
import sys
from pathlib import Path
from scripts.core import run_analysis, get_summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bayesian A/B Test Predictor - Run faster A/B tests using Bayesian inference"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing CSV datasets"
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results and plots (default: results)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for Bayesian decision (default: 0.95)"
    )

    parser.add_argument(
        "--revenue-per-week",
        type=float,
        default=50000,
        help="Revenue per week saved (for ROI calculation, default: 50000)"
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    print("\n" + "="*80)
    print("BAYESIAN A/B TEST PREDICTOR")
    print("="*80 + "\n")

    results_df = run_analysis(
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
        revenue_per_week=args.revenue_per_week,
    )

    # Print summary
    summary = get_summary(results_df)
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

    print(f"Results Summary:")
    print(f"  Average Speed-up: {summary['average_speedup_percent']:.2f}%")
    print(f"  Average Weeks Saved: {summary['average_weeks_saved']:.2f} weeks")
    print(f"  Total Weeks Saved: {summary['total_weeks_saved']:.2f} weeks")
    print(f"  Accuracy: {summary['accuracy'] * 100:.1f}%")
    print(f"  Tests where Bayesian faster: {summary['tests_bayesian_faster']}/{summary['total_tests']}")
    print(f"\nOutput saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
