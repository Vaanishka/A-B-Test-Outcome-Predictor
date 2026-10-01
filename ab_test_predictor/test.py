from ab_test_predictor import run_analysis

results = run_analysis(
    data_dir="data/raw",
    output_dir="results",
    confidence_threshold=0.95,
    revenue_per_week=50000
)