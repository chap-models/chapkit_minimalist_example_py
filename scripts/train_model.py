#!/usr/bin/env python3
"""Train a scikit-learn LinearRegression on rainfall + mean_temperature.

Minimal port of the original `dhis2-chap/minimalist_example_uv` model into the
chapkit shell-py scaffold: same features (rainfall, mean_temperature), same
target (disease_cases), same joblib persistence. Replace with your own ML
logic when forking.

ShellModelRunner copies the project into an isolated workspace and substitutes
{data_file} into the train command (see main.py). config.yml is always
present in the workspace; we don't read it for this toy linear model.

Usage:
    python train_model.py --data data.csv [--geo geo.json]
"""

import argparse
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a disease prediction model")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional, unused)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} training samples", file=sys.stderr)

        features = df[["rainfall", "mean_temperature"]].fillna(0)
        target = df["disease_cases"].fillna(0)

        model = LinearRegression()
        model.fit(features, target)
        joblib.dump(model, "model.pickle")
        print("Model saved to model.pickle", file=sys.stderr)
        print("SUCCESS: Training completed")
    except Exception as exc:
        print(f"ERROR: Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
