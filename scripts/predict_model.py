#!/usr/bin/env python3
"""Generate predictions using a trained LinearRegression model.

Minimal port of the original `dhis2-chap/minimalist_example_uv` predict step
into the chapkit shell-py scaffold: load the joblib-pickled model, predict
disease_cases from future rainfall + mean_temperature, write a single-sample
column (`sample_0`).

ShellModelRunner substitutes {historic_file}, {future_file}, {output_file}
into the predict command (see main.py). The historic CSV isn't used by this
toy linear model - kept in the signature for parity with chap-core's expected
predict shape.

Usage:
    python predict_model.py --historic historic.csv --future future.csv --output predictions.csv [--geo geo.json]
"""

import argparse
import sys

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate disease predictions")
    parser.add_argument("--historic", required=True, help="Path to historic data CSV (unused)")
    parser.add_argument("--future", required=True, help="Path to future climate data CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional, unused)")
    args = parser.parse_args()

    try:
        model = joblib.load("model.pickle")
        future = pd.read_csv(args.future)
        print(f"Loaded {len(future)} prediction rows", file=sys.stderr)

        features = future[["rainfall", "mean_temperature"]].fillna(0)
        future["sample_0"] = model.predict(features)
        future.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}", file=sys.stderr)
        print("SUCCESS: Prediction completed")
    except Exception as exc:
        print(f"ERROR: Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
