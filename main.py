"""Runs the function matching and classification pipeline."""

import logging
import os
import sys
from typing import Any, List

from src.analyzer import CurveMatcher, PointClassifier
from src.database import Database, IdealFunction, TestMapping, TrainRecord
from src.data_loader import IdealLoader, TestLoader, TrainLoader
from src.exceptions import AppError
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run data ingestion, model fitting, database storage, and plotting."""
    try:
        os.makedirs("output", exist_ok=True)

        # Load CSVs
        logger.info("Loading CSV files...")
        train_df = TrainLoader("data/train.csv").load()
        ideal_df = IdealLoader("data/ideal.csv").load()
        test_df = TestLoader("data/test.csv").load()

    # Database setup
        logger.info("Setting up SQLite database...")
        db = Database("sqlite:///output/functions_data.db")
        db.reset_database()  # Replaces db.create_tables()

        # Save training data
        logger.info("Saving training data...")
        train_rows: List[Any] = [
            TrainRecord(x=r["x"], y1=r["y1"], y2=r["y2"], y3=r["y3"], y4=r["y4"])
            for _, r in train_df.iterrows()
        ]
        db.insert_records(train_rows)

        # Save candidate functions
        logger.info("Saving candidate ideal functions...")
        ideal_rows: List[Any] = []
        for _, r in ideal_df.iterrows():
            row_dict = {"x": r["x"]}
            for i in range(1, 51):
                row_dict[f"y{i}"] = r[f"y{i}"]
            ideal_rows.append(IdealFunction(**row_dict))
        db.insert_records(ideal_rows)

        # Run curve fitting
        logger.info("Fitting training curves to ideal models...")
        matcher = CurveMatcher(train_df, ideal_df)
        fits = matcher.find_best_fits()

        for t_col, fit in fits.items():
            logger.info(
                f"Selected {fit.ideal_col} for {t_col} | "
                f"SSD: {fit.ssd:.4f} | Max Dev: {fit.max_dev:.4f} | Threshold: {fit.threshold:.4f}"
            )

        # Classify test points
        logger.info("Classifying test points...")
        classifier = PointClassifier(ideal_df, fits)
        test_matches = list(classifier.classify_stream(test_df))

        # Save mappings
        test_rows: List[Any] = [
            TestMapping(
                x=m.x,
                y=m.y,
                delta_y=m.delta_y,
                chosen_ideal_function=m.ideal_col,
            )
            for m in test_matches
        ]
        db.insert_records(test_rows)
        mapped_count = sum(1 for m in test_matches if m.ideal_col is not None)
        logger.info(f"Done: {mapped_count}/{len(test_matches)} test points mapped.")

        # Generate dashboard
        logger.info("Generating Bokeh dashboard...")
        viz = Visualizer("output/dashboard.html")
        viz.plot(train_df, ideal_df, fits, test_matches)
        logger.info("All tasks completed successfully.")

    except AppError as err:
        logger.error(f"Application error: {err}")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Unexpected error: {err}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()