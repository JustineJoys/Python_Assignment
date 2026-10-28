"""Unit tests for curve fitting, point classification, data loading, and database storage."""

import math
import os
import unittest
import numpy as np
import pandas as pd

from src.analyzer import CurveMatcher, ModelFit, PointClassifier
from src.database import Database, TestMapping
from src.data_loader import TrainLoader
from src.exceptions import DataValidationError, FittingError


class TestCurveMatcher(unittest.TestCase):
    """Verifies least-squares model selection and classification thresholds."""

    def setUp(self) -> None:
        self.x_vals = np.array([0.0, 1.0, 2.0, 3.0])
        self.train_df = pd.DataFrame({
            "x": self.x_vals,
            "y1": [1.0, 2.0, 3.0, 4.0],
        })
        self.ideal_df = pd.DataFrame({
            "x": self.x_vals,
            "y1": [0.9, 2.2, 2.9, 4.1],
            "y2": [-1.0, 0.0, 1.0, 2.0],
        })

    def test_least_squares_fit(self) -> None:
        matcher = CurveMatcher(self.train_df, self.ideal_df)
        fits = matcher.find_best_fits()

        self.assertIn("y1", fits)
        fit = fits["y1"]
        self.assertEqual(fit.ideal_col, "y1")
        self.assertAlmostEqual(fit.ssd, 0.07, places=4)
        self.assertAlmostEqual(fit.max_dev, 0.2, places=4)
        self.assertAlmostEqual(fit.threshold, 0.2 * math.sqrt(2.0), places=4)

    def test_valid_point_mapping(self) -> None:
        threshold = 0.2 * math.sqrt(2.0)
        fit = ModelFit(
            train_col="y1",
            ideal_col="y1",
            ssd=0.07,
            max_dev=0.2,
            threshold=threshold,
        )
        classifier = PointClassifier(self.ideal_df, {"y1": fit})

        # At x=1.0, ideal y1 is 2.2. A test y of 2.4 gives delta = 0.20 <= 0.28284
        result = classifier.classify_point(1.0, 2.4)
        self.assertEqual(result.ideal_col, "y1")
        self.assertAlmostEqual(result.delta_y, 0.2, places=4)

    def test_outlier_rejection(self) -> None:
        threshold = 0.2 * math.sqrt(2.0)
        fit = ModelFit(
            train_col="y1",
            ideal_col="y1",
            ssd=0.07,
            max_dev=0.2,
            threshold=threshold,
        )
        classifier = PointClassifier(self.ideal_df, {"y1": fit})

        # At x=1.0, ideal y1 is 2.2. A test y of 2.6 gives delta = 0.40 > 0.28284
        result = classifier.classify_point(1.0, 2.6)
        self.assertIsNone(result.ideal_col)
        self.assertIsNone(result.delta_y)

    def test_boundary_threshold(self) -> None:
        threshold = 0.2 * math.sqrt(2.0)
        fit = ModelFit(
            train_col="y1",
            ideal_col="y1",
            ssd=0.07,
            max_dev=0.2,
            threshold=threshold,
        )
        classifier = PointClassifier(self.ideal_df, {"y1": fit})

        # Test point right at the threshold boundary
        y_boundary = 2.2 + threshold
        result = classifier.classify_point(1.0, y_boundary)
        self.assertEqual(result.ideal_col, "y1")
        self.assertAlmostEqual(result.delta_y, threshold, places=5)

    def test_mismatched_x_coordinates(self) -> None:
        invalid_ideal_df = self.ideal_df.copy()
        invalid_ideal_df["x"] = [10.0, 11.0, 12.0, 13.0]

        with self.assertRaises(FittingError):
            CurveMatcher(self.train_df, invalid_ideal_df)


class TestLoaders(unittest.TestCase):
    """Tests CSV loading, file missing errors, and column validation."""

    def test_file_not_found(self) -> None:
        with self.assertRaises(DataValidationError):
            TrainLoader("data/non_existent_file.csv")

    def test_missing_columns(self) -> None:
        temp_csv = "data/temp_invalid.csv"
        invalid_df = pd.DataFrame({"x": [1, 2], "y1": [10, 20]})
        invalid_df.to_csv(temp_csv, index=False)

        try:
            loader = TrainLoader(temp_csv)
            with self.assertRaises(DataValidationError):
                loader.load()
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)


class TestDatabase(unittest.TestCase):
    """Tests SQLite table creation and record insertion in an in-memory database."""

    def setUp(self) -> None:
        self.db = Database(db_path="sqlite:///:memory:")
        self.db.create_tables()

    def test_insert_and_fetch(self) -> None:
        record = TestMapping(x=5.5, y=12.3, delta_y=0.15, chosen_ideal_function="y24")
        self.db.insert_records([record])

        session = self.db.SessionFactory()
        persisted = session.query(TestMapping).first()
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.x, 5.5)
        self.assertEqual(persisted.chosen_ideal_function, "y24")
        session.close()


if __name__ == "__main__":
    unittest.main()