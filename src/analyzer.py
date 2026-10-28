"""Curve fitting logic and test point classification."""

from dataclasses import dataclass
import math
from typing import Dict, Iterator, List, Optional
import numpy as np
import pandas as pd

from src.exceptions import FittingError


@dataclass(frozen=True)
class ModelFit:
    """Holds the best fit result for a training curve."""

    train_col: str
    ideal_col: str
    ssd: float
    max_dev: float
    threshold: float


@dataclass(frozen=True)
class PointMatch:
    """Stores the mapping result for an evaluated test point."""

    x: float
    y: float
    delta_y: Optional[float]
    ideal_col: Optional[str]


class CurveMatcher:
    """Matches training curves to candidate functions via least-squares."""

    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        if train_df.empty or ideal_df.empty:
            raise FittingError("Input DataFrames cannot be empty.")
        if not np.allclose(train_df["x"].values, ideal_df["x"].values):
            raise FittingError("Coordinate grids for training and ideal data do not match.")

        self.train_df = train_df
        self.ideal_df = ideal_df

    def find_best_fits(self) -> Dict[str, ModelFit]:
        """Find the best-fitting ideal function for each training column."""
        train_cols = [c for c in self.train_df.columns if c.startswith("y")]
        ideal_cols = [c for c in self.ideal_df.columns if c.startswith("y")]

        fits: Dict[str, ModelFit] = {}
        sqrt2 = math.sqrt(2.0)

        for t_col in train_cols:
            y_train = self.train_df[t_col].to_numpy()
            best_col = None
            min_ssd = float("inf")
            max_dev = 0.0

            for i_col in ideal_cols:
                y_ideal = self.ideal_df[i_col].to_numpy()
                residuals = y_train - y_ideal
                ssd = float(np.sum(residuals**2))

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_col = i_col
                    max_dev = float(np.max(np.abs(residuals)))

            if best_col is None:
                raise FittingError(f"Could not find an optimal model for training series '{t_col}'.")

            fits[t_col] = ModelFit(
                train_col=t_col,
                ideal_col=best_col,
                ssd=min_ssd,
                max_dev=max_dev,
                threshold=max_dev * sqrt2,
            )

        return fits


class PointClassifier:
    """Checks test coordinates against the chosen ideal models."""

    def __init__(self, ideal_df: pd.DataFrame, fits: Dict[str, ModelFit]):
        # Verify candidate grid has unique x coordinates
        if ideal_df["x"].duplicated().any():
            raise FittingError("Candidate ideal dataset contains duplicate x coordinates.")
        self.ideal_indexed = ideal_df.set_index("x")
        self.fits = fits

    def classify_point(self, x: float, y: float, grid_tolerance: float = 1e-4) -> PointMatch:
        """Match a test point to the closest ideal model if within domain and tolerance."""
        available_x = self.ideal_indexed.index
        min_dist = float(np.min(np.abs(available_x - x)))

        # Reject coordinates outside the observed grid domain
        if min_dist > grid_tolerance:
            return PointMatch(x=x, y=y, delta_y=None, ideal_col=None)

        matched_x = available_x[np.argmin(np.abs(available_x - x))]
        best_model: Optional[str] = None
        min_delta = float("inf")

        for fit in self.fits.values():
            expected_y = float(self.ideal_indexed.loc[matched_x, fit.ideal_col])
            delta = abs(y - expected_y)

            if delta <= fit.threshold and delta < min_delta:
                min_delta = delta
                best_model = fit.ideal_col

        if best_model is not None:
            return PointMatch(x=x, y=y, delta_y=min_delta, ideal_col=best_model)
        return PointMatch(x=x, y=y, delta_y=None, ideal_col=None)

    def classify_stream(self, test_df: pd.DataFrame) -> Iterator[PointMatch]:
        """Process test dataset rows one by one."""
        for _, row in test_df.iterrows():
            yield self.classify_point(float(row["x"]), float(row["y"]))