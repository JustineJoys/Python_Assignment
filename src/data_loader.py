"""Data loaders implementing object-oriented inheritance for dataset ingestion."""

from abc import ABC, abstractmethod
import os
from typing import List
import pandas as pd

from src.exceptions import DataValidationError


class BaseLoader(ABC):
    """Abstract base class establishing the contract for all dataset loaders."""

    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise DataValidationError(f"File not found: '{file_path}'")
        self.file_path = file_path
        self.df = pd.DataFrame()

    def load_csv(self) -> pd.DataFrame:
        """Read CSV into a DataFrame with error handling."""
        try:
            return pd.read_csv(self.file_path)
        except Exception as err:
            raise DataValidationError(f"Could not read CSV '{self.file_path}': {err}") from err

    @abstractmethod
    def validate(self) -> None:
        """Validate column names and structure."""
        pass

    def check_data_integrity(self) -> None:
        """Verify that columns contain complete, non-null numeric values."""
        if self.df.isnull().values.any():
            raise DataValidationError(f"File '{self.file_path}' contains missing values.")
        for col in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise DataValidationError(f"Column '{col}' in '{self.file_path}' must be numeric.")

    def load(self) -> pd.DataFrame:
        """Read, validate schema, and check data integrity."""
        self.df = self.load_csv()
        self.validate()
        self.check_data_integrity()
        return self.df


class TrainLoader(BaseLoader):
    """Ingests and validates the training dataset."""

    REQUIRED_COLS: List[str] = ["x", "y1", "y2", "y3", "y4"]

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise DataValidationError(f"Training CSV '{self.file_path}' missing columns: {missing}")


class IdealLoader(BaseLoader):
    """Ingests and validates the 50 candidate ideal functions."""

    def validate(self) -> None:
        expected = ["x"] + [f"y{i}" for i in range(1, 51)]
        missing = [c for c in expected if c not in self.df.columns]
        if missing:
            raise DataValidationError(
                f"Ideal functions CSV '{self.file_path}' missing columns: {missing[:5]}..."
            )


class TestLoader(BaseLoader):
    """Ingests and validates the test dataset."""

    REQUIRED_COLS: List[str] = ["x", "y"]

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise DataValidationError(f"Test CSV '{self.file_path}' missing columns: {missing}")