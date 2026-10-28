"""Custom domain exceptions for the data fitting and classification pipeline."""


class AppError(Exception):
    """Base exception for application-level errors."""

    pass


class DataValidationError(AppError):
    """Raised when an ingested CSV file fails schema or integrity checks."""

    pass


class FittingError(AppError):
    """Raised when least-squares model selection or comparison fails."""

    pass


class DatabaseError(AppError):
    """Raised when a SQLite or SQLAlchemy persistence operation fails."""

    pass