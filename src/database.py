"""Database models and SQLite connection handler using SQLAlchemy ORM."""

from typing import Any, Dict, List
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from src.exceptions import DatabaseError

Base = declarative_base()


class TrainRecord(Base):
    """ORM representation of the unified training dataset (Table 1)."""

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False, index=True)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)


# Dynamic schema definition for the 50 candidate ideal functions (Table 2)
ideal_table_cols: Dict[str, Any] = {
    "__tablename__": "ideal_functions",
    "id": Column(Integer, primary_key=True, autoincrement=True),
    "x": Column(Float, nullable=False, index=True),
}
for i in range(1, 51):
    ideal_table_cols[f"y{i}"] = Column(Float, nullable=False)

IdealFunction = type("IdealFunction", (Base,), ideal_table_cols)


class TestMapping(Base):
    """ORM representation of mapped test coordinates and deviations (Table 3)."""

    __tablename__ = "test_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    delta_y = Column(Float, nullable=True)
    chosen_ideal_function = Column(String(10), nullable=True)


class Database:
    """Handles the SQLite connection and table operations."""

    def __init__(self, db_path: str = "sqlite:///output/functions_data.db"):
        self.db_path = db_path
        try:
            self.engine = create_engine(self.db_path, echo=False)
            self.SessionFactory = sessionmaker(bind=self.engine)
        except SQLAlchemyError as err:
            raise DatabaseError(f"Failed to initialize database engine: {err}") from err

    def create_tables(self) -> None:
        """Create all tables defined in Base metadata."""
        try:
            Base.metadata.create_all(self.engine)
        except SQLAlchemyError as err:
            raise DatabaseError(f"Failed to create database tables: {err}") from err

    def reset_database(self) -> None:
        """Recreate all tables to ensure an idempotent database state."""
        try:
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
        except SQLAlchemyError as err:
            raise DatabaseError(f"Failed to reset database: {err}") from err

    def insert_records(self, records: List[Any]) -> None:
        """Persist a collection of ORM records in a single transactional commit."""
        if not records:
            return

        session = self.SessionFactory()
        try:
            session.bulk_save_objects(records)
            session.commit()
        except SQLAlchemyError as err:
            session.rollback()
            raise DatabaseError(f"Batch insertion failed: {err}") from err
        finally:
            session.close()