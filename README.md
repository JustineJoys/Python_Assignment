# Ideal Function Matching and Visualisation


## 1. Project Overview

A modular Python pipeline that fits empirical datasets to ideal candidate functions using least-squares optimization, assigns test points within analytical error margins, and visualizes the results alongside persisted SQLite records in an interactive Bokeh dashboard.

1. **Model Selection:** Identifies the optimal candidate function for each of the 4 training datasets by minimizing the sum of squared residuals across 50 candidate models.

2. **Threshold Calculation:** Extracts the peak deviation between empirical curves and their chosen ideal functions to set dynamic tolerance bounds.

3. **Streamed Classification:** Ingests test points sequentially, validating them against the calculated tolerance criteria and mapping them to the closest valid function.

4. **Database Integration:** Automates schema creation and stores all raw datasets, model mappings, and test classifications via SQLAlchemy and SQLite.

5. **Visualization Dashboard:** Produces a multi-panel Bokeh report featuring interactive tooltips, shaded error bands, and overlayed test-point mappings.

## 2. Project Architecture & Directory Structure

├── data/
│   ├── train.csv              # Empirical training datasets (x, y1..y4)
│   ├── ideal.csv              # Reference ideal functions (x, y1..y50)
│   └── test.csv               # Test coordinates (x, y)
├── output/
│   ├── functions_data.db      # SQLite relational database (generated)
│   └── dashboard.html         # Interactive Bokeh visualization (generated)
├── src/
│   ├── __init__.py
│   ├── analyzer.py            # OLS CurveMatcher and PointClassifier
│   ├── data_loader.py         # Abstract BaseLoader and CSV validators
│   ├── database.py            # SQLAlchemy ORM schemas and Database manager
│   ├── exceptions.py          # Custom domain exception hierarchy
│   └── visualizer.py          # Multi-panel Bokeh dashboard generator
├── tests/
│   ├── __init__.py
│   └── test_suite.py          # Unit tests covering ingestion, math, DB & bounds
├── main.py                    # Pipeline entry point
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation


## 3. Installation & Environment Setup

### 3.1 Prerequisites
* Python 3.10 or higher
* `pip` package manager

### 3.2 Virtual Environment Setup

Clone or extract the repository, open a terminal in the root directory, and execute:

```bash
# Create a local virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Dependencies (`requirements.txt`)
* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `sqlalchemy>=2.0.0`
* `bokeh>=3.0.0`


## 4. Running the Pipeline

Execute the primary script from the root directory:

```bash
python main.py
```


## 5. Running the Automated Unit Tests

The test suite validates data loading integrity, OLS curve fitting accuracy, boundary conditions, custom exception handling, and database operations.

Run the test suite using standard discovery:

```bash
python -m unittest discover -s tests -v
```

Or target the test module directly:

```bash
python -m unittest -v tests.test_suite
```

---

## 6. Mathematical Results Summary

* **Training Function $y_1$:** Matched with **$y_{13}$** (SSD = 34.0807, $\Delta_{\max} = 0.4992$, Threshold = 0.7060)
* **Training Function $y_2$:** Matched with **$y_{24}$** (SSD = 33.4518, $\Delta_{\max} = 0.4990$, Threshold = 0.7057)
* **Training Function $y_3$:** Matched with **$y_{36}$** (SSD = 35.5727, $\Delta_{\max} = 0.4989$, Threshold = 0.7056)
* **Training Function $y_4$:** Matched with **$y_{40}$** (SSD = 34.9989, $\Delta_{\max} = 0.4998$, Threshold = 0.7068)
* **Test Classification Result:** **34 out of 100 points** meet the $\Delta_{\max}\sqrt{2}$ acceptance criteria and are mapped to their respective ideal function; 66 points are categorized as unmapped ambient noise.

---

## 7. Defensive Programming Features

* **Data Integrity Checks:** Enforces strict numeric typing and verifies no null/NaN values exist before calculation begins.
* **Domain Proximity Guard:** Rejects coordinates outside the valid domain grid ($[-20.0, 20.0]$) rather than extrapolating or clamping arbitrarily.
* **Idempotent Database Execution:** Automatically cleans existing table state on re-run to prevent duplicate row accumulation.
* **Custom Exception Hierarchy:** Distinguishes between `DataValidationError`, `FittingError`, and `DatabaseError`, all subclassing `AppError`.