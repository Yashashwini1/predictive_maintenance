import pandas as pd

from config import DATA_PATH, FAILURE_TYPE_COLUMN, ID_COLUMNS, TARGET_COLUMN


def load_data(data_path=DATA_PATH):
    """Load raw predictive maintenance data from CSV."""
    df = pd.read_csv(data_path)
    return df


def validate_data(df):
    """Check that required target columns exist."""
    required_cols = {TARGET_COLUMN, FAILURE_TYPE_COLUMN}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def remove_identifier_columns(df):
    """Remove ID columns because they are identifiers, not predictive features."""
    return df.drop(columns=[col for col in ID_COLUMNS if col in df.columns])


def load_clean_data(data_path=DATA_PATH):
    """Load data, remove ID columns, and validate required columns."""
    df = load_data(data_path)
    df = remove_identifier_columns(df)
    df = validate_data(df)
    return df
