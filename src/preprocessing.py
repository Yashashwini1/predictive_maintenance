from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import FAILURE_TYPE_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def get_feature_columns(df):
    """Return model input columns by excluding target columns."""
    return [col for col in df.columns if col not in [TARGET_COLUMN, FAILURE_TYPE_COLUMN]]


def split_data(df):
    """Create a stratified train/test split based on the binary target."""
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )


def build_preprocessor(X_train):
    """Create preprocessing pipeline for categorical and numeric columns."""
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return preprocessor
