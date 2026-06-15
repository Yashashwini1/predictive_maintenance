from pathlib import Path

DATA_PATH = "data/predictive_maintenance.csv"

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")

ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
FAILURE_THRESHOLD = 0.50

EXPERIMENT_NAME = "Predictive_Maintenance_Hierarchical_Pipeline"
RUN_NAME = "RF_XGBoost_threshold_0_50"

ID_COLUMNS = ["UDI", "Product ID"]
TARGET_COLUMN = "Target"
FAILURE_TYPE_COLUMN = "Failure Type"
RANDOM_FAILURE_LABEL = "Random Failures"
NO_FAILURE_LABEL = "No Failure"
