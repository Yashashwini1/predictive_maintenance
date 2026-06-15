"""Compatibility wrapper.

The project now separates data loading and preprocessing into two files:
- data_loader.py
- preprocessing.py
"""

from data_loader import load_clean_data as load_data
from preprocessing import build_preprocessor, get_feature_columns, split_data
