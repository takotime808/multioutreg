# # Copyright (c) 2025 takotime808

# import os
# import importlib.util
# import pytest
# import pandas as pd

# # Dynamically load the streamlit module
# FILE_PATH = os.path.abspath("multioutreg/gui/pages/07_Other_Dataset_Examples.py")
# spec = importlib.util.spec_from_file_location("other_datasets", FILE_PATH)
# other_datasets = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(other_datasets)


# def test_generate_dataset_linear():
#     name, df = other_datasets.generate_dataset_linear()
#     assert isinstance(name, str)
#     assert isinstance(df, pd.DataFrame)
#     assert df.shape[1] == 5
#     assert {"x1", "x2", "x3", "y1", "y2"}.issubset(df.columns)


# def test_generate_dataset_nonlinear():
#     name, df = other_datasets.generate_dataset_nonlinear()
#     assert isinstance(name, str)
#     assert isinstance(df, pd.DataFrame)
#     assert df.shape[1] == 6
#     assert {"f1", "f2", "f3", "f4", "output1", "output2"}.issubset(df.columns)


# def test_generate_dataset_multifidelity():
#     name, df = other_datasets.generate_dataset_multifidelity()
#     assert isinstance(name, str)
#     assert isinstance(df, pd.DataFrame)
#     assert df["fidelity_level"].nunique() == 3
#     assert {"x1", "x2", "fidelity_level", "y1", "y2"}.issubset(df.columns)


# def test_generate_dataset_doe_with_class_imbalance():
#     name, df = other_datasets.generate_dataset_doe_with_class_imbalance()
#     assert isinstance(name, str)
#     assert isinstance(df, pd.DataFrame)
#     assert "timestamp" in df.columns
#     assert "class_label" in df.columns
#     assert df["class_label"].isin([0, 1]).all()


# def test_generate_dataset_doe_with_drift():
#     name, df = other_datasets.generate_dataset_doe_with_drift()
#     assert isinstance(name, str)
#     assert isinstance(df, pd.DataFrame)
#     assert "timestamp" in df.columns
#     assert "class_label" in df.columns
#     assert df["class_label"].isin([0, 1]).all()
