# # Copyright (c) 2025 takotime808

# import os
# import importlib.util
# import pytest
# import pandas as pd

# # Dynamically load the Streamlit page module
# FILE_PATH = os.path.abspath("multioutreg/gui/pages/05_Sampling_Datasets_Examples.py")
# spec = importlib.util.spec_from_file_location("sampling_datasets", FILE_PATH)
# sampling_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(sampling_module)


# def test_generate_lhs_sampling_dataset():
#     name, df = sampling_module.generate_lhs_sampling_dataset()
#     assert name == "LHS Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"x1", "x2", "response"}
#     assert len(df) == 100


# def test_generate_grid_sampling_dataset():
#     name, df = sampling_module.generate_grid_sampling_dataset(num_points_per_axis=10)
#     assert name == "Grid Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"x1", "x2", "x3", "x4", "x5", "response"}
#     assert len(df) == 100  # 10x10 grid


# def test_generate_random_sampling_dataset():
#     name, df = sampling_module.generate_random_sampling_dataset(n_points=100)
#     assert name == "Randomly Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"x1", "x2", "response"}
#     assert len(df) == 100


# def test_generate_sobol_sampling_dataset():
#     name, df = sampling_module.generate_sobol_sampling_dataset(n_points=128)
#     assert name == "Sobol Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"x1", "x2", "response"}
#     assert len(df) == 128


# def test_generate_uncertain_sampling_dataset():
#     name, df = sampling_module.generate_uncertain_sampling_dataset()
#     assert name == "'Uncertain' Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert set(df.columns) == {"x1", "x2", "response"}
#     assert len(df) == 6


# def test_fix_generate_grid_sampling_dataset():
#     name, df = sampling_module.fix_generate_grid_sampling_dataset()
#     assert name == "Grid Sampled Dataset"
#     assert isinstance(df, pd.DataFrame)
#     assert {"alpha_deg", "beta_deg", "Re", "Cl"}.issubset(df.columns)
#     assert len(df) > 0

