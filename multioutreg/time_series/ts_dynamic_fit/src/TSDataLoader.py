# Copyright (c) 2025 takotime808
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from multioutreg.time_series.ts_dynamic_fit.data_handling.ingest import (
    read_orders,
    read_sales,
    read_shifts,
    read_time_entries,
)
from multioutreg.time_series.ts_dynamic_fit.data_handling.Analyzer import Analyzer, DataHolder

class TSDataLoader:
    """
    Time-series data loader for OrderDetails, Shifts_Closed,
    sp_hourly_sales, and TimeEntries.
    """

    def __init__(self, data_dir: str = 'examples/example_data_ts/data'):
        """
        Initializes loader with target data directory.

        Args:
            data_dir (str): Path to folder containing the data files.
        """
        self.data_path: Path = Path(data_dir)

    def read_data(self, data_path: str) -> pd.DataFrame:
        """
        Generic CSV loader for single file.

        Args:
            data_path (str): Path to CSV file.

        Returns:
            pd.DataFrame: Data loaded from the file.
        """
        data = pd.read_csv(data_path)
        return data

    def get(self) -> DataFrame:
        """
        Loads time-series datasets based on file type prefix.

        Returns:
            DataFrame: Combined time-series analysis data.
        """
        # Each ingest function takes directory,
        # not specific file name.
        orders: pd.DataFrame = read_orders(str(self.data_path))
        shifts: pd.DataFrame = read_shifts(str(self.data_path))
        sales: pd.DataFrame = read_sales(str(self.data_path))
        timeentries: pd.DataFrame = read_time_entries(str(self.data_path))

        info = DataHolder(orders, shifts, sales, timeentries)
        analysis = Analyzer(info)
        tsdata: DataFrame = analysis.combine_relevant_data()
        tsdata = tsdata.reset_index() 

        return tsdata

