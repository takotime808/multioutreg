# Copyright (c) 2025 takotime808
from pathlib import Path
import pandas as pd
from typing import Optional

def find_file_by_prefix(directory: str, prefix: str, extension: Optional[str]=None) -> Optional[str]:
    """
    Finds the first file in the directory that starts with the given prefix.
    Optionally filter by file extension.

    Args:
        directory (str): The path to search.
        prefix (str): The prefix to match.
        extension (Optional[str]): Only match files with this extension (e.g. ".csv").

    Returns:
        Optional[str]: The found file path, or None.
    """
    for file in Path(directory).iterdir():
        if file.is_file() and file.name.startswith(prefix):
            if extension:
                if file.suffix == extension:
                    return str(file)
            else:
                return str(file)
    print(f"No file found starting with {prefix} in {directory}")
    return None

def read_orders(directory: str) -> pd.DataFrame:
    """
    Reads the most recent OrderDetails file from the directory.

    Args:
        directory (str): Folder containing data.

    Returns:
        pd.DataFrame: Orders data.
    """
    filename = find_file_by_prefix(directory, "OrderDetails_", ".csv")
    if filename:
        orders = (
            pd.read_csv(filename, parse_dates=["Opened"], date_format="%m/%d/%y %I:%M %p")
            .set_index("Opened")
            .sort_index()
        )
        return orders
    else:
        return pd.DataFrame()  # Or raise error as desired

def read_shifts(directory: str) -> pd.DataFrame:
    """
    Reads the most recent Shifts_Closed file from the directory.

    Args:
        directory (str): Folder containing data.

    Returns:
        pd.DataFrame: Shifts data.
    """
    filename = find_file_by_prefix(directory, "Shifts_Closed_", ".csv")
    if filename:
        shifts = pd.read_csv(
            filename,
            parse_dates=["In Date", "Shift Closed Date", "Out Date"],
            date_format="%m/%d/%y %I:%M %p",
        )
        shifts.drop("Unnamed: 0", inplace=True, axis=1)
        return shifts
    else:
        return pd.DataFrame()

def read_sales(directory: str) -> pd.DataFrame:
    """
    Reads the most recent sp_hourly_sales file from the directory.

    Args:
        directory (str): Folder containing data.

    Returns:
        pd.DataFrame: Sales data.
    """
    filename = find_file_by_prefix(directory, "sp_hourly_sales_", ".xlsx")
    if filename:
        sales = pd.read_excel(
            filename, sheet_name="Past6Months", parse_dates=["Start Time"]
        ).set_index("Start Time")
        return sales
    else:
        return pd.DataFrame()

def read_time_entries(directory: str) -> pd.DataFrame:
    """
    Reads the most recent TimeEntries file from the directory.

    Args:
        directory (str): Folder containing data.

    Returns:
        pd.DataFrame: Time Entries data.
    """
    filename = find_file_by_prefix(directory, "TimeEntries_", ".csv")
    if filename:
        timeentries = pd.read_csv(
            filename,
            parse_dates=["In Date", "Out Date"],
            date_format="%m/%d/%y %I:%M %p",
        )
        return timeentries
    else:
        return pd.DataFrame()

