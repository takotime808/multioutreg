import pandas as pd

class DataHolder:
    """
    DataHolder class to hold all relevant dataframes
    """
    def __init__(self, orders: pd.DataFrame, shifts: pd.DataFrame = None,
                 sales: pd.DataFrame = None, timeentries: pd.DataFrame = None):
        self.orders = orders
        self.shifts = shifts
        self.sales = sales
        self.timeentries = timeentries

class Analyzer:
    """
    Class to perform the following analyses on a restaurant:
    - How many patrons in a day over time?
    - What is the quality of each server in performance?
    - How many servers of what type on a shift in conjunction
      with number of patrons?
    - What is the most common server profile per day?
    - What is the count of servers per most popular time?
    - Restaurant is closed on Monday
    """

    def __init__(self, target: DataHolder):
        self.restaurant = target

    def classify_servers(self):
        """
        Try different bucket numbers:
        3 is a good separation, but Jordan says 2 is good
        Look into price per guest
        """
        ...

    def servers_per_day(self) -> pd.DataFrame:
        servers = (
            self.restaurant.orders.groupby([self.restaurant.orders.index.date])[
                "Server"
            ]
            .nunique()
            .reset_index(name="servers")
            .set_index("index")
        )
        return servers

    def customers_per_day(self) -> pd.DataFrame:
        customers = (
            self.restaurant.orders.groupby([self.restaurant.orders.index.date])[
                "# of Guests"
            ]
            .sum()
            .reset_index(name="customers")
            .set_index("index")
        )
        return customers

    def combine_relevant_data(self) -> pd.DataFrame:
        combined = self.servers_per_day().join(self.customers_per_day())
        return combined
