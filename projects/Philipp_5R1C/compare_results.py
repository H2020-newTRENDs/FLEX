import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from basics.db import DB
from config import config
from models.operation.enums import OperationTable


class CompareModels:
    def __init__(self):
        self.input_path = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data")



    def read_daniel_heat_demand(self):
        filename = "heating_demand_daniel.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";").drop(columns=["Unnamed: 0"])
        return demand_df

    def read_opt_heat_demand(self):
        demand = DB(config).read_dataframe(table_name=OperationTable.ResultOptHour.value,
                                           column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"])
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["1", "2", "3", "4", "5"]
        return df

    def read_electricity_price(self):

        pass


    def compare_model_results(self):

        pass

    def calculate_costs(self):

        pass


    def main(self):
        heat_demand_daniel = self.read_daniel_heat_demand()
        heat_demand_opt = self.read_opt_heat_demand()

        costs_daniel = self.calculate_costs()
        costs_opt = self.calculate_costs()

        self.compare_model_results()

if __name__ == "__main__":
    CompareModels().main()


