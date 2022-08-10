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
        # rename the columns to 1, 2, 3, 4, 5
        demand_df.columns = ["1", "2", "3", "4", "5"]
        return demand_df

    def read_heat_demand(self, table_name: str):
        demand = DB(config).read_dataframe(table_name=table_name,
                                           column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"])
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["1", "2", "3", "4", "5"]
        return df

    def read_electricity_price(self) -> np.array:
        price = DB(config).read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=["electricity_1"]). to_numpy()
        return price


    def calculate_costs(self):
        price_profile = self.read_electricity_price()


        pass


    def compare_heat_demand(self, heat_demand_daniel, heat_demand_ref):
        # compare hourly demand
        column_number = len(list(heat_demand_daniel.columns))
        x_axis = np.arange(8760)

        fig = make_subplots(
            rows=column_number,
            cols=1,
            subplot_titles=sorted(list(heat_demand_daniel.columns)),
            shared_xaxes=True,
        )

        for i, column_name in enumerate(sorted(list(heat_demand_daniel.columns))):
            fig.add_trace(
                go.Scatter(x=x_axis, y=heat_demand_daniel[column_name],
                           name="IDA ICE",
                           line=dict(color="blue")),

                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_axis, y=heat_demand_ref[column_name],
                           name="5R1C",
                           line=dict(color="red")),
                row=i + 1,
                col=1,
            )
        fig.show()

        # compare total demand
        total_daniel = heat_demand_daniel.sum()
        total_ref = heat_demand_ref.sum()
        total_df = pd.concat([total_daniel, total_ref], axis=1) / 1_000  # kW
        total_df.columns = ["IDA ICE", "5R1C"]
        fig2 = px.bar(
            data_frame=total_df,
            barmode="group",
        )
        fig2.update_layout(title_text="total heating demand")
        fig2.show()


    def compare_model_results(self, heat_demand_daniel, heat_demand_ref):
        # compare the heat demand:
        self.compare_heat_demand(heat_demand_daniel, heat_demand_ref)

        pass


    def main(self):
        heat_demand_daniel = self.read_daniel_heat_demand()
        # heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value)
        heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value)

        costs_daniel = self.calculate_costs()
        costs_opt = self.calculate_costs()

        self.compare_model_results(heat_demand_daniel, heat_demand_ref)

if __name__ == "__main__":
    CompareModels().main()


