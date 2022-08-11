import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable


class CompareModels:
    def __init__(self, project_name: str):
        self.input_path = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data")
        self.project_name = project_name

    def read_daniel_heat_demand(self):
        filename = "heating_demand_daniel.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";").drop(columns=["Unnamed: 0"])
        # rename the columns to 1, 2, 3, 4, 5
        demand_df.columns = ["1", "2", "3", "4", "5"]
        return demand_df

    def read_daniel_heat_demand_optimized(self):
        pass

    def read_heat_demand(self, table_name: str):
        demand = DB(get_config(self.project_name)).read_dataframe(table_name=table_name,
                                           column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"])
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["1", "2", "3", "4", "5"]
        return df

    def read_indoor_temp(self, table_name: str):
        temp = DB(get_config(self.project_name)).read_dataframe(table_name=table_name,
                                           column_names=["ID_Scenario", "T_Room", "Hour"])
        df = temp.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["1", "2", "3", "4", "5"]
        return df

    def read_costs(self, table_name: str):
        cost = DB(get_config(self.project_name)).read_dataframe(table_name=table_name,
                                                                column_names=["TotalCost"])
        cost.loc[:, "index"] = np.arange(1, len(cost)+1)
        df = cost.pivot_table(columns="index")
        df.columns = ["1", "2", "3", "4", "5"]
        return df

    def read_electricity_price(self) -> np.array:
        price = DB(get_config(self.project_name)).read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=["electricity_1"]). to_numpy()
        return price


    def calculate_costs(self):
        price_profile = self.read_electricity_price()


        pass


    def compare_hourly_profile(self, demand_1, demand_2, name_demand_1: str, name_demand_2: str):
        # compare hourly demand
        column_number = len(list(demand_1.columns))
        x_axis = np.arange(8760)

        fig = make_subplots(
            rows=column_number,
            cols=1,
            subplot_titles=sorted(list(demand_1.columns)),
            shared_xaxes=True,
        )

        for i, column_name in enumerate(sorted(list(demand_1.columns))):
            fig.add_trace(
                go.Scatter(x=x_axis, y=demand_1[column_name],
                           name=name_demand_1,
                           line=dict(color="blue")),

                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_axis, y=demand_2[column_name],
                           name=name_demand_2,
                           line=dict(color="red")),
                row=i + 1,
                col=1,
            )
        fig.show()

    def compare_yearly_value(self, profile_1, profile_2, name_profile_1, name_profile_2):
        # compare total demand
        total_daniel = profile_1.sum()
        total_ref = profile_2.sum()
        total_df = pd.concat([total_daniel, total_ref], axis=1) / 1_000  # kW
        total_df.columns = [name_profile_1, name_profile_2]
        fig2 = px.bar(
            data_frame=total_df,
            barmode="group",
        )
        fig2.update_layout(title_text="total heating demand")
        fig2.show()

    def compare_model_results(self, heat_demand_daniel, heat_demand_ref):
        # compare the heat demand:
        self.compare_hourly_profile(heat_demand_daniel, heat_demand_ref, "IDA ICE", "5R1C")

        pass

    def df_to_csv(self, df, path):
        column_names = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        df.columns = column_names
        df.to_csv(path, sep=";")

    def main(self):
        if self.project_name == "Flex":
            heat_demand_daniel = self.read_daniel_heat_demand()
            heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value)

            self.compare_hourly_profile(heat_demand_daniel, heat_demand_ref, "IDA ICE", "5R1C")
            self.compare_yearly_value(heat_demand_daniel, heat_demand_ref, "IDA ICE", "5R1C")

        else:
            heat_demand_daniel = self.read_daniel_heat_demand()
            heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value)
            heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value)
            indoor_temp_opt = self.read_indoor_temp(OperationTable.ResultOptHour.value)
            # save indoor temp opt to csv for daniel:
            self.df_to_csv(indoor_temp_opt.copy(), self.input_path.parent / Path("ouput_data/indoor_set_temp.csv"))

            indoor_temp_ref = self.read_indoor_temp(OperationTable.ResultRefHour.value)

            self.compare_hourly_profile(heat_demand_opt, heat_demand_ref, "5R1C optimized", "5R1C")

            self.compare_hourly_profile(indoor_temp_opt, indoor_temp_ref, "5R1C optimized", "5R1C")

            # costs
            costs_opt = self.read_costs(table_name=OperationTable.ResultOptYear.value)
            costs_ref = self.read_costs(table_name=OperationTable.ResultRefYear.value)
            self.compare_yearly_value(costs_opt, costs_ref, "5R1C optimized", "5R1C")



        costs_daniel = self.calculate_costs()
        costs_opt = self.calculate_costs()


if __name__ == "__main__":
    CompareModels(project_name="Flex_optimiert").main()

    # project names:
    # Flex_optimiert, Flex


