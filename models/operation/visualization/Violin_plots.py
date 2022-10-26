# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import functools
from typing import Callable
import re

from basics.plotter import Plotter
from basics.db import DB
from basics.config import Config
from config import config
from basics.enums import Color
from models.operation.enums import OperationTable, OperationScenarioComponent

ComposableFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


class ViolinPlots(Plotter):

    def __init__(self, config: "Config"):
        super().__init__(config)
        self.db = DB(config)

    def determine_ylabel(self, label) -> str:
        """returns the string of the ylabel"""
        if label == "Grid":
            ylabel = "Electricity demand from the grid (kWh)"
        elif label == "TotalCost":
            ylabel = "Operation cost in €/year"
        elif label == "PVSelfSufficiencyRate":
            ylabel = "self sufficiency (%)"
        elif label == "Load":
            ylabel = "total electricity demand (kWh)"
        elif label == "PV2Grid":
            ylabel = "PV generation sold to the grid (kWh)"
        elif label == "PVSelfConsumptionRate":
            ylabel = "Self-consumption rate (%)"
        else:
            print("cant determine label")
            ylabel = label

        return ylabel

    def add_building_ID_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """add the building ID to each scneario in the df"""
        building_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                              column_names=[OperationScenarioComponent.Building.id]).to_numpy()
        df.loc[:, OperationScenarioComponent.Building.id] = building_ids
        return df

    def add_PV_size_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """add the corresponding PV size to the df for each scenario"""
        id_pv = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                       column_names=[OperationScenarioComponent.PV.id]).to_numpy().flatten()
        pv_table = self.db.read_dataframe(table_name=OperationScenarioComponent.PV.table_name)
        sizes = {id: pv_table.loc[pv_table.loc[:, OperationScenarioComponent.PV.id] == id, "size"] for id in np.unique(id_pv)}
        df.loc[:, OperationScenarioComponent.PV.id] = [sizes[id] for id in id_pv]
        return df

    def add_Tank_size_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """add the corresponding tank size to the df for each scenario"""
        tank_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                          column_names=[OperationScenarioComponent.SpaceHeatingTank.id]).to_numpy().flatten()
        tank_table = self.db.read_dataframe(table_name=OperationScenarioComponent.SpaceHeatingTank.table_name)
        sizes = {id: tank_table.loc[tank_table.loc[:, OperationScenarioComponent.SpaceHeatingTank.id] == id, "size"] for id in np.unique(tank_ids)}
        df.loc[:, OperationScenarioComponent.SpaceHeatingTank.id] = [sizes[id] for id in tank_ids]
        return df

    def add_DHW_size_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """add the corresponding DHW size to the df for each scenario"""
        tank_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                          column_names=[OperationScenarioComponent.HotWaterTank.id]).to_numpy().flatten()
        dhw_table = self.db.read_dataframe(table_name=OperationScenarioComponent.HotWaterTank.table_name)
        sizes = {id: dhw_table.loc[dhw_table.loc[:, OperationScenarioComponent.HotWaterTank.id] == id, "size"] for id in np.unique(tank_ids)}
        df.loc[:, OperationScenarioComponent.HotWaterTank.id] = [sizes[id] for id in tank_ids]
        return df

    def add_battery_size_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        battery_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                             column_names=[OperationScenarioComponent.Battery.id]).to_numpy().flatten()
        battery_table = self.db.read_dataframe(table_name=OperationScenarioComponent.Battery.table_name)
        sizes = {id: battery_table.loc[battery_table.loc[:, OperationScenarioComponent.Battery.id] == id, "capacity"] for id in
                 np.unique(battery_ids)}
        df.loc[:, OperationScenarioComponent.Battery.id] = [sizes[id] for id in battery_ids]
        return df

    def add_boiler_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        boiler_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                            column_names=[OperationScenarioComponent.Boiler.id]).to_numpy().flatten()
        boiler_table = self.db.read_dataframe(table_name=OperationScenarioComponent.Boiler.table_name)
        types = {id: boiler_table.loc[boiler_table.loc[:, OperationScenarioComponent.Boiler.id] == id, "type"]
                 for id in
                 np.unique(boiler_ids)}
        df.loc[:, OperationScenarioComponent.Boiler.id] = [types[id] for id in boiler_ids]
        return df

    def add_space_cooling_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        ac_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                        column_names=[OperationScenarioComponent.SpaceCoolingTechnology.id]).to_numpy().flatten()
        ac_table = self.db.read_dataframe(table_name=OperationScenarioComponent.SpaceCoolingTechnology.table_name)
        sizes = {id: ac_table.loc[ac_table.loc[:, OperationScenarioComponent.SpaceCoolingTechnology.id] == id, "power"]
                 for id in np.unique(ac_ids)}
        df.loc[:, OperationScenarioComponent.SpaceCoolingTechnology.id] = [sizes[id] for id in ac_ids]
        return df

    def add_heating_element_id_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        heating_element_ids = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                                     column_names=[OperationScenarioComponent.HeatingElement.id]).to_numpy().flatten()
        ac_table = self.db.read_dataframe(table_name=OperationScenarioComponent.HeatingElement.table_name)
        sizes = {id: ac_table.loc[ac_table.loc[:, OperationScenarioComponent.HeatingElement.id] == id, "power"]
                 for id in np.unique(heating_element_ids)}
        df.loc[:, OperationScenarioComponent.HeatingElement.id] = [sizes[id] for id in heating_element_ids]
        return df

    def filter_after_price_ID(self, df: pd.DataFrame, price_id: int) -> pd.DataFrame:
        """return the df containing only the IDs which were calculated with a specific price ID"""
        scenario_table = self.db.read_dataframe(table_name=OperationTable.Scenarios.value,
                                                filter={OperationScenarioComponent.EnergyPrice.id: price_id})
        scenarios = scenario_table.loc[:, "ID_Scenario"]
        return_df = df.loc[df.loc[:, "ID_Scenario"].isin(list(scenarios.to_numpy())), :].reset_index(drop=True)
        return return_df

    def create_violin_plot(self, Frame: pd.DataFrame, x_axis: str, y_axis: str, title: str) -> None:
        matplotlib.rc("font", size=14)
        fig = plt.figure()
        fig.suptitle(title)
        ax = plt.gca()
        sns.violinplot(x=x_axis, y=y_axis, hue="", data=Frame.rename(columns={"Option": ""}),
                       ax=ax, split=True, inner="stick", palette="muted", cut=0)

        ax.set_ylabel(self.determine_ylabel(y_axis))
        ax.legend()

        ax.grid(axis="y")
        # format 1000 to 1 000 if the numbers in the y-axis are higher than 4 digits:
        if ax.get_ylim()[1] > 999:
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        ax.yaxis.set_tick_params(labelbottom=True)

        figure_name = f"ViolinPlot_{y_axis}_{x_axis}_with_{title}"
        plt.tight_layout()
        plt.savefig(self.fig_folder / Path(figure_name + ".png"), dpi=200, format='PNG')
        # plt.savefig(self.fig_folder / Path(figure_name + ".svg"))
        plt.show()

    def create_violin_plot_df(self) -> pd.DataFrame:
        # create a big frame which is used to generate the violin plots:
        reference_results = self.db.read_dataframe(table_name=OperationTable.ResultRefYear.value)
        optimization_results = self.db.read_dataframe(table_name=OperationTable.ResultOptYear.value)
        # add column with label of the model mode
        reference_results.loc[:, "Option"] = "simulation"
        optimization_results.loc[:, "Option"] = "SEMS"
        # add certain IDs for the analysis to the frame
        adding_composition = compose(self.add_building_ID_to_df,
                                     self.add_PV_size_to_df,
                                     self.add_Tank_size_to_df,
                                     self.add_DHW_size_to_df,
                                     self.add_battery_size_to_df,
                                     self.add_boiler_to_df,
                                     self.add_space_cooling_to_df)
        reference_results = adding_composition(reference_results)
        optimization_results = adding_composition(optimization_results)
        frame = pd.concat([reference_results, optimization_results], axis=0)
        # changing the total costs from cent/kWh to €/kWh
        frame.loc[:, "TotalCost"] = frame.loc[:, "TotalCost"] / 100
        # change the grid from W to kWh
        frame.loc[:, "Grid"] = frame.loc[:, "Grid"] / 1_000
        return frame

    def main(self):
        df = self.create_violin_plot_df()
        # this must be manually set!
        list_of_component_ids = [
            OperationScenarioComponent.PV.id,
            OperationScenarioComponent.Boiler.id,
            OperationScenarioComponent.Battery.id,
            OperationScenarioComponent.HotWaterTank.id,
            OperationScenarioComponent.SpaceCoolingTechnology.id,
            OperationScenarioComponent.SpaceHeatingTank.id,
            OperationScenarioComponent.Building.id
        ]
        # this part can be used if different price IDs are distinguished
        # for key, value in {1: "variable", 2: "flat"}.items():
        #     # key is the price ID
        #     price_type_frame = self.filter_after_price_ID(df=df, price_id=key).reset_index(drop=True)

        # iterate through the components and plot the violin plots:
        for component in list_of_component_ids:
            self.create_violin_plot(df, x_axis=component, y_axis="TotalCost", title=f"")
            self.create_violin_plot(df, x_axis=component, y_axis="Grid", title=f"")


if __name__ == "__main__":
    Violin = ViolinPlots(config)
    Violin.main()

