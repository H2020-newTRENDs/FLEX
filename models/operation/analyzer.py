from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from basics.plotter import Plotter
from basics.db import create_db_conn
from basics.enums import InterfaceTable
from basics import kit
from models.operation.enums import OperationScenarioComponent as OSC
from models.operation.enums import OperationTable
from models.operation.plotter import OperationPlotter

if TYPE_CHECKING:
    from basics.config import Config


logger = kit.get_logger(__name__)


class OperationAnalyzer:

    def __init__(self, config: 'Config', plotter: ClassVar['Plotter'] = OperationPlotter):
        self.db = create_db_conn(config)
        self.plotter = plotter(config)
        self.opt_hour_df = None
        self.opt_year_df = None
        self.ref_hour_df = None
        self.ref_year_df = None

    @property
    def opt_hour(self):
        if self.opt_hour_df is None:
            self.opt_hour_df = self.db.read_dataframe(OperationTable.ResultOptHour.value)
        return self.opt_hour_df

    @property
    def opt_year(self):
        if self.opt_year_df is None:
            self.opt_year_df = self.db.read_dataframe(OperationTable.ResultOptYear.value)
        return self.opt_year_df

    @property
    def ref_hour(self):
        if self.ref_hour_df is None:
            self.ref_hour_df = self.db.read_dataframe(OperationTable.ResultRefHour.value)
        return self.ref_hour_df

    @property
    def ref_year(self):
        if self.ref_year_df is None:
            self.ref_year_df = self.db.read_dataframe(OperationTable.ResultRefYear.value)
        return self.ref_year_df

    def get_hour_df(self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None):
        if model == "opt":
            df = kit.filter_df(self.opt_hour, filter_dict={OSC.Scenario.id: scenario_id})
        else:
            df = kit.filter_df(self.ref_hour, filter_dict={OSC.Scenario.id: scenario_id})
        if start_hour is not None and end_hour is not None:
            df = df[df["Hour"].between(start_hour, end_hour)]
        return df

    def get_year_df(self, scenario_id: int, model: str):
        if model == "opt":
            df = kit.filter_df(self.opt_year, filter_dict={OSC.Scenario.id: scenario_id})
        else:
            df = kit.filter_df(self.ref_year, filter_dict={OSC.Scenario.id: scenario_id})
        return df

    def compare_opt(self, id1, id2) -> None:
        df1 = kit.filter_df(self.opt_hour, filter_dict={OSC.Scenario.id: id1})
        df2 = kit.filter_df(self.opt_hour, filter_dict={OSC.Scenario.id: id2})
        name1 = f'opt_{id1}'
        name2 = f'opt_{id2}'
        self.gen_html(df1, df2, name1, name2)

    def compare_ref(self, id1, id2) -> None:
        df1 = kit.filter_df(self.ref_hour, filter_dict={OSC.Scenario.id: id1})
        df2 = kit.filter_df(self.ref_hour, filter_dict={OSC.Scenario.id: id2})
        name1 = f'ref_{id1}'
        name2 = f'ref_{id2}'
        self.gen_html(df1, df2, name1, name2)

    def compare_opt_ref(self, scenario_id: int) -> None:
        opt_df = kit.filter_df(self.opt_hour, filter_dict={OSC.Scenario.id: scenario_id})
        ref_df = kit.filter_df(self.ref_hour, filter_dict={OSC.Scenario.id: scenario_id})
        name1 = f'opt_{scenario_id}'
        name2 = f'ref_{scenario_id}'
        self.gen_html(opt_df, ref_df, name1, name2)

    @staticmethod
    def gen_html(df1, df2, name1: str, name2: str):

        assert sorted(list(df1.columns)) == sorted(list(df2.columns))
        # determine how many subplots are needed by excluding profiles that are zero in both modes
        for column_name in df1.columns:
            if (df1[column_name] == 0).all() and (df2[column_name] == 0).all():
                df1 = df1.drop(columns=[column_name])
                df2 = df2.drop(columns=[column_name])
                continue

        # count the columns which will be the number of subplots:
        subplots_number = len(list(df1.columns))
        fig = make_subplots(rows=subplots_number, cols=1,
                            subplot_titles=sorted(list(df1.columns)),
                            shared_xaxes=True)

        for i, column_name in enumerate(sorted(list(df1.columns))):
            fig.add_trace(go.Scatter(x=np.arange(8760), y=df1[column_name], name=name1), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(8760), y=df2[column_name], name=name2), row=i + 1, col=1)

        fig.update_layout(height=400 * subplots_number, width=1600)
        fig.show()

    def create_operation_energy_cost_table(self):

        scenarios = self.db.read_dataframe(OperationTable.Scenarios.value)

        def add_total_cost(df: pd.DataFrame, id_sems):
            sce = scenarios.rename(columns={OSC.Scenario.id: 'ID_OperationScenario'})
            sce.insert(loc=1, column="TotalCost", value=df["TotalCost"])
            sce.insert(loc=2, column="ID_SEMS", value=id_sems)
            return sce

        opt_sce = add_total_cost(self.opt_year, id_sems=1)
        ref_sce = add_total_cost(self.ref_year, id_sems=2)
        sce_summary = pd.concat([opt_sce, ref_sce], ignore_index=True, sort=False)
        sce_summary_ids = list(range(1, len(sce_summary) + 1))
        sce_summary.insert(loc=0, column=OSC.Scenario.id, value=sce_summary_ids)

        self.db.write_dataframe(InterfaceTable.OperationEnergyCost.value, sce_summary, if_exists='replace')
        self.db.save_dataframe_to_investment_input(sce_summary, InterfaceTable.OperationEnergyCost.value)

    def plot_operation_energy_cost_curve(self):

        operation_energy_cost = self.db.read_dataframe(InterfaceTable.OperationEnergyCost.value)
        building_ids = operation_energy_cost["ID_Building"].unique()
        scenario_cost_dict = {}

        for id_building in building_ids:
            df = kit.filter_df(operation_energy_cost, filter_dict={"ID_Building": id_building})
            costs = sorted(list(df["TotalCost"] / 100), reverse=True)
            scenario_cost_dict[f'ID_Building = {id_building}'] = costs

        self.plotter.step_figure(values_dict=scenario_cost_dict, fig_name=f"OperationEnergyCost",
                                 x_label="Scenarios", y_label="Energy Cost (€/a)",
                                 x_lim=(0, 160), y_lim=(0, 4000))

    def plot_electricity_balance(self, scenario_id: int, model: str,
                                 start_hour: int = None, end_hour: int = None):
        df = self.get_hour_df(scenario_id, model, start_hour, end_hour)
        values_dict = {
            "Appliance": np.array(df['BaseLoadProfile'])/1000,
            "SpaceHeating": np.array(df['E_Heating_HP_out'] + df['Q_HeatingElement'])/1000,
            "HotWater": np.array(df['E_DHW_HP_out'])/1000,
            "SpaceCooling": np.array(df['E_RoomCooling'])/1000,
            "BatteryCharge": np.array(df['BatCharge'])/1000,
            "VehicleCharge": np.array(df['EVCharge'])/1000,
            "Grid": np.array(-df['Grid'])/1000,
            "PV": np.array(-(df['PV2Load'] + df['PV2Bat'] + df['PV2EV']))/1000,
            "BatteryDischarge": np.array(-df['BatDischarge'])/1000,
            "VehicleDischarge": np.array(-df['EVDischarge'])/1000,
            "PV2Grid": np.array(-df['PV2Grid']) / 1000,
        }
        self.plotter.bar_plot(values_dict, f"ElectricityBalance_{model}_H{start_hour}To{end_hour}",
                              x_label="Hour", y_label="Electricity Demand and Supply (kW)",
                              x_lim=None, y_lim=(-8, 8))
























