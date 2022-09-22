import copy
from typing import TYPE_CHECKING, ClassVar, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from basics.db import create_db_conn
from basics.enums import InterfaceTable
from basics import kit
from models.operation.enums import OperationTable
from models.operation.plotter import OperationPlotter

if TYPE_CHECKING:
    from basics.config import Config
    from basics.plotter import Plotter


logger = kit.get_logger(__name__)


class OperationAnalyzer:
    def __init__(
        self, config: "Config", plotter: ClassVar["Plotter"] = OperationPlotter
    ):
        self.db = create_db_conn(config)
        self.plotter = plotter(config)
        self.opt_hour_df = None
        self.opt_year_df = None
        self.ref_hour_df = None
        self.ref_year_df = None

    @property
    def opt_hour(self):
        if self.opt_hour_df is None:
            self.opt_hour_df = self.db.read_dataframe(
                OperationTable.ResultOptHour.value
            )
        return self.opt_hour_df

    @property
    def opt_year(self):
        if self.opt_year_df is None:
            self.opt_year_df = self.db.read_dataframe(
                OperationTable.ResultOptYear.value
            )
        return self.opt_year_df

    @property
    def ref_hour(self):
        if self.ref_hour_df is None:
            self.ref_hour_df = self.db.read_dataframe(
                OperationTable.ResultRefHour.value
            )
        return self.ref_hour_df

    @property
    def ref_year(self):
        if self.ref_year_df is None:
            self.ref_year_df = self.db.read_dataframe(
                OperationTable.ResultRefYear.value
            )
        return self.ref_year_df

    def get_hour_df(
        self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None
    ):
        if model == "opt":
            df = kit.filter_df(
                self.opt_hour, filter_dict={"ID_Scenario": scenario_id}
            )
        else:
            df = kit.filter_df(
                self.ref_hour, filter_dict={"ID_Scenario": scenario_id}
            )
        if start_hour is not None and end_hour is not None:
            df = df[df["Hour"].between(start_hour, end_hour)]
        return df

    def get_year_df(self, scenario_id: int, model: str):
        if model == "opt":
            df = kit.filter_df(
                self.opt_year, filter_dict={"ID_Scenario": scenario_id}
            )
        else:
            df = kit.filter_df(
                self.ref_year, filter_dict={"ID_Scenario": scenario_id}
            )
        return df

    def compare_opt(self, id1, id2) -> None:
        df1 = kit.filter_df(self.opt_hour, filter_dict={"ID_Scenario": id1})
        df2 = kit.filter_df(self.opt_hour, filter_dict={"ID_Scenario": id2})
        name1 = f"opt_{id1}"
        name2 = f"opt_{id2}"
        self.gen_html(df1, df2, name1, name2)

    def compare_ref(self, id1, id2) -> None:
        df1 = kit.filter_df(self.ref_hour, filter_dict={"ID_Scenario": id1})
        df2 = kit.filter_df(self.ref_hour, filter_dict={"ID_Scenario": id2})
        name1 = f"ref_{id1}"
        name2 = f"ref_{id2}"
        self.gen_html(df1, df2, name1, name2)

    def compare_opt_ref(self, scenario_id: int) -> None:
        opt_df = kit.filter_df(
            self.opt_hour, filter_dict={"ID_Scenario": scenario_id}
        )
        ref_df = kit.filter_df(
            self.ref_hour, filter_dict={"ID_Scenario": scenario_id}
        )
        name1 = f"opt_{scenario_id}"
        name2 = f"ref_{scenario_id}"
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
        fig = make_subplots(
            rows=subplots_number,
            cols=1,
            subplot_titles=sorted(list(df1.columns)),
            shared_xaxes=True,
        )

        for i, column_name in enumerate(sorted(list(df1.columns))):
            fig.add_trace(
                go.Scatter(x=np.arange(8760), y=df1[column_name], name=name1),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=np.arange(8760), y=df2[column_name], name=name2),
                row=i + 1,
                col=1,
            )

        fig.update_layout(height=400 * subplots_number, width=1600)
        fig.show()

    def plot_electricity_balance(
        self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None
    ):
        df = self.get_hour_df(scenario_id, model, start_hour, end_hour)
        values_dict = {
            "Appliance": np.array(df["BaseLoadProfile"]) / 1000,
            "SpaceHeating": np.array(df["E_Heating_HP_out"] + df["Q_HeatingElement"]) / 1000,
            "HotWater": np.array(df["E_DHW_HP_out"]) / 1000,
            "SpaceCooling": np.array(df["E_RoomCooling"]) / 1000,
            "BatteryCharge": np.array(df["BatCharge"]) / 1000,
            "VehicleCharge": np.array(df["EVCharge"]) / 1000,
            "Grid": np.array(-df["Grid"]) / 1000,
            "PV": np.array(-(df["PV2Load"] + df["PV2Bat"] + df["PV2EV"])) / 1000,
            "BatteryDischarge": np.array(-df["BatDischarge"]) / 1000,
            "VehicleDischarge": np.array(-df["EVDischarge"]) / 1000,
            "PV2Grid": np.array(-df["PV2Grid"]) / 1000,
        }
        self.plotter.bar_figure(
            values_dict,
            f"ElectricityBalance_S{scenario_id}_H{start_hour}To{end_hour}_{model}",
            x_label="Hour",
            y_label="Electricity Demand and Supply (kW)",
            x_lim=None,
            # y_lim=(-8, 8),
        )

    def create_operation_energy_cost_table(self):

        scenarios = self.db.read_dataframe(OperationTable.Scenarios.value)

        def add_total_cost(df: pd.DataFrame, id_sems):
            sce = scenarios.rename(columns={"ID_Scenario": "ID_OperationScenario"})
            sce.insert(loc=1, column="TotalCost", value=df["TotalCost"])
            sce.insert(loc=2, column="ID_SEMS", value=id_sems)
            return sce

        opt_sce = add_total_cost(self.opt_year, id_sems=1)
        ref_sce = add_total_cost(self.ref_year, id_sems=2)
        sce_summary = pd.concat([opt_sce, ref_sce], ignore_index=True, sort=False)
        sce_summary_ids = list(range(1, len(sce_summary) + 1))
        sce_summary.insert(loc=0, column="ID_Scenario", value=sce_summary_ids)

        table_name = InterfaceTable.OperationEnergyCost.value
        self.db.write_dataframe(table_name, sce_summary, if_exists="replace")
        self.db.save_dataframe_to_investment_input(sce_summary, table_name)

    def plot_operation_energy_cost_curve(self):

        operation_energy_cost = self.db.read_dataframe(
            InterfaceTable.OperationEnergyCost.value
        )
        building_ids = operation_energy_cost["ID_Building"].unique()
        scenario_cost_dict = {}

        for id_building in building_ids:
            df = kit.filter_df(
                operation_energy_cost, filter_dict={"ID_Building": id_building}
            )
            costs = sorted(list(df["TotalCost"] / 100), reverse=True)
            scenario_cost_dict[f"ID_Building = {id_building}"] = costs

        self.plotter.step_figure(
            values_dict=scenario_cost_dict,
            fig_name=f"OperationEnergyCost",
            x_label="Scenarios",
            y_label="Energy Cost (€/a)",
            x_lim=(0, 160),
            y_lim=(0, 4000),
        )

    def get_component_energy_cost_change_dict(
        self, col_id_component: str, benchmark_id: int, state_id: int
    ):
        operation_energy_cost = self.db.read_dataframe(
            InterfaceTable.OperationEnergyCost.value
        )

        def get_total_cost(partial_filter: dict, component_state_id: int):
            f = copy.deepcopy(partial_filter)
            f[col_id_component] = component_state_id
            total_cost = kit.filter_df2s(operation_energy_cost, filter_dict=f)[
                "TotalCost"
            ]
            return total_cost

        df = kit.filter_df(
            operation_energy_cost, filter_dict={col_id_component: benchmark_id}
        )
        df = df.drop(
            ["ID_Scenario", "ID_OperationScenario", "TotalCost", col_id_component],
            axis=1,
        )
        component_energy_cost_change_dicts = []
        for index, row in df.iterrows():
            filter_dict = row.to_dict()
            cost_benchmark = get_total_cost(filter_dict, benchmark_id)
            cost_adoption = get_total_cost(filter_dict, state_id)
            cost_change = cost_benchmark - cost_adoption
            component_dict = copy.deepcopy(filter_dict)
            component_dict[col_id_component] = 0
            component_dict["ID_Benchmark"] = benchmark_id
            component_dict["ID_State"] = state_id
            component_dict["CostBenchmark"] = cost_benchmark
            component_dict["CostState"] = cost_adoption
            component_dict["CostChange"] = cost_change
            component_energy_cost_change_dicts.append(component_dict)

        return component_energy_cost_change_dicts

    def create_component_energy_cost_change_tables(
        self, component_changes: List[Tuple[str, int, int]]
    ):
        energy_cost_change_dicts = []
        for component_change_info in component_changes:
            component_name = component_change_info[0]
            benchmark_id = component_change_info[1]
            state_id = component_change_info[2]
            energy_cost_change_dicts += self.get_component_energy_cost_change_dict(
                component_name, benchmark_id, state_id
            )
        df = pd.DataFrame(energy_cost_change_dicts)
        table_name = InterfaceTable.OperationEnergyCostChange.value
        self.db.write_dataframe(table_name, df, if_exists="replace")
        self.db.save_dataframe_to_investment_input(df, table_name)

    def plot_operation_energy_cost_change_curve(
        self, component_changes: List[Tuple[str, int, int]]
    ):
        df = self.db.read_dataframe(InterfaceTable.OperationEnergyCostChange.value)
        values_dict = {}
        for component_change_info in component_changes:
            col_component_id = component_change_info[0]
            benchmark_id = component_change_info[1]
            state_id = component_change_info[2]
            filter_dict = {
                col_component_id: 0,
                "ID_Benchmark": benchmark_id,
                "ID_State": state_id,
            }
            component_df = kit.filter_df(df, filter_dict=filter_dict)
            cost_change = sorted(list(component_df["CostChange"] / 100))
            values_dict[f"{col_component_id}_{benchmark_id}To{state_id}"] = cost_change
        self.plotter.step_figure(
            values_dict=values_dict,
            fig_name=f"OperationEnergyCostChange",
            x_label="Scenarios",
            y_label="Energy Saving Benefit (€/a)",
            y_lim=(-100, 2000),
        )

    def get_operation_energy_cost_change(
        self, building_dict: dict, component_change_info: Tuple[str, int, int]
    ):
        operation_energy_cost_change = self.db.read_dataframe(
            InterfaceTable.OperationEnergyCostChange.value
        )
        col_component_id = component_change_info[0]
        benchmark_id = component_change_info[1]
        state_id = component_change_info[2]
        building_dict = copy.deepcopy(building_dict)
        building_dict[col_component_id] = 0
        building_dict["ID_Benchmark"] = benchmark_id
        building_dict["ID_State"] = state_id
        energy_cost_change = kit.filter_df2s(
            operation_energy_cost_change, filter_dict=building_dict
        )["CostChange"]
        return energy_cost_change

    @staticmethod
    def update_component_changes(
        building_dict: dict, component_changes: List[Tuple[str, int, int]]
    ):
        def implementable(b_dict, c_c_info):
            col_component_id = c_c_info[0]
            benchmark_id = c_c_info[1]
            return b_dict[col_component_id] < benchmark_id

        updated_component_changes = [
            info for info in component_changes if not implementable(building_dict, info)
        ]
        return updated_component_changes

    @staticmethod
    def implement_building_component_change(
        building_dict: dict, component_change_info: Tuple[str, int, int]
    ):
        col_component_id = component_change_info[0]
        benchmark_id = component_change_info[1]
        state_id = component_change_info[2]
        logger.info(
            f"ComponentChange: {col_component_id} = {benchmark_id} -> {col_component_id} = {state_id}"
        )
        assert (
            building_dict[col_component_id] == benchmark_id
        ), "ComponentChange cannot be implemented."
        building_dict[col_component_id] = state_id
        return building_dict

    def get_building_pathway(
        self, scenario_id: int, component_changes: List[Tuple[str, int, int]]
    ):
        df = self.db.read_dataframe(InterfaceTable.OperationEnergyCost.value)
        df = df.drop(["ID_OperationScenario", "TotalCost"], axis=1)
        building_dict = kit.filter_df2s(
            df, filter_dict={"ID_Scenario": scenario_id}
        ).to_dict()
        del building_dict["ID_Scenario"]
        logger.info(f"Initial building: {building_dict}.")
        component_changes = self.update_component_changes(
            building_dict, component_changes
        )

        while len(component_changes) > 0:
            component_change_benefit = []
            for component_change_info in component_changes:
                benefit = self.get_operation_energy_cost_change(
                    building_dict, component_change_info
                )
                component_change_benefit.append([component_change_info, benefit])
            component_change_benefit = sorted(
                component_change_benefit, key=lambda x: x[1], reverse=True
            )
            component_change_info, benefit = component_change_benefit.pop(0)
            building_dict = self.implement_building_component_change(
                building_dict, component_change_info
            )
            logger.info(f"Benefit: {round(benefit/100, 2)} (€/a).")
            # By combining the investment cost with benefit, we can identify if there is a "lock-in" effect.
            # Because the system can be totally stuck somewhere, or it can go to a different direction.
            # Then in total, we can start from all possible "starting points" and see,
            # how many of them are locked and how much further "energy transition" we could have gone.
            component_changes.remove(component_change_info)
            component_changes = self.update_component_changes(
                building_dict, component_changes
            )
        logger.info(f"Final building: {building_dict}.")
