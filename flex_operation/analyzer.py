import copy
import json
import os
from typing import TYPE_CHECKING, List, Tuple, Type
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flex.db import create_db_conn
from flex import kit
from flex_operation.constants import OperationTable
from flex_operation.plotter import OperationPlotter

if TYPE_CHECKING:
    from flex.config import Config
    from flex.plotter import Plotter

logger = kit.get_logger(__name__)


class OperationAnalyzer:
    def __init__(self, config: "Config", plotter_cls: Type["Plotter"] = OperationPlotter):
        self.db = create_db_conn(config)
        self.output_folder = config.output
        self.plotter = plotter_cls(config)
        self.opt_hour_df = None
        self.opt_year_df = None
        self.ref_hour_df = None
        self.ref_year_df = None

    @property
    def opt_hour(self):
        if self.opt_hour_df is None:
            self.opt_hour_df = self.db.read_dataframe(
                OperationTable.ResultOptHour
            )
        return self.opt_hour_df

    @property
    def opt_year(self):
        if self.opt_year_df is None:
            self.opt_year_df = self.db.read_dataframe(
                OperationTable.ResultOptYear
            )
        return self.opt_year_df

    @property
    def ref_hour(self):
        if self.ref_hour_df is None:
            self.ref_hour_df = self.db.read_dataframe(
                OperationTable.ResultRefHour
            )
        return self.ref_hour_df

    @property
    def ref_year(self):
        if self.ref_year_df is None:
            self.ref_year_df = self.db.read_dataframe(
                OperationTable.ResultRefYear
            )
        return self.ref_year_df

    def get_hour_df(self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None):
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

    def plot_scenario_electricity_balance(self, scenario_id: int):
        winter_hours = (25, 192)
        summer_hours = (4153, 4320)
        hour_ranges = [winter_hours, summer_hours]
        # models = ["opt", "ref"]
        models = ["ref"]  # for Kevan's work
        for model in models:
            for hour_range in hour_ranges:
                self.plot_electricity_balance(
                    scenario_id=scenario_id,
                    model=model,
                    start_hour=hour_range[0],
                    end_hour=hour_range[1],
                )

    def plot_electricity_balance(self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None):
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
            y_lim=(-5, 5),
        )

    def plot_electricity_balance_demand(self, scenario_id: int, model: str, start_hour: int = None, end_hour: int = None):
        df = self.get_hour_df(scenario_id, model, start_hour, end_hour).rolling(window=24).mean()
        df = df.rolling(window=24).mean()
        values_dict = {
            "Appliance": np.array(df["BaseLoadProfile"]) / 1000,
            "SpaceHeating": np.array(df["E_Heating_HP_out"] + df["Q_HeatingElement"]) / 1000,
            "HotWater": np.array(df["E_DHW_HP_out"]) / 1000,
            "SpaceCooling": np.array(df["E_RoomCooling"]) / 1000,
            # "BatteryCharge": np.array(df["BatCharge"]) / 1000,
            # "VehicleCharge": np.array(df["EVCharge"]) / 1000,
            # "Grid": np.array(-df["Grid"]) / 1000,
            # "PV": np.array(-(df["PV2Load"] + df["PV2Bat"] + df["PV2EV"])) / 1000,
            # "BatteryDischarge": np.array(-df["BatDischarge"]) / 1000,
            # "VehicleDischarge": np.array(-df["EVDischarge"]) / 1000,
            # "PV2Grid": np.array(-df["PV2Grid"]) / 1000,
        }
        self.plotter.bar_figure(
            values_dict,
            f"ElectricityBalanceDemand_S{scenario_id}_H{start_hour}To{end_hour}_{model}",
            x_label="Hour",
            y_label="Electricity Demand and Supply (kW)",
            x_lim=None,
            y_lim=(-5, 5),
        )

    def create_operation_energy_cost_table(self):

        scenarios = self.db.read_dataframe(OperationTable.Scenarios)

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
        table_name = OperationTable.ResultEnergyCost
        self.db.write_dataframe(table_name, sce_summary, if_exists="replace")

    def plot_operation_energy_cost_curve(self):

        operation_energy_cost = self.db.read_dataframe(OperationTable.ResultEnergyCost)
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

    def get_component_energy_cost_change_dict(self, col_id_component: str, benchmark_id: int, state_id: int):
        operation_energy_cost = self.db.read_dataframe(OperationTable.ResultEnergyCost)

        def get_total_cost(partial_filter: dict, component_state_id: int):
            f = copy.deepcopy(partial_filter)
            f[col_id_component] = component_state_id
            total_cost = kit.filter_df2s(operation_energy_cost, filter_dict=f)["TotalCost"]
            return total_cost

        df = kit.filter_df(operation_energy_cost, filter_dict={col_id_component: benchmark_id})
        df = df.drop(["ID_Scenario", "ID_OperationScenario", "TotalCost", col_id_component], axis=1)
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

    def create_component_energy_cost_change_tables(self, component_changes: List[Tuple[str, int, int]]):
        energy_cost_change_dicts = []
        for component_change_info in component_changes:
            component_name = component_change_info[0]
            benchmark_id = component_change_info[1]
            state_id = component_change_info[2]
            energy_cost_change_dicts += self.get_component_energy_cost_change_dict(
                component_name, benchmark_id, state_id
            )
        df = pd.DataFrame(energy_cost_change_dicts)
        table_name = OperationTable.ResultEnergyCostChange
        self.db.write_dataframe(table_name, df, if_exists="replace")

    def plot_operation_energy_cost_change_curve(
            self, component_changes: List[Tuple[str, int, int]]
    ):
        df = self.db.read_dataframe(OperationTable.ResultEnergyCostChange)
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

    def plot_component_interaction_specific(
            self,
            component_change: Tuple[str, int, int],
            identify_component: Tuple[str, int],
            save_figure: bool = True
    ):
        df = self.db.read_dataframe(OperationTable.ResultEnergyCostChange)
        values_dict = {}
        col_component_id = component_change[0]
        benchmark_id = component_change[1]
        state_id = component_change[2]
        filter_dict = {
            col_component_id: 0,
            "ID_Benchmark": benchmark_id,
            "ID_State": state_id,
        }
        component_df = kit.filter_df(df, filter_dict=filter_dict)
        component_df = component_df.sort_values(by=['CostChange', identify_component[0]], ascending=[True, False])
        component_df['Scenario'] = range(1, component_df.shape[0] + 1)
        for state in range(1, identify_component[1] + 1):
            filter_dict = {identify_component[0]: state}
            identify_df = kit.filter_df(component_df, filter_dict=filter_dict)
            values_dict[f"{identify_component[0]}_{state}"] = identify_df
        if save_figure:
            self.plotter.dot_figure(
                values_dict=values_dict,
                fig_name=f"OperationEnergyCostChange_{col_component_id}_{identify_component[0]}",
                x_label="Scenarios",
                y_label="Energy Saving Benefit (€/a)",
                y_lim=(-100, 2000),
            )
        return values_dict

    def plot_component_interaction_full(
            self,
            component_change: Tuple[str, int, int],
            identify_components: List[Tuple[str, int]],
    ):
        values_dict = {}
        for identify_component in identify_components:
            values_of_component = self.plot_component_interaction_specific(component_change, identify_component, save_figure=False)
            values_dict[identify_component[0]] = values_of_component

        self.plotter.dot_subplots_figure(
            values_dict=values_dict,
            components=identify_components,
            fig_name=f"OperationEnergyCostChange_{component_change[0]}",
            x_label="Scenarios",
            y_label="Energy Saving Benefit (€/a)",
            y_lim=(-100, 2000),
        )

    def plot_component_interaction_heatmap(
            self,
            component_change: Tuple[str, int, int],
            component_1: Tuple[str, int],
            component_2: Tuple[str, int],
            component_list: List[Tuple[str, int]],
    ):
        df = self.db.read_dataframe(OperationTable.ResultEnergyCostChange)
        component_list = [component[0] for component in component_list if
                          component[0] != component_2[0] and component[0] != component_1[0]
                          and component[0] != component_change[0]]
        df = df.sort_values(by=[component_1[0], component_2[0]] + component_list, ascending=False)
        filter_dict = {
            component_change[0]: 0,
            "ID_Benchmark": component_change[1],
            "ID_State": component_change[2],
        }
        df = kit.filter_df(df, filter_dict=filter_dict)
        y_labels = []

        cost_change = []
        for i, j in itertools.product(range(1, component_1[1]+1), range(1, component_2[1]+1)):
            filter_dict = {
                component_1[0]: i,
                component_2[0]: j,
            }
            component_df = kit.filter_df(df, filter_dict=filter_dict)#
            component_df = component_df['CostChange'] / 100
            cost_change.append(component_df.to_numpy())
            y_labels.append(f'{component_1[0]} = {i}, {component_2[0]} = {j}')

        cost_change = np.array(cost_change)
        x_labels = list(range(1, len(cost_change[0]) + 1))

        self.plotter.heatmap(
            data=cost_change,
            row_labels=y_labels,
            col_labels=x_labels,
            fig_name=f'CostChange_{component_change[0]}_{component_1[0]}_{component_2[0]}',
            title=f'CostChange_{component_change[0]}',
            explanation=f'Sorting: {component_list}',
            cbarlabel="Energy Saving Benefit (€/a)",
        )

    def get_operation_energy_cost_change(
            self, building_dict: dict, component_change_info: Tuple[str, int, int]
    ):
        operation_energy_cost_change = self.db.read_dataframe(OperationTable.ResultEnergyCostChange)
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

    def create_building_pathway_json(
            self,
            filename_building: str,
            filename_costs: str,
            component_changes: List[Tuple[str, int, int]],
            components=List[Tuple[str, int]]
    ):
        building_pathways = []
        scenario_costs = {}
        df = self.db.read_dataframe(OperationTable.ResultEnergyCost)
        for scenario_id in range(1, len(df) + 1):
            logger.info(f'Scenario = {scenario_id}')
            building_pathway, benefit_pathway, initial_cost = self.get_building_pathway(
                scenario_id,
                component_changes,
                components
            )
            building_pathways.append(building_pathway)
            scenario_costs[building_pathway[0]] = initial_cost
        with open(filename_building, "w") as file:
            json.dump(building_pathways, file)
        with open(filename_costs, "w") as file:
            json.dump(scenario_costs, file)

    def plot_building_pathway(
            self,
            scenario_id: int,
            component_changes: List[Tuple[str, int, int]],
            components=List[Tuple[str, int]]
    ):

        filename_building = os.path.join(self.output_folder, f'pathway_building.json')
        filename_costs = os.path.join(self.output_folder, f'pathway_cost.json')
        if not (os.path.isfile(filename_building) and os.path.isfile(filename_costs)):
            self.create_building_pathway_json(
                filename_building=filename_building,
                filename_costs=filename_costs,
                component_changes=component_changes,
                components=components
            )
        with open(filename_building, "r") as file:
            building_pathways = json.load(file)
        with open(filename_costs, "r") as file:
            scenario_costs = json.load(file)
        draw_pathway, draw_pathway_benefits, initial_cost = self.get_building_pathway(
            scenario_id,
            component_changes,
            components
        )
        self.plot_pathway_in_graph(
            components,
            scenario_costs,
            building_pathways,
            draw_pathway,
            draw_pathway_benefits,
            scenario_id
        )

    def get_building_pathway(
            self,
            scenario_id: int,
            component_changes: List[Tuple[str, int, int]],
            components=List[Tuple[str, int]]
    ):
        building_pathway = []
        benefit_pathway = []

        df = self.db.read_dataframe(OperationTable.ResultEnergyCost)
        initial_cost = df[df['ID_Scenario'] == scenario_id]['TotalCost'].item()
        df = df.drop(["ID_OperationScenario", "TotalCost"], axis=1)
        building_dict = kit.filter_df2s(
            df, filter_dict={"ID_Scenario": scenario_id}
        ).to_dict()
        del building_dict["ID_Scenario"]
        building_pathway.append(self.convert_building_dict_to_string(building_dict, components))
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
            building_pathway.append(self.convert_building_dict_to_string(building_dict, components))
            benefit_pathway.append(round(benefit / 100, 2))
            logger.info(f"Benefit: {round(benefit / 100, 2)} (€/a).")
            # By combining the investment cost with benefit, we can identify if there is a "lock-in" effect.
            # Because the system can be totally stuck somewhere, or it can go to a different direction.
            # Then in total, we can start from all possible "starting points" and see,
            # how many of them are locked and how much further "energy transition" we could have gone.
            component_changes.remove(component_change_info)
            component_changes = self.update_component_changes(
                building_dict, component_changes
            )
        logger.info(f"Final building: {building_dict}.")
        return building_pathway, benefit_pathway, initial_cost

    @staticmethod
    def convert_building_dict_to_string(building_dict: dict, components: List[Tuple[str, int]]):
        string_builder = ""
        for component in components:
            string_builder += str(building_dict[component[0]])
        return string_builder

    def plot_component_impact_violin(self):
        scenarios = self.db.read_dataframe(OperationTable.Scenarios).drop('ID_Scenario', axis=1)
        ref_year = copy.deepcopy(self.ref_year).assign(Option="simulation")
        opt_year = copy.deepcopy(self.opt_year).assign(Option="optimization")
        df = pd.concat(
            [
                pd.concat([ref_year, scenarios], axis=1),
                pd.concat([opt_year, scenarios], axis=1)
            ],
            axis=0
        )
        df.loc[:, "TotalCost"] = df.loc[:, "TotalCost"] / 100  # convert from cent to euro
        df.loc[:, "Grid"] = df.loc[:, "Grid"] / 1000  # convert from W to kW

        component_ids = [
            "ID_Building",
            "ID_Boiler",
            # "ID_HeatingElement",
            "ID_SpaceHeatingTank",
            "ID_HotWaterTank",
            "ID_SpaceCoolingTechnology",
            "ID_PV",
            "ID_Battery",
            # "ID_Vehicle",
        ]
        for component_id in component_ids:
            self.plotter.violin_figure(df,
                                       fig_name=f'ImpactViolin_{component_id[3:]}_EnergyCost',
                                       component=component_id, impacted_var="TotalCost",
                                       x_label=component_id, y_label="Operation Cost of Energy (€)")
            self.plotter.violin_figure(df,
                                       fig_name=f'ImpactViolin_{component_id[3:]}_ElectricityConsumption',
                                       component=component_id, impacted_var="Grid",
                                       x_label=component_id, y_label="Electricity Consumption (kWh)")

    def plot_pathway_in_graph(
            self,
            components: List[Tuple[str, int]],
            scenario_costs,
            building_pathways,
            draw_pathway,
            draw_pathway_benefits,
            scenario_id,
    ):
        # create all possible scenarios
        t = [(i for i in range(1, component[1] + 1)) for component in components]
        prod = itertools.product(*t)
        scenarios = []
        for p in prod:
            scenarios.append(''.join([str(value) for value in p]))

        graph = {}
        # create all possible pathway edges
        for key_scenario in scenarios:
            graph[key_scenario] = [value for value in scenarios if
                                   sum(c1 != c2 for c1, c2 in zip(value, key_scenario)) == 1
                                   and
                                   sum(c1 != c2 for c1, c2 in zip('1' * len(key_scenario), key_scenario)) >= sum(
                                       c1 != c2 for c1, c2 in zip('1' * len(key_scenario), value))]

        nodesize = dict.fromkeys(scenarios, 0)
        for pathway in building_pathways:
            for scenario in pathway:
                nodesize[scenario] += 1

        self.plotter.directed_graph(
            graph,
            nodeweight=scenario_costs,
            nodesize=nodesize,
            figname=f'Graph_{scenario_id}',
            draw_pathway=draw_pathway,
            draw_pathway_benefits=draw_pathway_benefits
        )
