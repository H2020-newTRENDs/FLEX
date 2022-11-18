from models.operation.analyzer import OperationAnalyzer
from config import config
from basics import kit

import json
import os

logger = kit.get_logger(__name__)

COMPONENT_CHANGES = [
    ("ID_Building", 3, 2),
    ("ID_Building", 3, 1),
    ("ID_Building", 2, 1),
    ("ID_SEMS", 2, 1),
    ("ID_Boiler", 2, 1),
    ("ID_SpaceHeatingTank", 2, 1),
    ("ID_HotWaterTank", 2, 1),
    # ("ID_SpaceCoolingTechnology", 2, 1),
    ("ID_PV", 2, 1),
    ("ID_Battery", 2, 1),
]
COMPONENTS = [
    ("ID_PV", 2),
    ("ID_SEMS", 2),
    ("ID_Boiler", 2),
    ("ID_SpaceHeatingTank", 2),
    ("ID_HotWaterTank", 2),
    # ("ID_SpaceCoolingTechnology", 2),
    ("ID_Battery", 2),
    ("ID_Building", 3),
]


class ProjectOperationAnalyzer(OperationAnalyzer):

    def plot_scenario_comparison(self):
        # self.compare_opt_ref(1)
        self.compare_opt(id1=119, id2=126)
        # self.compare_ref(id1=1, id2=16)

    def plot_scenario_electricity_balance(self):
        scenario_ids = [1]
        winter_hours = (25, 192)
        summer_hours = (5761, 5928)
        hour_ranges = [winter_hours, summer_hours]
        models = ["opt", "ref"]
        for scenario_id in scenario_ids:
            for model in models:
                for hour_range in hour_ranges:
                    self.plot_electricity_balance(
                        scenario_id=scenario_id,
                        model=model,
                        start_hour=hour_range[0],
                        end_hour=hour_range[1],
                    )

    def summarize_operation_energy_cost(self):
        self.create_operation_energy_cost_table()
        self.plot_operation_energy_cost_curve()

    def summarize_operation_energy_cost_change(self):
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_operation_energy_cost_change_curve(COMPONENT_CHANGES)

    def summarize_operation_energy_cost_change_relation_two_components(
            self,
            component_change, # (str: component name, int: start state, int: end state)
            identify_component): # (str: component name, int: amount possible states)
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_operation_energy_cost_change_relation_two_components(component_change, identify_component)

    def summarize_operation_energy_cost_change_for_one_component(
            self,
            component_change): # (str: component name, int: start state, int: end state)
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        components = list(filter(lambda item: item[0] != component_change[0], COMPONENTS)) # so that we dont change COMPONENTS
        self.plot_operation_energy_cost_change_curve_one_component(component_change, components)

    def summarize_operation_energy_cost_change_heatmap(self, component_change, component_1, component_2):
        # Order of List COMPONENTS is important on position of Scenarios in HeatMap and therefor crucial for the interpretation of the figure.
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_operation_energy_cost_change_heatmap(component_change, component_1, component_2, COMPONENTS)

    def plot_building_pathway(
            self,
            start_scenario_id: int = 192,
            end_scenario_id: int = 1  # has to be smaller than start id#
    ):
        dirname = os.path.dirname(__file__)
        filename_building = os.path.join(dirname,
                                         f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_building.json')
        filename_costs = os.path.join(dirname,
                                      f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_costs.json')

        building_pathways = []
        scenario_costs = {}
        for scenario_id in range(start_scenario_id, end_scenario_id - 1, -1):
            logger.info(f'Scenario = {scenario_id}')
            building_pathway, benefit_pathway, initial_cost = self.get_building_pathway(scenario_id, COMPONENT_CHANGES,
                                                                                        COMPONENTS)
            building_pathways.append(building_pathway)
            scenario_costs[building_pathway[0]] = initial_cost
        with open(filename_building, "w") as file:
            json.dump(building_pathways, file)
        with open(filename_costs, "w") as file:
            json.dump(scenario_costs, file)

    def summarize_building_pathway(
            self,
            scenario_id: int,
            usefile: bool = False,
            start_scenario_id: int = 192,
            end_scenario_id: int = 1,  # has to be smaller than start id
    ):
        dirname = os.path.dirname(__file__)
        filename_building = os.path.join(dirname,
                                         f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_building.json')
        filename_costs = os.path.join(dirname,
                                      f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_costs.json')

        if not usefile:
            self.plot_building_pathway()  # create files

        with open(filename_building, "r") as file:
            building_pathways = json.load(file)
        with open(filename_costs, "r") as file:
            scenario_costs = json.load(file)

        draw_pathway, draw_pathway_benefits, initial_cost = self.get_building_pathway(scenario_id, COMPONENT_CHANGES,
                                                                                      COMPONENTS)
        self.plot_pathway_in_graph(COMPONENTS, scenario_costs, building_pathways, draw_pathway, draw_pathway_benefits, scenario_id)


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    # ana.summarize_operation_energy_cost_change_relation_two_components(("ID_PV", 2, 1), ("ID_Battery", 2))
    # ana.summarize_operation_energy_cost_change_for_one_component(("ID_PV", 2, 1))
    # ana.plot_building_pathway(start_scenario_id=192, end_scenario_id=192)
    # ana.summarize_building_pathway(usefile=True, scenario_id=96)
    ana.summarize_operation_energy_cost_change_heatmap(("ID_PV", 2, 1), ("ID_Boiler", 2), ("ID_SpaceHeatingTank", 2))
