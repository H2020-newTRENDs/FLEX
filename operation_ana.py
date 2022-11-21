from models.operation.analyzer import OperationAnalyzer
from config import config
from basics import kit

import json
import os

logger = kit.get_logger(__name__)

# (str: component name, int: start state, int: end state)
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

# (str: component name, int: amount possible states)
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

    def summarize_component_interaction_specific(self, component_change, identify_component):
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_component_interaction_specific(component_change, identify_component)

    def summarize_component_interaction_full(self, component_change):
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        components = list(filter(lambda item: item[0] != component_change[0], COMPONENTS))  # We don't change COMPONENTS
        self.plot_component_interaction_full(component_change, components)

    def summarize_component_interaction_heatmap(self, component_change, component_1, component_2):
        # Order of List COMPONENTS is important on position of Scenarios in HeatMap
        # and therefore crucial for the interpretation of the figure.
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_component_interaction_heatmap(component_change, component_1, component_2, COMPONENTS)

    def create_building_pathway_json(self, total_scenario: int, filename_building: str, filename_costs: str):
        building_pathways = []
        scenario_costs = {}
        for scenario_id in range(1, total_scenario + 1):
            logger.info(f'Scenario = {scenario_id}')
            building_pathway, benefit_pathway, initial_cost = self.get_building_pathway(scenario_id, COMPONENT_CHANGES,
                                                                                        COMPONENTS)
            building_pathways.append(building_pathway)
            scenario_costs[building_pathway[0]] = initial_cost
        with open(filename_building, "w") as file:
            json.dump(building_pathways, file)
        with open(filename_costs, "w") as file:
            json.dump(scenario_costs, file)

    def plot_building_pathway(self, scenario_id: int, total_scenario: int):

        filename_building = os.path.join(self.output_folder, f'pathway_building.json')
        filename_costs = os.path.join(self.output_folder, f'pathway_cost.json')
        if not (os.path.isfile(filename_building) and os.path.isfile(filename_costs)):
            self.create_building_pathway_json(
                total_scenario=total_scenario,
                filename_building=filename_building,
                filename_costs=filename_costs
            )

        with open(filename_building, "r") as file:
            building_pathways = json.load(file)
        with open(filename_costs, "r") as file:
            scenario_costs = json.load(file)

        draw_pathway, draw_pathway_benefits, initial_cost = self.get_building_pathway(
            scenario_id,
            COMPONENT_CHANGES,
            COMPONENTS
        )
        self.plot_pathway_in_graph(
            COMPONENTS,
            scenario_costs,
            building_pathways,
            draw_pathway,
            draw_pathway_benefits,
            scenario_id
        )


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    # ana.plot_scenario_comparison()
    # ana.plot_scenario_electricity_balance()
    # ana.summarize_operation_energy_cost()
    # ana.summarize_operation_energy_cost_change()
    # ana.summarize_component_interaction_specific(("ID_PV", 2, 1), ("ID_Battery", 2))
    # ana.summarize_component_interaction_full(("ID_PV", 2, 1))
    # ana.summarize_component_interaction_heatmap(("ID_PV", 2, 1), ("ID_Boiler", 2), ("ID_SpaceHeatingTank", 2))
    # ana.plot_building_pathway(scenario_id=384, total_scenario=384)

