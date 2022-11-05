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
        ("ID_Building", 3),
        ("ID_SEMS", 2),
        ("ID_Boiler", 2),
        ("ID_SpaceHeatingTank", 2),
        ("ID_HotWaterTank", 2),
        # ("ID_SpaceCoolingTechnology", 2),
        ("ID_PV", 2),
        ("ID_Battery", 2),
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

    def plot_building_pathway(self):
        for scenario_id in range(384, 385):
            logger.info(f'Scenario = {scenario_id}')
            self.get_building_pathway(scenario_id, COMPONENT_CHANGES, COMPONENTS)

    def summarize_building_pathway(
        self,
        usefile: bool = True,
        start_scenario_id: int = 192,
        end_scenario_id: int = 1 #has to be smaler than start id
    ):
        dirname = os.path.dirname(__file__)
        filename_building = os.path.join(dirname, f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_building.json')
        filename_benefit = os.path.join(dirname, f'data/output/pathways_from_{start_scenario_id}_to_{end_scenario_id}_benefit.json')

        if(usefile):
            with open(filename_building, "r") as file:
                building_pathways = json.load(file)
            with open(filename_benefit, "r") as file:
                benefit_pathways = json.load(file)
        else:
            building_pathways = []
            benefit_pathways = []
            for scenario_id in range(start_scenario_id, end_scenario_id-1, -1):
                building_pathway, benefit_pathway = self.get_building_pathway(scenario_id, COMPONENT_CHANGES, COMPONENTS)
                building_pathways.append(building_pathway)
                benefit_pathways.append(benefit_pathway)
            with open(filename_building, "w") as file:
                json.dump(building_pathways, file)
            with open(filename_benefit, "w") as file:
                json.dump(benefit_pathways, file)



if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    # ana.summarize_operation_energy_cost()
    ana.plot_building_pathway()
    #ana.summarize_building_pathway(usefile=False, start_scenario_id=192, end_scenario_id=192)
