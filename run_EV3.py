from db_init import ProjectDatabaseInit
import logging
import copy

import pandas as pd
from flex.config import Config
from flex_operation.run import run_operation_model
from flex_operation.run import find_infeasible_scenarios
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


def gen_scenario_table(config):
    VEHICLE_TABLE = "OperationScenario_Component_Vehicle.csv"
    SCENARIO_TABLE = "OperationScenario.xlsx"

    def get_vehicle_ids():
        df = pd.read_csv(f'{config.input_operation}/{VEHICLE_TABLE}')
        return df["ID_Vehicle"].tolist()

    scenarios = []
    cols = [
        "ID_Region",
        "ID_Building",
        "ID_Boiler",
        "ID_HeatingElement",
        "ID_SpaceHeatingTank",
        "ID_HotWaterTank",
        "ID_SpaceCoolingTechnology",
        "ID_EnergyPrice",
        "ID_Behavior",
        "ID_PV",
        "ID_Battery",
        "ID_Vehicle",
    ]
    other_ids = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1, 1, 1, 1, 1],
    ]
    pv_ids = [1, 2]
    battery_ids = [1, 2]
    vehicle_ids = get_vehicle_ids()
    for other_id in other_ids:
        for pv_id in pv_ids:
            for battery_id in battery_ids:
                for vehicle_id in vehicle_ids:
                    scenario_ids = copy.copy(other_id) + [pv_id, battery_id, vehicle_id]
                    scenarios.append(scenario_ids)
    df = pd.DataFrame(scenarios, columns=cols)
    df.insert(0, "ID_Scenario", value=list(range(1, len(df) + 1)))
    df.to_excel(f'{config.input_operation}/{SCENARIO_TABLE}', index=False)


def run_init(config):
    init = ProjectDatabaseInit(config=config)
    init.run()


def run_scenarios(config):
    operation_scenario_ids = [id_scenario for id_scenario in range(5781, 6177)]  # 6177
    run_operation_model(operation_scenario_ids, config)


if __name__ == "__main__":
    cfg = Config(project_name="EV6000")
    # gen_scenario_table(cfg)
    # run_init(cfg)
    run_scenarios(cfg)
    # find_infeasible_scenarios(cfg)
