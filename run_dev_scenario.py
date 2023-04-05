import copy

import pandas as pd
from flex.config import Config
cfg = Config(project_name="EV1000")
VEHICLE_TABLE = "OperationScenario_Component_Vehicle.csv"
SCENARIO_TABLE = "OperationScenario.xlsx"


def get_vehicle_ids():
    df = pd.read_csv(f'{cfg.input_operation}/{VEHICLE_TABLE}')
    return df["ID_Vehicle"].tolist()


def gen_scenario_table():
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
        [1, 3, 2, 1, 1, 1, 1, 1, 1],
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
    df.insert(0, "ID_Scenario", value=list(range(1, len(df)+1)))
    df.to_excel(f'{cfg.input_operation}/{SCENARIO_TABLE}', index=False)


if __name__ == "__main__":
    gen_scenario_table()
