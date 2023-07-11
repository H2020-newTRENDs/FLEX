import pandas as pd
import numpy as np
from typing import List

from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable, OperationScenarioComponent


def create_scenario_selection_table(project_name: str) -> pd.DataFrame:
    # grab the hourly load profiles for these ids:
    db = DB(config=Config(project_name))
    selection_table = db.read_dataframe(table_name=OperationTable.Scenarios)
    # replace each ID with the corresponding value except ID Scenario and ID Building:
    column_dict = {
        "ID_PV": "size",
        "ID_Boiler": "type",
        "ID_HotWaterTank": "size",
        "ID_SpaceHeatingTank": "size",
        "ID_Region": "code",
        "ID_HeatingElement": "power",
        "ID_SpaceCoolingTechnology": "power",
        "ID_Battery": "capacity",
        "ID_Vehicle": "capacity",
        "ID_EnergyPrice": "id_electricity",
    }
    for column in selection_table.columns:
        if column == "ID_Scenario" or column == "ID_Building" or column == "ID_Behavior":
            continue
        # read the corresponding input table
        df = db.read_dataframe(table_name=getattr(OperationScenarioComponent, column.replace("ID_", "")).table_name,
                               column_names=[column, column_dict[column]])
        if df.shape[0] > 1:
            mapping = df.set_index(column).squeeze()
        else:
            mapping = df.set_index(column).iloc[:, 0]
        selection_table[column] = selection_table[column].map(mapping)

    return selection_table


def create_scenario_selection_table_with_numbers(project_name: str) -> pd.DataFrame:
    db = DB(config=Config(project_name))
    selection_table = create_scenario_selection_table(project_name)
    numbers_df = db.read_dataframe(table_name="ScenarioNumbers")
    # map numbers to scenario table
    selection_table["number"] = selection_table["ID_Scenario"].map(
        numbers_df.set_index("ID_Scenario", drop=True).squeeze())

    return selection_table


def grab_scneario_table_with_numbers(project_name: str):
    scenario_table = create_scenario_selection_table(project_name=project_name)

    db = DB(config=Config(project_name))
    numbers_df = db.read_dataframe(table_name="ScenarioNumbers")
    # map numbers to scenario table
    scenario_table["number"] = scenario_table["ID_Scenario"].map(
        numbers_df.set_index("ID_Scenario", drop=True).squeeze())
    # drop behavior column and region as i never use them and scenario ID because it doesn't provide additional information
    df = scenario_table.drop(columns=["ID_Region", "ID_Behavior", "ID_Scenario"])
    return df


def load_electricity_price(countries: list[str], years: list[int], project_prefix: str):
    big_df = pd.DataFrame()
    for country in countries:
        for year in years:
            try:
                project_name = f"{project_prefix}_{country}_{year}"
                db = DB(config=Config(project_name))
                numbers_df = db.read_dataframe(table_name=OperationTable.EnergyPriceProfile)
                elec_scenarios = list(db.read_dataframe(table_name=OperationScenarioComponent.EnergyPrice.table_name)["id_electricity"])
                profiles = numbers_df.loc[:, [f"electricity_{i}" for i in elec_scenarios]].astype(float) * 1_000  # cent/kWh
                # rename the electricity_1 to 1 etc..
                profiles.columns = [name.replace("electricity_", "") for name in profiles.columns if "electricity_" in name]
                melted_profiles = profiles.melt(value_name="electricity price (cent/kWh)", var_name="price_id")
                melted_profiles["country"] = country
                melted_profiles["year"] = year
                big_df = pd.concat([big_df, melted_profiles])
            except:
                print(f"electricity price could not be loaded for {country} {year}")
    return big_df


def load_building_age_table(project_name: str):
    db = DB(config=Config(project_name))
    # load the building input table
    building_df = db.read_dataframe(table_name=OperationScenarioComponent.Building.table_name,
                                    column_names=["ID_Building", "construction_period_start",
                                                  "construction_period_end", "type",
                                                  "number_buildings_heat_pump_air",
                                                  "number_buildings_heat_pump_ground",
                                                  "number_buildings_electricity"])
    # merge the period-start and period-end columns
    building_df['construction_period'] = building_df['construction_period_start'].astype(str).str.cat(
        building_df['construction_period_end'].astype(str), sep='-')
    df = building_df.drop(columns=["construction_period_start", "construction_period_end"])
    return_df = df.melt(value_vars=["number_buildings_heat_pump_air",
                                                  "number_buildings_heat_pump_ground",
                                                  "number_buildings_electricity"],
                        id_vars=["ID_Building", "construction_period", "type"],
                        var_name="heating_system",
                        value_name="number"
                        )
    return return_df


def grab_yearly_results(project_name: str) -> pd.DataFrame:
    """ returns the reference and optimization yearly results in one Dataframe"""
    db = DB(config=Config(project_name))
    df_ref = db.read_dataframe(
        table_name=OperationTable.ResultRefYear,
    )
    df_opt = db.read_dataframe(
        table_name=OperationTable.ResultOptYear,
    )
    df_ref["mode"] = "reference"
    df_opt["mode"] = "optimization"
    return pd.concat([df_ref, df_opt], axis=0)



if __name__ == "__main__":
    grab_yearly_results("AUT", 2030, ["TotalCost"])
