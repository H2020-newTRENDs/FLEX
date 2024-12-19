import pandas as pd
import numpy as np
from pathlib import Path

from utils.db import create_db_conn
from projects.main import get_config

from flex_operation import components
import multiprocessing
import shutil
import sqlalchemy
from joblib import Parallel, delayed


COUNTRY_LIST = [
    "AUT",  
    "BEL", 
    # "POL",
    # "CYP", 
    # "PRT",
    # "DNK",
    # "FRA",
    # "CZE",  
    # "DEU", 
    # "HRV",
    # "HUN",
    # "ITA",  
    # "LUX",
    # "NLD",
    # "SVK",
    # "SVN",
    # 'IRL',
    # 'ESP',  
    # 'SWE',
    # 'GRC',
    # 'LVA',  
    # 'LTU',
    # 'MLT',
    # 'ROU',
    # 'BGR',  
    # 'FIN',
    # 'EST',
]

YEARS = [2020, 2030, 2040, 2050]








logger = get_logger(__name__)

def get_european_countries_dict() -> dict:
    EUROPEAN_COUNTRIES = {
        'AUT': 'Austria',  #
        'BEL': 'Belgium',  #
        'BGR': 'Bulgaria',  #
        'HRV': 'Croatia',  #
        "CYP": "Cyprus",
        'CZE': 'Czechia',
        'DNK': 'Denmark',  #
        'EST': 'Estonia',  #
        'FIN': 'Finland',  #
        'FRA': 'France',  #
        'DEU': 'Germany',  #
        'GRC': 'Greece',  #
        'HUN': 'Hungary',  #
        'IRL': 'Ireland',  #
        'ITA': 'Italy',  #
        'LVA': 'Latvia',  #
        'LTU': 'Lithuania',  #
        'LUX': 'Luxembourg',  #
        'MLT': 'Malta',  #
        'NLD': 'Netherlands',  #
        'POL': 'Poland',  #
        'PRT': 'Portugal',  #
        'ROU': 'Romania',  #
        'SVK': 'Slovakia',  #
        'SVN': 'Slovenia',  #
        'ESP': 'Spain',  #
        'SWE': 'Sweden'  #
    }
    return EUROPEAN_COUNTRIES

def get_country_correction_code() -> dict:
    country_code_correction = {
        "AUT": "AT",
        "BEL": "BE",
        "BGR": "BG",
        "HRV": "HR",
        "CYP": "CY",
        "CZE": "CZ",
        "DNK": "DK",
        "EST": "EE",
        "FIN": "FI",
        "FRA": "FR",
        "DEU": "DE",
        "GRC": "GR",
        "HUN": "HU",
        "IRL": "IE",
        "ITA": "IT",
        "LVA": "LV",
        "LTU": "LT",
        "LUX": "LU",
        "MLT": "MT",
        "NLD": "NL",
        "POL": "PL",
        "PRT": "PT",
        "ROU": "RO",
        "SVK": "SK",
        "SVN": "SI",
        "ESP": "ES",
        "SWE": "SE"
    }
    return country_code_correction

# HWB_STEPS = 10  # variable that defines how many buildings will be merged together
def get_battery_penetration() -> dict:
    BATTERY_PENETRATION = {
        2020: 0.05,
        2030: 0.1,
        2040: 0.15,
        2050: 0.2
    }
    return BATTERY_PENETRATION

def get_sfh_mfh_dict() ->dict:
    SFH_MFH = {
        1: "SFH",
        2: "SFH",
        5: "MFH",
        6: "MFH"
    }
    return SFH_MFH


def load_invert_data(path_to_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(path_to_file)
    return df


def create_dict_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def pv_area_to_kWp(name: str) -> float:
    # determine size:
    area = [int(s) for s in name.split("_") if s.isdigit()][0]  # only returns one number
    # area is given in square meters, transfer to kWp
    factor = 8
    return area / factor


def pv_kWp_to_area(number: float) -> float:
    return number * 8


def create_pv_id_table(df: pd.DataFrame) -> pd.DataFrame:
    """ creates the operationScenario_component_pv dataframe that can be the excel input"""

    column_names = [name for name in df.columns.to_list() if "PV" in name]
    pv_table = pd.DataFrame(columns=["ID_PV", "size", "size_unit"], index=range(len(column_names)))
    id_pv = 0
    for i, name in enumerate(column_names):
        pv_power = pv_area_to_kWp(name)
        pv_table.iloc[id_pv, :] = [id_pv + 1, pv_power, "kW_peak"]

        id_pv += 1
    return pv_table


def create_building_id_table(df: pd.DataFrame) -> pd.DataFrame:
    """ creates the operationScenario_component_building dataframe that can be the excel input"""
    columns = list(components.Building.__dataclass_fields__) + ["number_buildings_heat_pump_air",
                                                                "number_buildings_heat_pump_ground",
                                                                "number_buildings_electricity",
                                                                "number_of_buildings"]
    building_table = pd.DataFrame(columns=["ID_Building"] + columns, index=range(len(df)))
    # insert building ID
    building_table.loc[:, "ID_Building"] = building_table.index + 1
    building_table.loc[:, "type"] = df["building_categories_index"].map(get_sfh_mfh_dict())

    # rename the window areas in df
    df = df.rename(columns={"average_effective_area_wind_west_east_red_cool": "effective_window_area_west_east",
                            "average_effective_area_wind_south_red_cool": "effective_window_area_south",
                            "average_effective_area_wind_north_red_cool": "effective_window_area_north",
                            "number_of_persons_per_dwelling": "person_num",
                            "spec_int_gains_cool_watt": "internal_gains"})
    for name in columns:
        if name in df.columns.to_list():
            building_table.loc[:, name] = df.loc[:, name]

    # insert the grid power max:
    building_table.loc[:, "grid_power_max"] = 50_000_000

    return building_table


def map_pv_to_building_id(component_pv: pd.DataFrame,
                          df: pd.DataFrame) -> pd.DataFrame:
    # the PV IDs are added after the permutations to the scenario table:
    column_names = [name for name in df.columns.to_list() if "PV" in name]
    scenario = df[["index"] + column_names]
    new_column_names = ["index"] + [
        str(int(component_pv.loc[component_pv.loc[:, "size"] == pv_area_to_kWp(name), "ID_PV"])) for name in column_names
    ]
    scenario.columns = new_column_names
    scenario_df = scenario.melt(id_vars="index", var_name="ID_PV", value_name="number_of_buildings")
    # drop the zeros:
    scenario_df = scenario_df[scenario_df["number_of_buildings"] > 0].rename(columns={"index": "ID_Building"})
    # check numbers:
    assert round(scenario_df["number_of_buildings"].sum()) == round(df[column_names].sum().sum()), "numbers dont add up"
    return scenario_df.reset_index(drop=True)


def copy_file_to_folder(source: Path, destination: Path, force_copy: bool = False):
    """Copy a file from src to dst, creating any missing directories in the destination path."""
    if force_copy:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
        logger.info(f"copied {source.name} to \n {destination}")
    else:
        # Check if the destination file already exists
        if destination.exists():
            logger.info(f"{destination} already exists. Skipping copy operation.")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
        logger.info(f"copied {source.name} to \n {destination}")


def calc_mean(data: dict) -> float:
    # Multiply each key with its corresponding value and add them
    sum_products = sum(key * value for key, value in data.items())
    # Calculate the sum of the values
    sum_values = sum(value for value in data.values())
    # Return the mean value
    if sum_values == 0:
        return np.nan
    return sum_products / sum_values


def calculate_mean_supply_temperature(grouped_df: pd.DataFrame,
                                      heating_system_name: str = None,
                                      helper_name: str = None) -> float:
    # add supply temperature to bc df:
    # check if there are multiple supply temperatures:
    supply_temperatures = list(grouped_df.loc[:, "supply_temperature"].unique())
    if len(supply_temperatures) > 1:
        # group by supply temperature
        supply_temperature_group = grouped_df.groupby("supply_temperature")
        nums = {}
        for temp in supply_temperatures:
            if helper_name == "get_number_of_buildings":
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)["number_of_buildings"].sum()
            else:
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)[heating_system_name].sum()

            nums[temp] = number_buildings_sup_temp
        # calculate the mean:
        # sometimes the supply temperature is 1000 °C because
        mean_sup_temp = calc_mean(nums)
    else:
        mean_sup_temp = supply_temperatures[0]

    return mean_sup_temp


def calculate_mean(grouped: pd.DataFrame, names: list, heating_system_name: str):
    if grouped[heating_system_name].sum() == 0:  # create nan row so it can be easily dropped later
        new_row = pd.Series(data=[np.nan] * len(grouped[names].columns), index=grouped[names].columns)
    else:
        weights = grouped[heating_system_name] / grouped[heating_system_name].sum()
        new_row = (grouped[names].T * weights).T.sum()
    return new_row


def create_representative_building(group: pd.DataFrame,
                                   heating_system_name: str,
                                   merging_names: list,
                                   adding_names: list) -> pd.DataFrame:
    new_row = pd.DataFrame(columns=group.columns, index=[0])
    # representative air source HP building
    new_row.loc[0, merging_names] = calculate_mean(group, names=merging_names, heating_system_name=heating_system_name)
    new_row.loc[0, adding_names] = group.loc[:, adding_names].sum()

    new_row.loc[0, "supply_temperature"] = calculate_mean_supply_temperature(
        group,
        heating_system_name=heating_system_name
    )
    # new name is first 5 letters of first name + heating system
    heating_system = heating_system_name.split("_")[-1]
    new_row["construction_period_start"] = group["construction_period_start"].unique()
    new_row["construction_period_end"] = group["construction_period_end"].unique()[0]
    new_row.loc[0, "name"] = f"{str(group.loc[:, 'name'].iloc[0][0:5])[2:]}_{heating_system}"
    new_row.loc[0, "index"] = group.loc[:, "index"].iloc[0]  # new index is the first index of merged rows
    return new_row


def reduce_number_of_buildings(df: pd.DataFrame) -> pd.DataFrame:
    """ reducing the number of buildings by merging buildings with a similar norm heating value (hwb_steps)
    the hwb steps describe the step size in which buildings are merged. Eg.: hwb_steps = 5 buildings with
    hwb 66, 67,.., 69 will be merged. The hwb_steps start from 0."""
    new_df = df.copy()
    # test:
    air_hp_numbers = df["number_buildings_heat_pump_air"].sum()
    ground_hp_numbers = df["number_buildings_heat_pump_ground"].sum()

    # fix the supply temperatures: some buildings have supply temperature around 1000°C, for these buildings the supply
    # temperature will be set to 60°C
    new_df.loc[new_df["supply_temperature"] > 60, "supply_temperature"] = 60

    # columns where numbers are summed up (PV and number of buildings)
    adding_names = [name for name in new_df.columns if "number" in name]
    # columns to merge: [2:] so index and name are left out
    merging_names = [
                        name for name in new_df.columns if "PV" and "number" not in name and "construction" not in name
                    ][2:] + adding_names[:3]
    # except number of persons and number of dwellings ([3:]) left out
    adding_names = adding_names[3:]

    # round the hwb column to the next hwb step so they can be merged
    # new_df.loc[:, "hwb_stepped"] = np.ceil(new_df.loc[:, "hwb"] / HWB_STEPS) * HWB_STEPS

    # create groups based on the floor area and building type:
    grouped_df = new_df.groupby(["construction_period_start", "building_categories_index"])
    # for each group create a typical building for air HP, for ground HP and for direct electric heating,
    # because the difference in numbers is relatively high and this way the representative building actually
    # represents the buildings using air source HP in this timeframe
    heating_systems = [
        "number_buildings_heat_pump_air",
        "number_buildings_heat_pump_ground",
    ]
    big_df = pd.DataFrame()
    for index, group in grouped_df:
        air_hp = create_representative_building(group,
                                                heating_system_name="number_buildings_heat_pump_air",
                                                merging_names=merging_names,
                                                adding_names=adding_names)
        ground_hp = create_representative_building(group,
                                                   heating_system_name="number_buildings_heat_pump_ground",
                                                   merging_names=merging_names,
                                                   adding_names=adding_names)

        # set the number of heating systems for the specific buildings to zero except the main one:
        air_hp[["number_buildings_heat_pump_ground", "number_buildings_electricity"]] = 0
        ground_hp[["number_buildings_heat_pump_air", "number_buildings_electricity"]] = 0

        new_group = pd.concat([air_hp, ground_hp]).reset_index(drop=True)
        # sometimes the number of heating systems is zero: drop these columns

        # now we have 2 representative buildings for the group but all these buildings have the total number of PV
        # installations etc. So we have to divide these numbers up as well: take the number of buildings with the
        # respective heating systems as weights
        pv_columns = [name for name in new_group.columns if "PV" in name]

        weights = (new_group.fillna(0)[heating_systems].sum() / new_group.fillna(0)[heating_systems].sum().sum()).reset_index(drop=True)
        new_group[pv_columns] = new_group[pv_columns].astype(float).mul(weights, axis=0)
        # drop the rows that are merged
        # add the new representative buildings
        big_df = pd.concat([big_df, new_group])

    # drop the nan rows because the number of buildings with the respective heating system is zero:
    return_df = big_df.dropna(axis=0).reset_index(drop=True)
    return_df["index"] = return_df.index + 1
    # set the building category index as int
    return_df["building_categories_index"] = return_df["building_categories_index"].apply(lambda x: int(round(x))).astype(int)
    # check:
    assert round(return_df["number_buildings_heat_pump_air"].sum()) == round(air_hp_numbers) and \
           round(return_df["number_buildings_heat_pump_ground"].sum()) == round(ground_hp_numbers) and \
           f"the number of buildings is altered"

    return return_df


def copy_input_operation_files(destination: Path, force_copy: bool = False):
    source_files = ["OperationScenario_BehaviorProfile.xlsx",
                    "OperationScenario_Component_Battery.xlsx",
                    "OperationScenario_Component_Boiler.xlsx",
                    "OperationScenario_Component_EnergyPrice.xlsx",
                    "OperationScenario_Component_HeatingElement.xlsx",
                    "OperationScenario_Component_HotWaterTank.xlsx",
                    "OperationScenario_Component_SpaceCoolingTechnology.xlsx",
                    "OperationScenario_Component_SpaceHeatingTank.xlsx",
                    "OperationScenario_Component_Vehicle.xlsx",
                    ]

    for file in source_files:
        source_file = Path(__file__).parent / "data" / "input_operation" / file
        destination_file = destination / file
        copy_file_to_folder(source_file, destination_file, force_copy)


def create_operation_region_file(year: int, country: str) -> pd.DataFrame:
    columns = ["ID_Region", "code", "year"]
    df = pd.DataFrame(columns=columns, data=[[1, get_country_correction_code()[country], year]])
    return df




def create_energy_price_scenario_file(country: str, year: int, source_path: Path) -> pd.DataFrame:
    columns = ["region", "year", "id_hour", "electricity_1", "electricity_2", "electricity_feed_in_1", "unit"]
    price_df = pd.DataFrame(columns=columns)

    if year > 2020:
        # import electricity prices from AURES: shiny happy corresponds to newTrends and leviathan to no significant changes
        df_newTrends = pd.read_excel(
            source_path / Path("elec_prices_AURES.xlsx"),
            engine="openpyxl",
            sheet_name=f"shiny happy {year}",
            header=3
        ).dropna(axis=1).drop(0)
        # replace "EPS" with nan and thenfrontfill:
        df_newTrends = df_newTrends.replace("EPS", np.nan)
        df_newTrends = df_newTrends.ffill().bfill()
        # 24 hours of data are missing for the AURES prices, therefore we will repeat the last day
        df_newTrends_1 = pd.concat([df_newTrends, df_newTrends.iloc[-24:, :]], axis=0).reset_index(drop=True)
        # €/MWh -> €/kWh -> cent/kWh -> cent/Wh
        elec_price_1 = df_newTrends_1[get_country_correction_code()[country]].astype(float).to_numpy() / 1_000 * 100 / 1_000
        price_df["electricity_1"] = elec_price_1

        # save the leviathan variation as price 2:
        df_oldTrends = pd.read_excel(
            source_path / Path(r"elec_prices_AURES.xlsx"),
            engine="openpyxl",
            sheet_name=f"leviathan {year}",
            header=3
        ).dropna(axis=1).drop(0)
        # replace "EPS" with nan and thenfrontfill:
        df_oldTrends = df_oldTrends.replace("EPS", np.nan)
        df_oldTrends = df_oldTrends.ffill().bfill()
        # 24 hours of data are missing for the AURES prices, therefore we will repeat the last day
        df_newTrends_2 = pd.concat([df_oldTrends, df_oldTrends.iloc[-24:, :]], axis=0).reset_index(drop=True)
        # €/MWh -> €/kWh -> cent/kWh -> cent/Wh
        elec_price_2 = df_newTrends_2[get_country_correction_code()[country]].astype(float).to_numpy() / 1_000 * 100 / 1_000
        price_df["electricity_2"] = elec_price_2

    else:  # prices are taken from ENTSO-E from 2019
        df = pd.read_csv(  # price is in cent/kWh
            source_path / Path(r"ENTSO-E_prices_europe_2019_newTrends.csv"),
            sep=",",
            index_col=False
        )
        # ffill because sometimes a few values in the data are missing
        elec_price = df[get_country_correction_code()[country]].ffill().astype(float).to_numpy()
        # make sure that there are no negative prices by increasing the costs so that the minimum price will be at 20
        # cent/kWh:
        # min_price = abs(np.floor(elec_price.min()))
        elec_price_positive = (elec_price) / 1_000  # cent/kWh->cent/Wh
        # for 2020 only one price scenario exists therefore electricity_1 and electricity_2 will be identical
        price_df["electricity_1"] = elec_price_positive
        price_df["electricity_2"] = elec_price_positive

    # fill the other columns of the file
    price_df["region"] = country
    price_df["year"] = year
    price_df["unit"] = "cent/Wh"
    price_df["id_hour"] = np.arange(1, 8761)
    # feed in tariff for all eu countries are 10cent/kWh to make comparison easier
    price_df["electricity_feed_in_1"] = 0  # cent/Wh
    price_df["gases_1"] = 0  # will not be used

    return price_df


def filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["country"].isin(get_european_countries_dict().values())].reset_index(drop=True)


def load_european_population_df(source_path: Path) -> pd.DataFrame:
    population = pd.read_excel(
        source_path / Path(r"Europe_population_2020.xlsx"),
        sheet_name="Sheet 1",
        engine="openpyxl",
        skiprows=8,
    ).drop(
        columns=["Unnamed: 2"]
    )
    population.columns = ["country", "population"]
    population_df = filter_countries(population)
    return population_df


def load_forecast_appliance_consumption_change(source_path: Path) -> pd.DataFrame:
    forecast = pd.read_excel(
        io=source_path / Path(r"FORECAST_Appliances_consumption.xlsx"),
        engine="openpyxl",
        sheet_name=0,
    )
    return forecast


def load_european_consumption_df(source_path: Path) -> pd.DataFrame:
    consumption = pd.read_excel(
        source_path / Path(r"Europe_residential_energy_consumption_2020.xlsx"),
        sheet_name="Sheet 1",
        engine="openpyxl",
        skiprows=9,
    )
    consumption.columns = ["country", "type", "consumption (TJ)"]
    consumption["type"] = consumption["type"].str.replace(
        "Final consumption - other sectors - households - energy use - ", ""
    ).replace(
        "Final consumption - other sectors - households - energy use", "total"
    )
    consumption_df = filter_countries(consumption)
    # replace : with 0
    consumption_df["consumption (TJ)"] = consumption_df["consumption (TJ)"].apply(
        lambda x: float(str(x).replace(":", "0")))
    # convert terra joule in kW
    consumption_df["consumption (GWh)"] = consumption_df["consumption (TJ)"].astype(float) / 3_600 * 1_000

    # drop tJ column
    consumption_df = consumption_df.drop(columns=["consumption (TJ)"])
    return consumption_df


def specific_DHW_per_person_EU(source_path: Path) -> pd.DataFrame:
    consumption = load_european_consumption_df(source_path).query("type == 'water heating'")
    population = load_european_population_df(source_path)
    df = pd.merge(consumption, population, on="country")
    df["consumption (Wh)"] = df["consumption (GWh)"] / df["population"] * 1_000 * 1_000 * 1_000
    return df


def appliance_electricity_demand_per_person_EU(year: int, source_path: Path) -> pd.DataFrame:
    consumption = load_european_consumption_df(source_path).query("type == 'lighting and electrical appliances'")
    population = load_european_population_df(source_path)
    df = pd.merge(consumption, population, on="country")
    forecast = load_forecast_appliance_consumption_change(source_path)
    merged = pd.merge(df, forecast.rename(columns={"Country": "country"}), on="country")
    merged["consumption (Wh)"] = merged["consumption (GWh)"] / merged["population"] * 1_000 * 1_000 * 1_000 * merged[str(year)]
    return merged


def create_behavior_component_file(country: str, year: int, source_path: Path) -> pd.DataFrame:
    # copy behavior from main folder
    original_file = Path(__file__).parent / "data" / "input_operation" / "OperationScenario_Component_Behavior.xlsx"
    df = pd.read_excel(io=original_file, engine="openpyxl", sheet_name=0)
    dhw_per_person = specific_DHW_per_person_EU(source_path).query(
        f"country == '{get_european_countries_dict()[country]}'"
    )["consumption (Wh)"]
    electricity_per_person = appliance_electricity_demand_per_person_EU(year, source_path).query(
        f"country == '{get_european_countries_dict()[country]}'"
    )["consumption (Wh)"]
    "FORECAST_Appliances_consumption.xlsx"
    df["hot_water_demand_annual"] = float(dhw_per_person)
    df["appliance_electricity_demand_annual"] = float(electricity_per_person)
    return df


def load_reduced_building_parquet_file(year: int, country: str, source_path: Path) -> pd.DataFrame:

    project_name = f"{country}_{year}"
    config = get_config(project_name=project_name)
    invert_file = config.input / f"INVERT_{country}_{year}.csv"
    df = pd.read_csv(invert_file, sep=",")
    int_columns = ["construction_period_start", "construction_period_end", "building_categories_index",
                   "number_of_floors"]
    df[int_columns] = df[int_columns].astype(int)
    # replace nan with 0:
    df = df.fillna(0)

    # round the number of buildings
    df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]] = df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]].astype(float).round()
    # we are ignoring PV installations in this analysis, drop the PV columns:
    pv_columns = [c for c in df.columns if "PV_number" in c]
    df.drop(columns=pv_columns, inplace=True)
    # filter out buildings that have no hp installations
    df_hp = df.loc[(df["number_buildings_heat_pump_air"] != 0) &  (df["number_buildings_heat_pump_ground"] != 0), :].copy()  # filter out buildings without electric heating

    # there are a lot of building archetypes with less than 10 heat pumps, more than half of of all:
    df_low = df_hp[(df_hp["number_buildings_heat_pump_air"]<=10) & (df_hp["number_buildings_heat_pump_ground"]<=10)]
    
    df_reduced = reduce_number_of_buildings(df_hp)  # reduce the number of buildings by merging similar ones



    # set the number of 0 m2 PV to the number of heating systems minus the other PV installations
    pv_columns = [name for name in df_reduced.columns if "PV" in name if not "number_of_0_m2" in name]
    df_reduced["PV_number_of_0_m2"] = df_reduced["number_buildings_heat_pump_air"] + \
                                      df_reduced["number_buildings_heat_pump_ground"] + \
                                      df_reduced["number_buildings_electricity"] - df_reduced[pv_columns].sum(axis=1)
    return df_reduced


def create_input_excels(year: int,
                        country: str,
                        project_prefix: str,
                        source_path: Path,
                        force_copy: bool = False):
    df_reduced = load_reduced_building_parquet_file(year, country, source_path)
    # tables:
    operation_scenario_component_pv = create_pv_id_table(df_reduced)
    operation_scenario_building = create_building_id_table(df_reduced)
    region_scenario_table = create_operation_region_file(year, country)
    weather_scenario_table = create_operation_weather_file(country)
    energy_price_scenario_table = create_energy_price_scenario_file(country, year, source_path)
    operation_behavior_component = create_behavior_component_file(country, year, source_path)

    # save the operation_scenario_component tables as excel:
    input_operation_path = Path(__file__).parent / "data" / "input_operation" / f"{project_prefix}_{country}_{year}"
    # create the folders:
    create_dict_if_not_exists(input_operation_path)
    # save the excel files:
    operation_scenario_component_pv.to_excel(input_operation_path / "OperationScenario_Component_PV.xlsx", index=False)
    operation_scenario_building.to_excel(input_operation_path / "OperationScenario_Component_Building.xlsx",
                                         index=False)
    region_scenario_table.to_excel(input_operation_path / "OperationScenario_Component_Region.xlsx", index=False)
    weather_scenario_table.to_excel(input_operation_path / "OperationScenario_RegionWeather.xlsx", index=False)
    energy_price_scenario_table.to_excel(input_operation_path / "OperationScenario_EnergyPrice.xlsx", index=False)
    operation_behavior_component.to_excel(input_operation_path / "OperationScenario_Component_Behavior.xlsx",
                                          index=False)

    # copy the other input excel files to each country from the input_operation folder:
    copy_input_operation_files(destination=input_operation_path, force_copy=force_copy)

    # create config class with different project names, for country and year:
    cfg = Config(project_name=f"{project_prefix}_{country}_{year}")
    # create a database (sqlite file) for every country and year:
    ProjectDatabaseInit(cfg).load_operation_component_tables()
    ProjectDatabaseInit(cfg).load_operation_profile_tables()


def map_heat_pump_to_scenario_table(scenario_df: pd.DataFrame, large_df: pd.DataFrame, test_numbers: dict):
    # ID 1 == Air_HP, ID 2 == Ground_HP
    scenario_df["ID_Boiler"] = 0
    # we dont consider buildings with heat pumps if there are less than 0 heat pumps in the stock
    id_air = large_df.query("number_buildings_heat_pump_air > 0")["index"].to_list()
    id_ground = large_df.query("number_buildings_heat_pump_ground > 0")["index"].to_list()
    id_elec = large_df.query("number_buildings_electricity > 0")["index"].to_list()

    # compare the masks to see which IDs will be doubled:
    scenario_df.loc[scenario_df["ID_Building"].isin(id_air), "ID_Boiler"] = 1
    scenario_df.loc[scenario_df["ID_Building"].isin(id_ground), "ID_Boiler"] = 2
    scenario_df.loc[scenario_df["ID_Building"].isin(id_elec), "ID_Boiler"] = 3

    # test numbers:
    assert test_numbers["air_numbers"] == round(scenario_df.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(scenario_df.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(scenario_df.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
        "number of buildings dont add up"
    return scenario_df


def add_tank_to_scenario_df(scenario_df: pd.DataFrame,
                            tank_table_name: str,
                            building_types: list,
                            building_table: pd.DataFrame,
                            cfg: Config,
                            test_numbers: dict) -> pd.DataFrame:
    # load the heating tank table
    tank_table = DatabaseInitializer(cfg).db.read_dataframe(tank_table_name)
    # get the names of the columns for ID and building type from tank table
    name_id = [name for name in list(tank_table.columns) if "ID" in name][0]
    name_building_type = [name for name in list(tank_table.columns) if "type" in name][0]
    if name_building_type == None or len(name_building_type) == 0:
        logger.warning("No building type defined in the tank excel.")

    tables_for_concat = []
    # iterate through tank IDs and create a copy of each scenario table with the new tank ID added:
    for i, row in tank_table.iterrows():
        tank_id = row[name_id]
        building_type_to_merge_to = row[name_building_type]
        technology_penetration = row["percentage_technology_penetration"]
        new_table = scenario_df.copy()
        if building_type_to_merge_to in building_types:
            ids_to_merge = building_table.loc[building_table.loc[:, "type"] == building_type_to_merge_to, "ID_Building"]
            new_table = new_table.loc[new_table.loc[:, "ID_Building"].isin(ids_to_merge)]
            new_table.loc[:, name_id] = tank_id
            new_table["number_of_buildings"] = new_table["number_of_buildings"] * technology_penetration
            tables_for_concat.append(new_table)
            del new_table
        else:  # type is not in table so tank will be used for all buildings:
            new_table.loc[:, name_id] = tank_id
            new_table["number_of_buildings"] = new_table["number_of_buildings"] * technology_penetration
            tables_for_concat.append(new_table)
            del new_table
    new_scenario_table = pd.concat(tables_for_concat, axis=0).reset_index(drop=True)
    assert test_numbers["air_numbers"] == round(new_scenario_table.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(new_scenario_table.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(new_scenario_table.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
           "building numbers dont add up"
    return new_scenario_table


def tanks_to_scenario_table_per_building_type(scenario_df: pd.DataFrame, cfg: "Config",
                                              test_numbers: dict) -> pd.DataFrame:
    # load the building table to get the building ID with corresponding building type:
    building_table = DatabaseInitializer(cfg).db.read_dataframe(OperationScenarioComponent.Building.table_name)
    table_to_map = building_table.loc[:, ["ID_Building", "type"]]
    building_types = list(table_to_map.loc[:, "type"].unique())
    # add both tanks after each other:
    scenario_table_with_dhw_tank = add_tank_to_scenario_df(scenario_df=scenario_df,
                                                           tank_table_name="OperationScenario_Component_HotWaterTank",
                                                           building_types=building_types,
                                                           building_table=building_table,
                                                           cfg=cfg,
                                                           test_numbers=test_numbers)

    scenario_table_with_both_tanks = add_tank_to_scenario_df(scenario_df=scenario_table_with_dhw_tank,
                                                             tank_table_name="OperationScenario_Component_SpaceHeatingTank",
                                                             building_types=building_types,
                                                             building_table=building_table,
                                                             cfg=cfg,
                                                             test_numbers=test_numbers)
    # check numbers:
    assert test_numbers["air_numbers"] == round(scenario_table_with_both_tanks.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(scenario_table_with_both_tanks.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(scenario_table_with_both_tanks.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
           "building numbers dont add up"
    # buildings with electric heating do not have a heating buffer storage, ID Boiler = 3 for electric heating
    id_boiler_3 = scenario_table_with_both_tanks.query("ID_Boiler == 3")
    id_boiler_3_keep = id_boiler_3.query(f"ID_SpaceHeatingTank in {[1]}")
    id_boiler_3_delete = id_boiler_3.query(f"ID_SpaceHeatingTank in {[2, 3]}")
    # delete each row and add the number of buildings to the respective scenario without space heating tank
    id_space_heating_tank = 1  # this id is fixed
    id_to_gather = ["ID_Building", "ID_PV", "ID_Battery", "ID_HotWaterTank"]
    for i, row in id_boiler_3_delete.iterrows():
        ids = row[id_to_gather].to_dict()
        number_of_buildings = row["number_of_buildings"]
        # add the number to  the keep df:
        id_boiler_3_keep.loc[
            (id_boiler_3_keep[list(ids)] == pd.Series(ids)).all(axis=1), "number_of_buildings"] += number_of_buildings

    # now delete the id_boiler_3  from the scenarios and add the id_boiler keep
    return_df = pd.concat([scenario_table_with_both_tanks.query("ID_Boiler != 3"), id_boiler_3_keep]).reset_index(
        drop=True)
    # check numbers:
    assert test_numbers["air_numbers"] == round(return_df.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(return_df.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(return_df.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
           "building numbers dont add up"
    return return_df


def drop_scenarios_with_low_number_of_buildings(min_number_buildings: int,
                                                scenario_table: pd.DataFrame,
                                                ) -> pd.DataFrame:
    # now we have a lot of scenarios where some scenarios do not even represent a whole building (0.5 buildings etc)
    new_table = scenario_table[scenario_table['number_of_buildings'] > min_number_buildings].reset_index(drop=True)
    logger.warning(
        f"{round(scenario_table[scenario_table['number_of_buildings'] < min_number_buildings]['number_of_buildings'].sum())} "
        f"out of {round(scenario_table['number_of_buildings'].sum())}"
        f" Buildings are ignored and \n"
        f"{round(scenario_table[scenario_table['number_of_buildings'] < min_number_buildings].query('ID_Battery != 1')['number_of_buildings'].sum())} "
        f"out of {round(scenario_table.query('ID_Battery != 1')['number_of_buildings'].sum())} "
        f"Batteries are ignored"
    )
    new_table.loc[:, "ID_Scenario"] = new_table.index + 1
    return new_table


def map_battery_to_scenario_table(scenario_df, cfg: Config, year: int, test_numbers: dict):
    db = DB(config=cfg)
    battery_df = db.read_dataframe(OperationScenarioComponent.Battery.table_name)
    # buildings with PV are distinguished in buildings with Battery and without
    # buildings without PV are not added
    no_pv_buildings = scenario_df.query("ID_PV == 1").copy()
    no_pv_buildings["ID_Battery"] = 1
    pv_buildings = scenario_df.query("ID_PV != 1").copy()
    pv_building_list = []
    for battery_id in battery_df["ID_Battery"]:
        part_table = pv_buildings.copy()
        part_table["ID_Battery"] = battery_id
        if battery_id == 1:
            technology_penetration = 1 - get_battery_penetration()[year]
            part_table["number_of_buildings"] = part_table["number_of_buildings"] * technology_penetration
        else:
            technology_penetration = get_battery_penetration()[year]
            part_table["number_of_buildings"] = part_table["number_of_buildings"] * technology_penetration
        pv_building_list.append(part_table)
    new_scenario = pd.concat([no_pv_buildings] + pv_building_list)
    # check numbers
    assert test_numbers["air_numbers"] == round(new_scenario.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(
        new_scenario.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(new_scenario.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
        "building numbers dont add up"
    return new_scenario.reset_index(drop=True)


def create_scenario_tables(country: str,
                           year: int,
                           minimum_number_buildings: int,
                           project_prefix: str,
                           source_path: Path,
                           ) -> None:
    logger.info(f"clearing the log files for {country} {year}.")
    # create config class with different project names, for country and year:
    cfg = Config(project_name=f"{project_prefix}_{country}_{year}")
    clear_result_files(country, year, project_prefix=project_prefix, source_path=source_path)
    ProjectDatabaseInit(cfg).drop_tables()  # drop result tables if they exist
    # start creating the scenario tables
    logger.info(f"Creating Scenario Table for {country} {year}.")

    # load the parquet file:
    df_reduced = load_reduced_building_parquet_file(year, country, source_path)
    # create numbers for tests:
    test_numbers = {
        "air_numbers": round(df_reduced["number_buildings_heat_pump_air"].sum()),
        "ground_numbers": round(df_reduced["number_buildings_heat_pump_ground"].sum()),
        "elec_numbers": round(df_reduced["number_buildings_electricity"].sum()),
    }

    operation_scenario_component_pv = create_pv_id_table(df_reduced)
    building_pv_table = map_pv_to_building_id(operation_scenario_component_pv, df_reduced)

    building_pv_hp_table = map_heat_pump_to_scenario_table(building_pv_table, df_reduced, test_numbers)
    battery_building_pv_hp_table = map_battery_to_scenario_table(building_pv_hp_table, cfg, year, test_numbers)
    scenario_table = tanks_to_scenario_table_per_building_type(scenario_df=battery_building_pv_hp_table,
                                                               cfg=cfg, test_numbers=test_numbers)

    engine = DatabaseInitializer(config=cfg).db.get_engine().connect()
    for key, value in OperationScenarioComponent.__dict__.items():
        if key == "Building" or key == "PV" or key == "Boiler" or key == "SpaceHeatingTank" or key == "HotWaterTank" or key == "Battery":
            continue
        else:
            if isinstance(value, OperationComponentInfo):
                # add the scenarios to the scenario table
                # load the IDs from the database tables
                if engine.dialect.has_table(engine, value.table_name):
                    new_ids = DatabaseInitializer(cfg).db.read_dataframe(value.table_name)[value.id_name].unique()
                    # create a list of tables that will be merged below each other with the new IDs attached as column:
                    part_tables = [pd.concat([scenario_table, pd.Series(index=range(len(scenario_table)),
                                                                        name=value.id_name,
                                                                        data=np.full(shape=(len(scenario_table)),
                                                                                     fill_value=id))],
                                             axis=1) for id in new_ids]
                    # concat the tables from the list into the new scenario table:
                    scenario_table = pd.concat(part_tables, axis=0).reset_index(drop=True)

    # add a scenario ID to the table:
    scenario_df = scenario_table.reset_index().rename(columns={"index": "ID_Scenario"})
    scenario_df.loc[:, "ID_Scenario"] = scenario_df.index + 1
    # check numbers:
    assert test_numbers["air_numbers"] == round(scenario_df.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(scenario_df.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           test_numbers["elec_numbers"] == round(scenario_df.query("ID_Boiler == 3")["number_of_buildings"].sum()), \
           "building numbers dont add up"

    # drop the scenarios where there are not more than 5 buildings:
    new_scenario_table_with_numbers = drop_scenarios_with_low_number_of_buildings(
        min_number_buildings=minimum_number_buildings,
        scenario_table=scenario_df,
    )

    # drop numbers for scenario table in sqlite
    new_scenario_table_sqlite = new_scenario_table_with_numbers.drop(columns=["number_of_buildings"])
    # numbers df:
    numbers_df = new_scenario_table_with_numbers.loc[:, ["ID_Scenario", "number_of_buildings"]]

    # save the scenario table to the sqlite database:
    data_types = {name: sqlalchemy.types.Integer for name in new_scenario_table_sqlite.columns}
    ProjectDatabaseInit(cfg).db.write_dataframe(
        table_name=OperationTable.Scenarios,
        data_frame=new_scenario_table_sqlite,
        data_types=data_types,
        if_exists="replace"
    )
    # save numbers df to sqlite database:
    ProjectDatabaseInit(cfg).db.write_dataframe(
        table_name="ScenarioNumbers",
        data_frame=numbers_df,
        if_exists="replace"
    )


def clear_result_files(country: str, year: int, project_prefix: str, source_path: Path):
    """
    deletes the content of the result folder
    Returns: None

    """
    path = source_path.parent.parent / "data" / "output" / f"{project_prefix}_{country}_{year}"
    extensions = ["*.log", "*.csv", "*.xlsx", "*.parquet.gzip"]
    for ext in extensions:
        files = path.glob(ext)
        for file in files:
            file.unlink()


def main(country_list: list, years: list, minimum_number_buildings: int, project_prefix: str, source_path: Path):

    # --------------------------------------------------------------------
    # # FOR DEBUGGING:
    # country = "CZE"
    # create_input_excels(2020,
    #                     country=country,
    #                     weather_df=original_df,
    #                     force_copy=True)
    #
    # create_scenario_tables(country=country,
    #                        year=2020,
    #                        minimum_number_buildings=minimum_number_buildings)
    # --------------------------------------------------------------------
    # for country in country_list:
    #     for year in years:
    #         create_input_excels(year=year, country=country, weather_df=original_df, force_copy=True)
    #         create_scenario_tables(country=country, year=year,
    #                                minimum_number_buildings=minimum_number_buildings)

    answer = input(f"Delete the results of all scenarios? (y/n)")
    force_copy = True
    if answer.lower() == "y":
        logger.info("Dropping old results and creating new input...")
        # use multiprocessing to speed it up creating all the input data:
        input_list = [(year, country, project_prefix, source_path, force_copy)
                      for year in years for country in country_list]
        number_of_physical_cores = int(multiprocessing.cpu_count() / 2)
        Parallel(n_jobs=number_of_physical_cores)(delayed(create_input_excels)(*inst) for inst in input_list)

        # with multiprocessing.Pool(number_of_physical_cores) as pool:
        #     pool.starmap(create_input_excels, input_list)

         # create the individual scenario tables:
        scenario_tables_pool_list = [(country, year, minimum_number_buildings, project_prefix, source_path)
                                     for year in years for country in country_list]
        Parallel(n_jobs=number_of_physical_cores)(delayed(create_scenario_tables)(*inst) for inst in scenario_tables_pool_list)

        # with multiprocessing.Pool(number_of_physical_cores) as pool:
        #     pool.starmap(create_scenario_tables, scenario_tables_pool_list)


    # else:
    #     logger.info("stopped creating individual scenarios.")


if __name__ == "__main__":
    country_list = [
        # 'BEL',
        'AUT',
        # 'HRV',
        # "CYP",
        # 'CZE',
        # 'DNK',
        # 'FRA',
        # 'DEU',
        # 'LUX',
        # 'HUN',
        # 'IRL',
        # 'ITA',
        # 'NLD',
        # 'POL',
        # 'PRT',
        # 'SVK',
        # 'SVN',
        # 'ESP',
        # 'SWE',
        # 'GRC',
        # 'LVA',
        # 'LTU',
        # 'MLT',
        # 'ROU',
        # 'BGR',
        # 'FIN',
        # 'EST',
    ]
    # country_list = [ 'ROU',]#'MLT','LTU', 'GRC']
    years = [2020, 2030, 2040, 2050]
    project_prefix = f""
    source_path = Path(__file__).parent / "projects" / f""
    minimum_number_buildings = 1  # if a scenario has less buildings than that, it will be ignored
    main(country_list, years, minimum_number_buildings, project_prefix, source_path)

