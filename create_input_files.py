import pandas as pd
import numpy as np
from pathlib import Path
import logging
from utils.db import create_db_conn
from projects.main import get_config, Config

from models.operation import components
import multiprocessing
import shutil
import sqlalchemy
from joblib import Parallel, delayed


INPUT_FOLDER = Path().cwd() / "projects" / "AURESII_DATA"


# Ensure the log directory exists
log_file_path = Path(__file__).parent / "create_inputs_logfile.log"
# delete old logfile:
if log_file_path.exists():
    log_file_path.unlink()
log_level = logging.INFO
# Configure the logger
logging.basicConfig(
    level=log_level,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Write logs to the file
        logging.StreamHandler()             # Optionally output logs to the console
    ]
)
LOGGER = logging.getLogger(__name__)


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

def appliance_electricity_demand_per_person_EU() -> pd.DataFrame:
    consumption = load_european_consumption_df().query("type == 'lighting and electrical appliances'")
    population = load_european_population_df()
    df = pd.merge(consumption, population, on="country")
    forecast = load_forecast_appliance_consumption_change()
    merged = pd.merge(df, forecast.rename(columns={"Country": "country"}), on="country")
    merged[["consumption 2020 (Wh)", "consumption 2030 (Wh)", "consumption 2040 (Wh)", "consumption 2050 (Wh)", ]] = merged[["2020", "2030", "2040", "2050"]].mul(merged["consumption (GWh)"] / merged["population"]*1_000 * 1_000 * 1_000, axis=0)
    merged.drop(columns=["consumption (GWh)", "population", "2020", "2030", "2040", "2050"], inplace=True)
    return merged


def specific_DHW_per_person_EU() -> pd.DataFrame:
    consumption = load_european_consumption_df().query("type == 'water heating'")
    population = load_european_population_df()
    df = pd.merge(consumption, population, on="country")
    df["consumption (Wh)"] = df["consumption (GWh)"] / df["population"] * 1_000 * 1_000 * 1_000
    df.drop(columns=["consumption (GWh)", "population"], inplace=True)
    return df

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


def create_building_id_table(df: pd.DataFrame, year: int, country: str) -> pd.DataFrame:
    """ creates the operationScenario_component_building dataframe that can be the excel input"""
    columns = list(components.Building.__dataclass_fields__) + [
        "ID_Building",
        "number_buildings_heat_pump_air",
        "number_buildings_heat_pump_ground",
        "number_of_buildings",
        "ID_HotWaterTank",
        "ID_SpaceHeatingTank"
    ]
    building_table = pd.DataFrame(columns=columns, index=range(len(df)))
    # insert building ID
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

    appliance_demand = APPLICANCE_DEMAND_PER_PERSON.loc[APPLICANCE_DEMAND_PER_PERSON["country"]==get_european_countries_dict()[country], f"consumption {year} (Wh)"].values[0]
    building_table.loc[:, "appliance_electricity_demand_per_person"] = appliance_demand

    hot_water_demand = DHW_DEMAND_PER_PERSON.loc[DHW_DEMAND_PER_PERSON["country"]==get_european_countries_dict()[country], f"consumption (Wh)"].values[0]
    building_table.loc[:, "hot_water_demand_per_person"] = hot_water_demand


    building_table.drop(columns="id_demand_profile_type", inplace=True)
    return building_table


def map_pv_to_building_id(component_pv: pd.DataFrame,
                          df: pd.DataFrame) -> pd.DataFrame:    
    # the PV IDs are added after the permutations to the scenario table:
    column_names = [name for name in df.columns.to_list() if "PV" in name]
    scenario = df[["ID_Building"] + column_names].fillna(0)
    new_column_names = ["ID_Building"] + [
        str(int(component_pv.loc[component_pv.loc[:, "size"] == pv_area_to_kWp(name), "ID_PV"].iloc[0])) for name in column_names
    ]
    scenario.columns = new_column_names
    scenario_df = scenario.melt(id_vars="ID_Building", var_name="ID_PV", value_name="number_of_buildings")
    # drop the zeros:
    scenario_df = scenario_df[scenario_df["number_of_buildings"] > 0]

    # check numbers:
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
        air_hp["number_buildings_heat_pump_ground"] = 0
        ground_hp["number_buildings_heat_pump_air"] = 0

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
    return_df["ID_Building"] = return_df.index + 1
    # set the building category index as int
    return_df["building_categories_index"] = return_df["building_categories_index"].apply(lambda x: int(round(x))).astype(int)
    # check:
    assert round(return_df["number_buildings_heat_pump_air"].sum()) == round(air_hp_numbers) and \
           round(return_df["number_buildings_heat_pump_ground"].sum()) == round(ground_hp_numbers) and \
           f"the number of buildings is altered"

    return return_df


def create_operation_region_file(year: int, country: str) -> pd.DataFrame:
    columns = ["ID_Region", "code", "year"]
    df = pd.DataFrame(columns=columns, data=[[1, get_country_correction_code()[country], year]])
    return df


def create_energy_price_scenario_file(country: str, year: int) -> pd.DataFrame:
    columns = ["region", "year", "id_hour", "electricity_1", "electricity_2", "electricity_feed_in_1", "unit"]
    price_df = pd.DataFrame(columns=columns)
    grid_fees = 0.02 # cent/Wh

    if year > 2020:
        # import electricity prices from AURES: shiny happy corresponds to newTrends and we add for elec 2 a fixed tarif
        df_newTrends = pd.read_excel(
            INPUT_FOLDER / Path("elec_prices_AURES.xlsx"),
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

        # save with additional  fixed tariff as price 2:
       
        elec_price_2 = elec_price_1 + grid_fees
        price_df["electricity_2"] = elec_price_2

    else:  # prices are taken from ENTSO-E from 2019
        df = pd.read_csv(  # price is in cent/kWh
            INPUT_FOLDER / Path(r"ENTSO-E_prices_europe_2019_newTrends.csv"),
            sep=",",
            index_col=False
        )
        # ffill because sometimes a few values in the data are missing
        elec_price = df[get_country_correction_code()[country]].ffill().astype(float).to_numpy()
        # make sure that there are no negative prices by increasing the costs so that the minimum price will be at 20
        # cent/kWh:
        # min_price = abs(np.floor(elec_price.min()))
        elec_price_positive = (elec_price) / 1_000  # cent/kWh->cent/Wh
        # for 2020
        price_df["electricity_1"] = elec_price_positive
        price_df["electricity_2"] = elec_price_positive + grid_fees

    # fill the other columns of the file
    price_df["region"] = country
    price_df["year"] = year
    price_df["unit"] = "cent/Wh"
    price_df["id_hour"] = np.arange(1, 8761)
    # feed in tariff for all eu countries are 10cent/kWh to make comparison easier
    price_df["electricity_feed_in_1"] = 0  # cent/Wh
    price_df["gases_1"] = 0  # will not be used

    return price_df

def create_energy_price_id_file():
    df = pd.DataFrame(
        columns=["ID_EnergyPrice", "id_electricity", "id_electricity_feed_in", "price_unit"],
        data=[
            [1, 1, 1, "cent/Wh"], 
            [2, 2, 1, "cent/Wh"], 
        ]
    )
    return df


def filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["country"].isin(get_european_countries_dict().values())].reset_index(drop=True)


def load_european_population_df() -> pd.DataFrame:
    population = pd.read_excel(
        INPUT_FOLDER / Path(r"Europe_population_2020.xlsx"),
        sheet_name="Sheet 1",
        engine="openpyxl",
        skiprows=8,
    ).drop(
        columns=["Unnamed: 2"]
    )
    population.columns = ["country", "population"]
    population_df = filter_countries(population)
    return population_df


def load_forecast_appliance_consumption_change() -> pd.DataFrame:
    forecast = pd.read_excel(
        io=INPUT_FOLDER / Path(r"FORECAST_Appliances_consumption.xlsx"),
        engine="openpyxl",
        sheet_name=0,
    )
    return forecast


def load_european_consumption_df() -> pd.DataFrame:
    consumption = pd.read_excel(
        INPUT_FOLDER / Path(r"Europe_residential_energy_consumption_2020.xlsx"),
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


def load_buildings(year: int, country: str, cfg: Config) -> pd.DataFrame:

    invert_file = cfg.input / f"INVERT_{country}_{year}.csv"
    df = pd.read_csv(invert_file, sep=",")
    int_columns = [
        "construction_period_start", 
        "construction_period_end", 
        "building_categories_index",
        "number_of_floors"
    ]
    df[int_columns] = df[int_columns].astype(int)
    # replace nan with 0:
    df = df.fillna(0)

    # load old file:
    old_file = cfg.input / "OperationScenario_Component_Building_old.csv"
    df_old = pd.read_csv(old_file, sep=",")
    df_old = df_old.query("id_demand_profile_type == 1").reset_index(drop=True)
    df_old.Af = df_old.Af.round(5)
    
    # get the correct decimal person num from old df
    df = pd.merge(
        left=df, 
        right=df_old[ [ "construction_period_start", "construction_period_end", "Af", "person_num"] ], 
        on= [ "construction_period_start", "construction_period_end", "Af", ]
    )
    

    # round the number of buildings
    df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]] = df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]].astype(float).round()

    # filter out buildings that have no hp installations
    df_hp = df.loc[(df["number_buildings_heat_pump_air"] != 0) &  (df["number_buildings_heat_pump_ground"] != 0), :].copy()  # filter out buildings without electric heating

    df_reduced = reduce_number_of_buildings(df_hp).drop(columns="index")  # reduce the number of buildings by merging similar ones

    # set the number of 0 m2 PV to the number of heating systems minus the other PV installations
    pv_columns = [name for name in df_reduced.columns if "PV" in name if not "number_of_0_m2" in name]
    df_reduced[pv_columns] = df_reduced[pv_columns].clip(lower=0)
    df_reduced["PV_number_of_0_m2"] = df_reduced["number_buildings_heat_pump_air"] + \
                                      df_reduced["number_buildings_heat_pump_ground"] - \
                                      df_reduced[pv_columns].sum(axis=1)
    df_reduced[df_reduced["PV_number_of_0_m2"]<0] = 0
    return df_reduced


def create_boiler_table(year: int):
    efficiencies = {
        2020: {"air": 0.35, "ground": 0.4},
        2030: {"air": 0.37, "ground": 0.42},
        2040: {"air": 0.39, "ground": 0.44},
        2050: {"air": 0.41, "ground": 0.43},
    }
    df = pd.DataFrame(
        columns=["ID_Boiler", "type", "power_max",  "power_max_unit", "carnot_efficiency_factor", "fuel_boiler_efficiency"],
        data=[
            [1, "Air_HP", 1, "W", efficiencies[year]["air"], 0.9], 
            [2, "Ground_HP", 1, "W", efficiencies[year]["ground"], 0.9], 
        ]
    )
    return df

def create_dhw_table(building_table: pd.DataFrame):
    # DHW Tank: 100 L per person
    liter_per_person = 100
    building_table["DHW_size"] = (building_table["person_num"] * liter_per_person).astype(float).round()
    unique_sizes = building_table["DHW_size"].unique()
    dhw_sizes = {i+1: unique_sizes[i] for i in range(len(unique_sizes))}

    df = pd.DataFrame(
        columns=["ID_HotWaterTank", "size", "size_unit", "loss",  "loss_unit", "temperature_start", "temperature_max", "temperature_min", "temperature_surrounding", "temperature_unit", "note"],
        data=[ [i+1, dhw_sizes[i+1], "L", 0.2, "W/m2K", 20, 55, 20, 20, "°C", "SFH"] for i in range(len(unique_sizes)) ]
    )

    building_table["ID_HotWaterTank"] = building_table["DHW_size"].map({value: key for key, value in dhw_sizes.items()})
    return df, building_table

def create_heating_tank_table():
    # buffer tank: is calculated within the model with 30L per kW thermal power output
    df = pd.DataFrame(
        columns=["ID_SpaceHeatingTank", "size", "size_unit", "loss",  "loss_unit", "temperature_start", "temperature_max", "temperature_min", "temperature_surrounding", "temperature_unit", "note"],
        data=[[1, 1, "L", 0.2, "W/m2K", 20, 45, 20, 20, "°C", "SFH"]]
    )
    return df

def create_cooling_table():
    df = pd.DataFrame(
        columns=["ID_SpaceCoolingTechnology", "efficiency", "power", "power_unit"],
        data=[ 
            [1, 4, 0, "W"],
            [2, 4, 1, "W"],
        ]
    )
    return df

def create_EV_table():
    df = pd.DataFrame(
        columns=["ID_Vehicle", "type", "capacity"],  # rest is not needed because we dont use it
        data=[ 
            [1, "electric", 0,],
        ]
    )
    return df

def create_battery_table():
    df = pd.DataFrame(
        columns=["ID_Battery", "capacity", "capacity_unit", "charge_efficiency", "charge_power_max", "charge_power_max_unit", "discharge_efficiency", "discharge_power_max", "discharge_power_max_unit"],
        data=[ 
            [1, 0, "Wh", 0.95, 4500, "W", 0.95, 4500, "W",],
        ]
    )
    return df

def create_heating_element_table():
    df = pd.DataFrame(
        columns=["ID_HeatingElement", "power", "power_unit", "efficiency"],
        data=[ 
            [1, 0, "W", 1],
        ]
    )
    return df


def change_name_of_old_building_file(cfg: Config):
    # do this only once so the real old file is not overwritten:
    file = cfg.input / "OperationScenario_Component_Building.csv"
    old_file = cfg.input / "OperationScenario_Component_Building_old.csv"
    if not old_file.exists():
        file.rename(cfg.input / "OperationScenario_Component_Building_old.csv")

def create_behavior_table(cfg: Config):
    # TODO copy the behavior profiles from the "testbed" and then check if the total demand is 
    # correctly taken from the "building table", same with appliance demand
    path_2_testbed_behavior = Path(r"C:\Users\mascherbauer\OneDrive\PycharmProjects\FLEX\projects\Test_bed\input") / "OperationScenario_BehaviorProfile.xlsx"
    destination = cfg.input / "OperationScenario_BehaviorProfile.xlsx"
    # copy file
    shutil.copy(src=path_2_testbed_behavior, dst=destination)



def create_input_excels(year: int,
                        country: str,
                    ):
    
    project_name = f"{country}_{year}"
    config = get_config(project_name=project_name)
    change_name_of_old_building_file(cfg=config)

    df_buildings = load_buildings(year, country, cfg=config)
    
    # weather table is already correct
    # region is already correct and not needed

    # tables:
    operation_scenario_component_pv = create_pv_id_table(df_buildings)
    region_scenario_table = create_operation_region_file(year, country)
    energy_price_table = create_energy_price_scenario_file(country, year)
    energy_price_scenario_table = create_energy_price_id_file()
    boiler_table = create_boiler_table(year=year)
    dhw_table, df_buildings = create_dhw_table(df_buildings)
    heating_tank_table = create_heating_tank_table()
    operation_scenario_building = create_building_id_table(df_buildings, year=year, country=country)
    cooling_table = create_cooling_table()
    battery_table = create_battery_table()
    ev_table = create_EV_table()
    heating_element_table = create_heating_element_table()
    create_behavior_table(cfg=config) # copys the input file from testbed because the other one is for a single house

    # save the csv files:
    operation_scenario_component_pv.to_csv(config.input / "OperationScenario_Component_PV.csv", index=False, sep=";")
    operation_scenario_building.to_csv(config.input / "OperationScenario_Component_Building.csv", index=False, sep=";")
    region_scenario_table.to_csv(config.input / "OperationScenario_Component_Region.csv", index=False, sep=";")
    energy_price_table.to_csv(config.input / "OperationScenario_EnergyPrice.csv", index=False, sep=";")
    boiler_table.to_csv(config.input / "OperationScenario_Component_Boiler.csv", index=False, sep=";")
    dhw_table.to_csv(config.input / "OperationScenario_Component_HotWaterTank.csv", index=False, sep=";")
    heating_tank_table.to_csv(config.input / "OperationScenario_Component_SpaceHeatingTank.csv", index=False, sep=";")
    cooling_table.to_csv(config.input / "OperationScenario_Component_SpaceCoolingTechnology.csv", index=False, sep=";")
    battery_table.to_csv(config.input / "OperationScenario_Component_Battery.csv", index=False, sep=";")
    ev_table.to_csv(config.input / "OperationScenario_Component_Vehicle.csv", index=False, sep=";")
    heating_element_table.to_csv(config.input / "OperationScenario_Component_HeatingElement.csv", index=False, sep=";")
    energy_price_scenario_table.to_csv(config.input / "OperationScenario_Component_EnergyPrice.csv", sep=";", index=False)
    LOGGER.info(f"created input csvs for {country} {year}")

    # delete the same excel files:
    excel_files = [
        config.input / "OperationScenario_Component_PV.xlsx", 
        config.input / "OperationScenario_Component_Building.xlsx",
        config.input / "OperationScenario_Component_Region.xlsx",
        config.input / "OperationScenario_EnergyPrice.xlsx",
        config.input / "OperationScenario_Component_Boiler.xlsx",
        config.input / "OperationScenario_Component_HotWaterTank.xlsx",
        config.input / "OperationScenario_Component_SpaceHeatingTank.xlsx",
        config.input / "OperationScenario_Component_SpaceCoolingTechnology.xlsx",
        config.input / "OperationScenario_Component_Battery.xlsx",
        config.input / "OperationScenario_Component_HeatingElement.xlsx",
        config.input / "OperationScenario_Component_EnergyPrice.xlsx",
        config.input / "OperationScenario_BehaviorProfile.csv", # only file where the csv (orig) needs to be deleted
    ]
    for file in excel_files:
        if file.exists():
            file.unlink()


def map_heat_pump_to_scenario_table(scenario_df: pd.DataFrame, large_df: pd.DataFrame, test_numbers: dict):
    # ID 1 == Air_HP, ID 2 == Ground_HP
    scenario_df["ID_Boiler"] = 0
    # we dont consider buildings with heat pumps if there are less than 0 heat pumps in the stock
    id_air = large_df.query("number_buildings_heat_pump_air > 0")["ID_Building"].to_list()
    id_ground = large_df.query("number_buildings_heat_pump_ground > 0")["ID_Building"].to_list()

    # compare the masks to see which IDs will be doubled:
    scenario_df.loc[scenario_df["ID_Building"].isin(id_air), "ID_Boiler"] = 1
    scenario_df.loc[scenario_df["ID_Building"].isin(id_ground), "ID_Boiler"] = 2

    # test numbers:
    assert test_numbers["air_numbers"] == round(scenario_df.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(scenario_df.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
        "number of buildings dont add up"
    return scenario_df


def tanks_to_scenario_table_per_building_type(scenario_df: pd.DataFrame, 
                                              cfg: "Config",
                                              test_numbers: dict,
                                              building_table: pd.DataFrame) -> pd.DataFrame:
    building_table.loc[:, "type"] = building_table["building_categories_index"].map(get_sfh_mfh_dict())

    # load tanks 
    _, dhw_building_table = create_dhw_table(building_table)
    to_merge = dhw_building_table[["ID_Building", "ID_HotWaterTank"]]
    scenario_table_with_dhw_tank = pd.merge(left=scenario_df, right=to_merge, on="ID_Building")

    # the buffer tank size is determined in the model and here only 1 ID is given
    scenario_table_with_both_tanks = scenario_table_with_dhw_tank.copy()
    scenario_table_with_both_tanks["ID_SpaceHeatingTank"] = 1

    # check numbers:
    assert test_numbers["air_numbers"] == round(scenario_table_with_both_tanks.query("ID_Boiler == 1")["number_of_buildings"].sum()) and \
           test_numbers["ground_numbers"] == round(scenario_table_with_both_tanks.query("ID_Boiler == 2")["number_of_buildings"].sum()) and \
           "building numbers dont add up"

    return scenario_table_with_both_tanks


def drop_scenarios_with_low_number_of_buildings(min_number_buildings: int,
                                                scenario_table: pd.DataFrame,
                                                ) -> pd.DataFrame:
    # now we have a lot of scenarios where some scenarios do not even represent a whole building (0.5 buildings etc)
    new_table = scenario_table[scenario_table['number_of_buildings'] > min_number_buildings].reset_index(drop=True)
    LOGGER.warning(
        f"{round(scenario_table[scenario_table['number_of_buildings'] < min_number_buildings]['number_of_buildings'].sum())} "
        f"out of {round(scenario_table['number_of_buildings'].sum())}"
        f" Buildings are ignored and \n"
        f"{round(scenario_table[scenario_table['number_of_buildings'] < min_number_buildings].query('ID_Battery != 1')['number_of_buildings'].sum())} "
        f"out of {round(scenario_table.query('ID_Battery != 1')['number_of_buildings'].sum())} "
        f"Batteries are ignored"
    )
    new_table.loc[:, "ID_Scenario"] = new_table.index + 1
    return new_table


def map_battery_to_scenario_table(scenario_df: pd.DataFrame):
    scenario_df["ID_Battery"] = 1
    return scenario_df

def cooling_to_scenario_table(scen_df: pd.DataFrame):
    copy_df = scen_df.copy()
    scen_df["ID_SpaceCoolingTechnology"] = 1
    copy_df["ID_SpaceCoolingTechnology"] = 2
    df = pd.concat([scen_df, copy_df], axis=0)
    return df

def price_to_scenario_table(scen_df: pd.DataFrame):
    scen_copy = scen_df.copy()
    scen_df["ID_EnergyPrice"] = 1
    scen_copy["ID_EnergyPrice"] = 2
    df = pd.concat([scen_df, scen_copy], axis=0)
    return df

def create_scenario_tables(country: str,
                           year: int,
                           ) -> None:
    # create config class with different project names, for country and year:
    cfg = get_config(project_name=f"{country}_{year}")
    clear_result_files(cfg)
    db = create_db_conn(config=cfg)
    
    db.clear_database()  # drop result tables if they exist
    # start creating the scenario tables
    LOGGER.info(f"Creating Scenario Table for {country} {year}.")

    # load the building df:
    df_buildings = load_buildings(year, country, cfg=cfg)
    # create numbers for tests:
    test_numbers = {
        "air_numbers": round(df_buildings["number_buildings_heat_pump_air"].sum()),
        "ground_numbers": round(df_buildings["number_buildings_heat_pump_ground"].sum()),
    }

    operation_scenario_component_pv = create_pv_id_table(df_buildings)
    building_pv_table = map_pv_to_building_id(operation_scenario_component_pv, df_buildings)

    building_pv_hp_table = map_heat_pump_to_scenario_table(building_pv_table, df_buildings, test_numbers)
    battery_building_pv_hp_table = map_battery_to_scenario_table(building_pv_hp_table)
    battery_building_pv_hp_tanks_table = tanks_to_scenario_table_per_building_type(scenario_df=battery_building_pv_hp_table,
                                                               cfg=cfg, 
                                                               test_numbers=test_numbers,
                                                               building_table=df_buildings)
    
    battery_building_pv_hp_tanks_cooling_table = cooling_to_scenario_table(scen_df=battery_building_pv_hp_tanks_table)

    scenario_table = price_to_scenario_table(scen_df=battery_building_pv_hp_tanks_cooling_table)
    # add the rest of the IDs which are just 1:
    scenario_table["ID_HeatingElement"] = 1
    scenario_table["ID_Region"] = 1
    scenario_table["ID_Behavior"] = 1
    scenario_table["ID_Vehicle"] = 1

    # add a scenario ID to the table:
    scenario_df = scenario_table.reset_index(drop=True)
    scenario_df.loc[:, "ID_Scenario"] = scenario_df.index + 1

    scenario_df.to_csv(cfg.input / "OperationScenario.csv", sep=";", index=False)


def clear_result_files(config: Config):
    """
    deletes the content of the result folder
    Returns: None

    """
    extensions = ["*.log", "*.csv", "*.xlsx", "*.parquet.gzip"]
    for ext in extensions:
        files = config.output.glob(ext)
        for file in files:
            file.unlink()


def main(country_list: list, years: list):

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

    # create_input_excels(year=2030, country="AUT")
    # create_scenario_tables("AUT", 2030)
    # use multiprocessing to speed it up creating all the input data:
    input_list = [(year, country) for year in years for country in country_list]
    for (year, country) in input_list:
        create_input_excels(year=year, country=country)
        create_scenario_tables(year=year, country=country)
    # number_of_physical_cores = 2# int(multiprocessing.cpu_count() / 2)
    # Parallel(n_jobs=number_of_physical_cores)(delayed(create_input_excels)(*inst) for inst in input_list)

    # # create the individual scenario tables:
    # scenario_tables_pool_list = [(country, year) for year in years for country in country_list]
    
    # Parallel(n_jobs=number_of_physical_cores)(delayed(create_scenario_tables)(*inst) for inst in scenario_tables_pool_list)



APPLICANCE_DEMAND_PER_PERSON = appliance_electricity_demand_per_person_EU()
DHW_DEMAND_PER_PERSON = specific_DHW_per_person_EU()


if __name__ == "__main__":
    country_list = [
        'BEL',
        'AUT',
        'HRV',
        # "CYP",
        'CZE',
        'DNK',
        'FRA',
        'DEU',
        'LUX',
        'HUN',
        'IRL',
        'ITA',
        'NLD',
        'POL', # 2020 hat polen PV problem
        'PRT',
        'SVK',
        'SVN',
        'ESP',
        'SWE',
        'GRC',
        'LVA',
        'LTU',
        # 'MLT',
        'ROU',
        'BGR',
        'FIN',
        'EST',
    ]
    # country_list = [ 'ROU',]#'MLT','LTU', 'GRC']
    years = [2020, 2030, 2040, 2050]
    main(country_list, years)

