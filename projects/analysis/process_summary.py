import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "ECEMF_Summary",
    "input"
)

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "ECEMF_Summary",
    "output"
)

SUBMISSION_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "ECEMF_Summary",
    "submission"
)

SUMMARY_YEAR = "_SummaryYear.csv"
SUMMARY_HOUR = "_SummaryHour.csv"
INVERT_SUMMARY = "_INVERT_Summary.xlsx"
SUMMARY_HOUR_SELECTIVE_AGGREGATION = "SummaryHourSelectiveAggregation.csv"
SUMMARY_HOUR_SELECTIVE_AGGREGATION_INVERT_HARMONIZED = "SummaryHourSelectiveAggregationInvertHarmonized.csv"
IAMC_YEAR = "IamcYear.xlsx"
IAMC_YEAR_SUBMISSION = "IamcYear_Submission.xlsx"
IAMC_HOUR = "IamcHour.csv"
IAMC_HOUR_MIXED = "IamcHour_Mixed.csv"
IAMC_HOUR_SUBMISSION = "IamcHour_Submission.csv"


def concat_summary():
    summary_year_l = []
    summary_hour_l = []
    for csv_file_path in glob.glob(f"{INPUT_FOLDER}/*.csv"):
        file_name, _ = os.path.splitext(os.path.basename(csv_file_path))
        if file_name.endswith("_SummaryHour"):
            summary_hour_l.append(pd.read_csv(csv_file_path))
        elif file_name.endswith("_SummaryYear"):
            summary_year_l.append(pd.read_csv(csv_file_path))
    pd.concat(summary_year_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_YEAR), index=False)
    pd.concat(summary_hour_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR), index=False)


def start_from_8760h_summary():

    indices_aggregated = [
        "Country",
        "Year",
        "ID_SEMS",
        "ID_Teleworking",
        "YearHour",
        "DayHour",
        "demand_unit",
    ]

    cols_kept = [
        "E_Heating_HP_out",
        "E_DHW_HP_out",
        "BaseLoadProfile",
        "PhotovoltaicProfile",
        "PV2Load",
        "PV2Bat",
        "Grid",
        "Fuel",
        "ElectricityPrice"
    ]

    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l_aggregated = []
    for country in tqdm(countries):
        for year in years:
            for id_sems in [1, 2]:
                for id_teleworking in [1, 2]:
                    df_filtered = df.loc[
                        (df["Country"] == country) &
                        (df["Year"] == year) &
                        (df["ID_SEMS"] == id_sems) &
                        (df["ID_Teleworking"] == id_teleworking)
                        ]
                    l_aggregated.append(df_filtered.groupby(indices_aggregated, as_index=False)[cols_kept].sum())
    df_result = pd.concat(l_aggregated, axis=0, ignore_index=True)
    df_result["ElectricityPrice"] = df_result["ElectricityPrice"] / 8
    df_result_aut = df_result.loc[df_result["Country"] == "AUT"]
    df_result.to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_SELECTIVE_AGGREGATION), index=False)
    df_result_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{SUMMARY_HOUR_SELECTIVE_AGGREGATION[0:-4]}_AUT.xlsx"), index=False)


def get_invert_summary():
    df = pd.read_excel(os.path.join(OUTPUT_FOLDER, INVERT_SUMMARY))
    countries = df["country"].unique().tolist()
    years = df["year"].unique().tolist()
    invert_summary = {}
    for country in countries:
        for year in years:
            df_filter = df.loc[(df["country"] == country) & (df["year"] == year)]
            invert_summary[(country, year)] = {}
            for _, row in df_filter.iterrows():
                invert_summary[(country, year)][row["energy_carrier"]] = row["value"]
    return invert_summary


def harmonize_with_invert():

    fuel_energy_carriers = [
        "electricity",
        "wood",
        "coal",
        "district_heating",
        "gas",
        "oil"
    ]

    other_energy_carriers = [
        "solar_thermal",
    ]

    invert_summary = get_invert_summary()
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_SELECTIVE_AGGREGATION))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l = []
    for country in tqdm(countries):
        for year in years:
            df_filtered = df.loc[(df["Country"] == country) & (df["Year"] == year)]
            total_fuel = 0
            for fuel in fuel_energy_carriers:
                total_fuel += invert_summary[(country, year)][fuel]
            for _, row in df_filtered.iterrows():
                # collect heating system
                d = row.to_dict()
                for fuel in fuel_energy_carriers:
                    d[fuel] = d["Fuel"] * invert_summary[(country, year)][fuel] / total_fuel
                for other_carrier in other_energy_carriers:
                    d[other_carrier] = d["Fuel"] * invert_summary[(country, year)][other_carrier] / total_fuel
                l.append(d)
    df_result = pd.DataFrame(l)
    df_result_aut = df_result.loc[df_result["Country"] == "AUT"]
    df_result.to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_SELECTIVE_AGGREGATION_INVERT_HARMONIZED), index=False)
    df_result_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{SUMMARY_HOUR_SELECTIVE_AGGREGATION_INVERT_HARMONIZED[0:-4]}_AUT.xlsx"), index=False)


D_SCENARIOS = {
        (1, 1): "WP4|Residential|Digitalization in Residential Sector|Reference",
        (2, 1): "WP4|Residential|Digitalization in Residential Sector|Prosumaging",
        (1, 2): "WP4|Residential|Digitalization in Residential Sector|Teleworking",
        (2, 2): "WP4|Residential|Digitalization in Residential Sector|Prosumaging and Teleworking"
    }


def implement_iamc_formatting_year():

    variables = {
        "Final Energy|Residential|Electricity": ["Grid", "electricity"],
        "Final Energy|Residential|Appliances|Electricity": ["BaseLoadProfile"],
        "Final Energy|Residential|Space and Water Heating": ["E_Heating_HP_out", "E_DHW_HP_out", "electricity", "wood", "coal", "district_heating", "gas", "oil", "solar_thermal"],
        "Final Energy|Residential|Space and Water Heating|Electricity": ["E_Heating_HP_out", "E_DHW_HP_out", "electricity"],
        "Final Energy|Residential|Space and Water Heating|Solids": ["wood", "coal"],
        "Final Energy|Residential|Space and Water Heating|Solids|Biomass": ["wood"],
        "Final Energy|Residential|Space and Water Heating|Solids|Fossil": ["coal"],
        "Final Energy|Residential|Space and Water Heating|Heat": ["district_heating"],
        "Final Energy|Residential|Space and Water Heating|Gases": ["gas"],
        "Final Energy|Residential|Space and Water Heating|Gases|Fossil": ["gas"],
        "Final Energy|Residential|Space and Water Heating|Liquids": ["oil"],
        "Final Energy|Residential|Space and Water Heating|Liquids|Fossil": ["oil"],
        "Final Energy|Residential|Space and Water Heating|Solar": ["solar_thermal"],
        "Energy Supply|Residential|PV": ["PhotovoltaicProfile"],
        "Energy Supply|Residential|PV|SelfConsumption": ["PV2Load", "PV2Bat"],
    }

    def base_dict():
        d = {
            "model": "FLEX v1.0.0",
            "scenario": D_SCENARIOS[(id_sems, id_teleworking)],
            "region": country,
            "variable": variable,
            "unit": "EJ/yr",
            "subannual": "Year"
        }
        return d

    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_SELECTIVE_AGGREGATION_INVERT_HARMONIZED))
    countries = df["Country"].unique().tolist()
    l = []
    for country in tqdm(countries):
        for id_sems in [1, 2]:
            for id_teleworking in [1, 2]:
                df_filtered = df.loc[
                    (df["Country"] == country) &
                    (df["ID_SEMS"] == id_sems) &
                    (df["ID_Teleworking"] == id_teleworking)
                ]
                for variable, cols in variables.items():
                    d = base_dict()
                    for year in [2020, 2030, 2040, 2050]:
                        df_filtered_year = df_filtered.loc[df_filtered["Year"] == year]
                        d[year] = 0
                        for col in cols:
                            d[year] += df_filtered_year[col].sum() * 3.6 * 10**(-6)  # from GWh to EJ
                    l.append(d)
    pd.DataFrame(l).to_excel(os.path.join(OUTPUT_FOLDER, IAMC_YEAR), index=False, sheet_name="data")


def mix_scenarios_year():

    def gen_mix_scenarios(df1, share1, df2):
        l = []
        for _, row in tqdm(df1.iterrows(), total=len(df1)):
            d = row.to_dict()
            s2 = df2.loc[
                (df2["region"] == row["region"]) &
                (df2["variable"] == row["variable"])
                ].iloc[0]
            d["scenario"] = s2["scenario"]
            for year in [2020, 2030, 2040, 2050]:
                if row["variable"] != "Energy Supply|Residential|PV":
                    d[year] = d[year] * share1 + s2[year] * (1 - share1)
            l.append(d)
        return pd.DataFrame(l)

    def insert_pv_2_grid():
        l_to_concat = []
        scenarios = df_mixed["scenario"].unique().tolist()
        regions = df_mixed["region"].unique().tolist()
        for scenario in scenarios:
            for region in regions:
                df_mixed_filtered = df_mixed.loc[(df_mixed["scenario"] == scenario) & (df_mixed["region"] == region)]
                new_row = df_mixed_filtered.iloc[0]
                new_row["variable"] = "Energy Supply|Residential|PV|PV2Grid"
                for year in [2020, 2030, 2040, 2050]:
                    self_consumption = df_mixed_filtered.loc[df_mixed_filtered["variable"] == "Energy Supply|Residential|PV|SelfConsumption"].iloc[0][year]
                    generation = df_mixed_filtered.loc[df_mixed_filtered["variable"] == "Energy Supply|Residential|PV"].iloc[0][year]
                    new_row[year] = generation - self_consumption
                df_mixed_filtered = pd.concat([df_mixed_filtered, new_row.to_frame().T])
                l_to_concat.append(df_mixed_filtered)
        return pd.concat(l_to_concat)

    df = pd.read_excel(os.path.join(OUTPUT_FOLDER, IAMC_YEAR), sheet_name="data")
    df_ref = df.loc[df["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Reference"]
    df_sems = df.loc[df["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Prosumaging"]
    df_tele = df.loc[df["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Teleworking"]
    df_sems_tele = df.loc[df["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Prosumaging and Teleworking"]
    l_mixed = [df_ref, df_sems]
    l_mixed.append(gen_mix_scenarios(df_ref, 0.75, df_tele))
    l_mixed.append(gen_mix_scenarios(df_sems, 0.75, df_sems_tele))
    df_mixed = pd.concat(l_mixed)
    df_mixed = insert_pv_2_grid()
    df_mixed.to_excel(os.path.join(OUTPUT_FOLDER, IAMC_YEAR_SUBMISSION), index=False, sheet_name="data")


def implement_iamc_formatting_hour():

    variables = {
        "Final Energy|Residential|Electricity": ["Grid", "electricity"],
        "Energy Supply|Residential|PV": ["PhotovoltaicProfile"],
        "Energy Supply|Residential|PV|SelfConsumption": ["PV2Load", "PV2Bat"],
        "Energy Price|Residential|Electricity": ["ElectricityPrice"],
    }

    def base_dict():
        d = {
            "model": "FLEX v1.0.0",
            "scenario": D_SCENARIOS[(id_sems, id_teleworking)],
            "region": country,
            "variable": "",
            "unit": "GWh",
            "year": year,
            "year_hour": [],
            "day_hour": [],
            "value": np.zeros(8760, ),
        }
        return d

    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_SELECTIVE_AGGREGATION_INVERT_HARMONIZED))
    countries = df["Country"].unique().tolist()
    l = []
    for country in tqdm(countries):
        for year in [2020, 2030, 2040, 2050]:
            for id_sems in [1, 2]:
                for id_teleworking in [1, 2]:
                    df_filtered = df.loc[
                        (df["Country"] == country) &
                        (df["Year"] == year) &
                        (df["ID_SEMS"] == id_sems) &
                        (df["ID_Teleworking"] == id_teleworking)
                    ]
                    for variable, cols in variables.items():
                        d = base_dict()
                        d["year_hour"] = df_filtered["YearHour"].to_list()
                        d["day_hour"] = df_filtered["DayHour"].to_list()
                        d["variable"] = variable
                        if variable == "Energy Price|Residential|Electricity":
                            d["unit"] = "Euro/kWh"
                        for col in cols:
                            d["value"] += df_filtered[col].to_numpy()
                        l.append(pd.DataFrame(d))
    df_result = pd.concat(l)
    df_result.to_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR), index=False)
    df_result_aut = df_result.loc[df_result["region"] == "AUT"]
    df_result_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{IAMC_HOUR[0:-4]}_AUT.xlsx"), index=False, sheet_name="data")


def mix_scenarios_hour():

    def gen_mix_scenarios(df1, share1, df2):
        l = []
        for _, row in tqdm(df1.iterrows(), total=len(df1)):
            d = row.to_dict()
            s2 = df2.loc[
                (df2["region"] == row["region"]) &
                (df2["variable"] == row["variable"]) &
                (df2["year"] == row["year"]) &
                (df2["year_hour"] == row["year_hour"])
                ].iloc[0]
            d["scenario"] = s2["scenario"]
            if d["variable"] != "Energy Supply|Residential|PV":
                d["value"] = d["value"] * share1 + s2["value"] * (1 - share1)
            l.append(d)
        return pd.DataFrame(l)

    l_mixed = []
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR))
    regions = df["region"].unique().tolist()
    for region in tqdm(regions):
        df_filtered = df.loc[df["region"] == region]
        df_ref = df_filtered.loc[df_filtered["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Reference"]
        df_sems = df_filtered.loc[df_filtered["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Prosumaging"]
        df_tele = df_filtered.loc[df_filtered["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Teleworking"]
        df_sems_tele = df_filtered.loc[df_filtered["scenario"] == "WP4|Residential|Digitalization in Residential Sector|Prosumaging and Teleworking"]
        l_mixed.append(df_ref)
        l_mixed.append(df_sems)
        l_mixed.append(gen_mix_scenarios(df_ref, 0.75, df_tele))
        l_mixed.append(gen_mix_scenarios(df_sems, 0.75, df_sems_tele))
    df_mixed = pd.concat(l_mixed)
    df_mixed.to_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR_MIXED), index=False)
    df_mixed_aut = df_mixed.loc[df_mixed["region"] == "AUT"]
    df_mixed_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{IAMC_HOUR_MIXED[0:-4]}_AUT.xlsx"), index=False, sheet_name="data")


def insert_pv_2_grid_hour():
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR_MIXED))
    scenarios = df["scenario"].unique().tolist()
    regions = df["region"].unique().tolist()
    l = []
    for scenario in tqdm(scenarios):
        for region in regions:
            for year in [2020, 2030, 2040, 2050]:
                df_filtered = df.loc[
                    (df["scenario"] == scenario) &
                    (df["region"] == region) &
                    (df["year"] == year)
                ]
                df_pv2grid = df_filtered.loc[df_filtered["variable"] == "Final Energy|Residential|Electricity"]
                df_pv2grid["variable"] = "Energy Supply|Residential|PV|PV2Grid"
                df_pv = df_filtered.loc[df_filtered["variable"] == "Energy Supply|Residential|PV"]
                df_pv_self_consumption = df_filtered.loc[df_filtered["variable"] == "Energy Supply|Residential|PV|SelfConsumption"]
                df_pv2grid["value"] = df_pv["value"].to_numpy() - df_pv_self_consumption["value"].to_numpy()
                l.append(pd.concat([df_filtered, df_pv2grid]))
    df_inserted = pd.concat(l)
    df_inserted.to_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR_SUBMISSION), index=False)
    df_inserted_aut = df_inserted.loc[df_inserted["region"] == "AUT"]
    df_inserted_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{IAMC_HOUR_SUBMISSION[0:-4]}_AUT.xlsx"), index=False, sheet_name="data")


def filter_aut():
    df_inserted = pd.read_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR_SUBMISSION))
    df_inserted_aut = df_inserted.loc[df_inserted["region"] == "AUT"]
    df_inserted_aut.to_excel(os.path.join(OUTPUT_FOLDER, f"{IAMC_HOUR_SUBMISSION[0:-4]}_AUT.xlsx"), index=False,
                             sheet_name="data")


def aggregate_by_day_hour():
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, IAMC_HOUR_SUBMISSION))
    df = df.groupby([
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        "year",
        "day_hour"
    ], as_index=False)["value"].mean()
    df.to_excel(os.path.join(OUTPUT_FOLDER, "IamcHour_DayHourAverage.xlsx"), index=False)


def divide_electricity_price_with_eight():
    file_name = "ECEMF_T4.2_Digitalization in Residential Sector_Hour Results"
    df = pd.read_csv(os.path.join(SUBMISSION_FOLDER, f"{file_name}.csv"))
    filtered_rows = df[df["variable"] == "Energy Price|Residential|Electricity"]
    filtered_rows["value"] = filtered_rows["value"] / 8
    df.update(filtered_rows)
    df.to_csv(os.path.join(SUBMISSION_FOLDER, f"{file_name}.csv"), index=False)


def update_hour_submission():

    variables = [
        "Final Energy|Residential|Electricity",
        "Energy Supply|Residential|PV",
        "Energy Supply|Residential|PV|SelfConsumption",
        "Energy Price|Residential|Electricity"
    ]

    def get_country_code():
        country_code_df = pd.read_excel(os.path.join(SUBMISSION_FOLDER, f"country_codes.xlsx"))
        return dict(zip(country_code_df["code"], country_code_df["name"]))

    def update_region_name(row):
        return country_code[row["region_ios3"]]

    def get_sub_annual_col():
        from datetime import datetime, timedelta
        start_date = datetime(2023, 1, 1, 0, 0)
        datetime_list = [start_date + timedelta(hours=i) for i in range(8760)]
        return [dt.strftime("%m-%d %H:%M+01:00") for dt in datetime_list]

    file_name = "ECEMF_T4.2_Digitalization in Residential Sector_Hour Results"
    df = pd.read_csv(os.path.join(SUBMISSION_FOLDER, f"{file_name}.csv"))
    # file_name = "ECEMF_T4.2_Digitalization in Residential Sector_Hour Results (Austria)"
    # df = pd.read_excel(os.path.join(SUBMISSION_FOLDER, f"{file_name}.xlsx"))
    country_code = get_country_code()
    df = df.rename(columns={'region': 'region_ios3'})
    df['region'] = df.apply(update_region_name, axis=1)
    df = df.drop('region_ios3', axis=1)
    dfs = []
    for _, name in tqdm(country_code.items(), desc="EU27 --> "):
        df_r = df.loc[df["region"] == name]
        if len(df_r) > 0:
            for scenario in D_SCENARIOS.values():
                df_rs = df_r.loc[df_r["scenario"] == scenario]
                for variable in variables:
                    df_rsv = df_rs.loc[df_rs["variable"] == variable]
                    pivoted_df = df_rsv.pivot_table(
                        index=[
                            "model",
                            "scenario",
                            "region",
                            "variable",
                            "year_hour",
                            "unit"
                        ],
                        columns='year',
                        values='value'
                    ).reset_index()
                    pivoted_df.insert(6, "subannual", get_sub_annual_col())
                    pivoted_df.drop('year_hour', axis=1, inplace=True)
                    dfs.append(pivoted_df)
    final = pd.concat(dfs)
    final.to_csv("ECEMF_T4.2_Digitalization in Residential Sector_Hour Results (wide).csv", index=False)
    final.to_csv(
        os.path.join(
            SUBMISSION_FOLDER,
            "ECEMF_T4.2_Digitalization in Residential Sector_Hour Results (wide).csv"
        ),
        index=False)


def separate_hour_submission():
    file_name = "ECEMF_T4.2_Digitalization in Residential Sector_Hour Results (wide)"
    df = pd.read_csv(os.path.join(SUBMISSION_FOLDER, f"{file_name}.csv"))
    for scenario in D_SCENARIOS.values():
        df_filter = df.loc[df["scenario"] == scenario]
        df_filter.to_csv(
            os.path.join(
                SUBMISSION_FOLDER,
                f"ECEMF_T4.2_Digitalization in Residential Sector_Hour Results (wide_{scenario.split('|')[-1]}).csv"
            ),
            index=False)
