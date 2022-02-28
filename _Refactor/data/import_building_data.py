import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker


def show_building_numbers(building_df: pd.DataFrame) -> None:
    # create small overview plot
    plot_df = building_df.copy()
    plot_df["year"] = ""
    for index, row in plot_df.iterrows():
        if "gen" in row["name"].lower():
            name_extension = "_gen" + row["name"][row["name"].find("gen") + 3]
        else:
            name_extension = ""
        if "denkmalschutz" in row["name"].lower():
            plot_df.loc[index, "name"] = row["name"][:5] + "_mon" + name_extension
        else:
            plot_df.loc[index, "name"] = row["name"][:5] + name_extension

    plot_df = plot_df.drop(columns=["construction_period",
                                    "Af", "Hop", "Htr_w", "Hve", "CM_factor", "Am_factor",
                                    "average_effective_area_wind_west_east_red_cool",
                                    "average_effective_area_wind_south_red_cool",
                                    "average_effective_area_wind_north_red_cool",
                                    "spec_int_gains_cool_watt", "hwb_norm", "country", "invert_index"])

    plot_df = plot_df.melt(id_vars=["name", "year"]).rename(
        columns={"value": "number of buildings"}
    )
    ax = sns.barplot(data=plot_df, x="name", y="number of buildings", hue="variable", ci=None)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " "))
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def convert_2_digit_country_code_into_3_digit(country_code: str) -> str:
    country_abbreviations = [
         ['Austria', 'AUT', 'AT'],
         ['Belgium', 'BEL', 'BE'],
         ['Bulgaria', 'BGR', 'BG'],
         ['Croatia', 'HRV', 'HR'],
         ['Cyprus', 'CYP', 'CY'],
         ['Czech Republic', 'CZE', 'CZ'],
         ['Denmark', 'DNK', 'DK'],
         ['Estonia', 'EST', 'EE'],
         ['Finland', 'FIN', 'FI'],
         ['France', 'FRA', 'FR'],
         ['Germany', 'DEU', 'DE'],
         ['Greece', 'GRC', 'GR'],
         ['Hungary', 'HUN', 'HU'],
         ['Ireland', 'IRL', 'IE'],
         ['Italy', 'ITA', 'IT'],
         ['Latvia', 'LVA', 'LV'],
         ['Lithuania', 'LTU', 'LT'],
         ['Luxembourg', 'LUX', 'LU'],
         ['Malta', 'MLT', 'MT'],
         ['Netherlands', 'NLD', 'NL'],
         ['Poland', 'POL', 'PL'],
         ['Portugal', 'PRT', 'PT'],
         ['Romania', 'ROU', 'RO'],
         ['Slovakia', 'SVK', 'SK'],
         ['Slovenia', 'SVN', 'SI'],
         ['Spain', 'ESP', 'ES'],
         ['Sweden', 'SWE', 'SE'],
         ['United Kingdom', 'GBR', 'UK'],
    ]
    for country_list in country_abbreviations:
        if country_list[2] == country_code:
            return country_list[1]
        else:
            print("country code is not in country list \n")


def load_building_data_from_json(country: str) -> pd.DataFrame:

    absolut_path = Path("__file__").parent.parent.resolve() / Path(f"SFH_building_data.json")

    building_json = pd.read_json(absolut_path, orient="table")
    country_code = convert_2_digit_country_code_into_3_digit(country)

    # filter our data for the country
    building_json = building_json.loc[building_json.loc[:, "country"] == country_code, :]
    # make construction period only one column instead of 2:
    for index, row in building_json.iterrows():
        building_json.loc[index, "construction_period"] = str(row["construction_period_start"]) + "-" + \
                                                          str(row["construction_period_end"])
    building_json = building_json.drop(columns=["construction_period_start",
                                                "construction_period_end",
                                                "bc_index",
                                                "building_categories_index"])
    if country == "AT":
        show_building_numbers(building_json)

    # drop country code from df:
    building_json = building_json.drop(columns=["country"])
    # add ID_Building:
    building_json.loc[:, "ID_Building"] = np.arange(1, len(building_json)+1)

    return building_json


if __name__ == "__main__":
    building_data = load_building_data_from_json("AT")







