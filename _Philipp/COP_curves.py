# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def COP_air_HP_Thomas(outside_temperature, water_temperature):
    """
    calculates the COP based on a performance factor and the massflow temperatures
    """
    efficiency_air = 0.4
    efficiency_water = 0.35
    COP = [efficiency_air * (water_temperature + 273.15) / (water_temperature - temp) for temp in outside_temperature]
    return COP


def COP_air_HP(outside_temperature, water_temperature, thermal_power=8):
    """
    Returns the COP for an air source heat pump when provided the thermal power (between 5 and 80 kW, standard is
    set to 8) and the outside temperature as well as the nessecary hot water temperature.
    Source: "SIMULATION VON WÄRMEPUMPENSYSTEMEN AUF DER GRUNDLAGE VON
    KORRELATIONSFUNKTIONEN FÜR DIE LEISTUNGSDATEN DER WÄRMEPUMPE, Thomas Kemmler, Bernd Thomas"
    """
    # heat pumps are distinguished between small(5-18kW), medium(18-35kW), large(35-80kW)
    if thermal_power <= 18:
        HP_size = "small"
    elif 18 < thermal_power <= 35:
        HP_size = "medium"
    elif thermal_power > 35:
        HP_size = "large"

    # coefficients are in the dictionary packed:
    # 1) thermal power small, medium, large
    # 2) coefficient a, b, c, d, e, f
    # 3) outside temperature as tuple with lower and upper bound
    coeff = {"small": {  # 5-18 kW thermal
        "a": {"cold": 5.398,  # -5 to 7 °C
              "medium": 6.22734,  # 7 to 10 °C
              "warm": 5.59461},  # 10 to 25 °C

        "b": {"cold": -0.05601,
              "medium": -0.07497,
              "warm": -0.06710},

        "c": {"cold": 0.14818,
              "medium": 0.07841,
              "warm": 0.17291},

        "d": {"cold": -0.00185,
              "medium": 0,
              "warm": -0.00097},

        "e": {"cold": 0,
              "medium": 0,
              "warm": 0},

        "f": {"cold": 0.00080,
              "medium": 0,
              "warm": -0.00206}},
        # 18-35 kW_th
        "medium": {
            "a": {"cold": 4.79304,
                  "medium": 6.34439,
                  "warm": 5.07629},

            "b": {"cold": -0.04132,
                  "medium": -0.10430,
                  "warm": -0.04833},

            "c": {"cold": 0.05651,
                  "medium": 0.07510,
                  "warm": 0.09969},

            "d": {"cold": 0,
                  "medium": -0.00016,
                  "warm": -0.00096},

            "e": {"cold": 0,
                  "medium": 0.00059,
                  "warm": 0.00009},

            "f": {"cold": 0,
                  "medium": 0,
                  "warm": 0}},
        # 35-80 kW_th
        "large": {
            "a": {"cold": 6.28133,
                  "medium": 6.23384,
                  "warm": 5.00190},

            "b": {"cold": -0.10087,
                  "medium": -0.09963,
                  "warm": -0.04138},

            "c": {"cold": 0.11251,
                  "medium": 0.11295,
                  "warm": 0.10137},

            "d": {"cold": -0.00097,
                  "medium": -0.00061,
                  "warm": -0.00112},

            "e": {"cold": 0.00056,
                  "medium": 0.00052,
                  "warm": 0},

            "f": {"cold": 0.00069,
                  "medium": 0,
                  "warm": 0.00027}}}

    COP = []
    for temp in outside_temperature:
        if temp <= 7:
            wheater = "cold"
        elif 7 < temp <= 10:
            wheater = "medium"
        elif temp > 10:
            wheater = "warm"
        COP.append(coeff[HP_size]["a"][wheater] +
                   coeff[HP_size]["b"][wheater] * water_temperature +
                   coeff[HP_size]["c"][wheater] * temp +
                   coeff[HP_size]["d"][wheater] * water_temperature * temp +
                   coeff[HP_size]["e"][wheater] * water_temperature ** 2 +
                   coeff[HP_size]["f"][wheater] * temp ** 2)

    # COP = a + b * T_vl + c * T_source + d * T_vl * T_source + e * T_vl ** 2 + f * T_source ** 2
    return COP


def main():
    # import data from excel
    air_hp_imported_data = pd.read_excel(
        "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/COP_DynamicAndSeasonal.xlsx",
        sheet_name="Luft",
        engine="openpyxl",
        skiprows=1,
        skipfooter=5)
    # drop empty columns
    air_hp_imported_data = air_hp_imported_data.dropna(axis=1)

    hot_water_temperature = air_hp_imported_data.loc[:, "Warmwassertemperatur"]
    outside_temperature = air_hp_imported_data.loc[:, "Außentemperatur"]

    COP_computed = {i: COP_air_HP(np.arange(-10, 15), i) for i in hot_water_temperature}
    COP_computed_thomas = {i: COP_air_HP_Thomas(np.arange(-10, 15), i) for i in hot_water_temperature}
    # data2plot = air_hp_imported_data.drop(columns=["Warmwassertemperatur", "Außentemperatur"]).replace("-", 0)
    data2plot = air_hp_imported_data.replace("-", np.nan)

    # melted frame for seaborn:
    df_melted = pd.melt(data2plot,
                        id_vars=["Warmwassertemperatur", "Außentemperatur"],
                        var_name="heat pump type",
                        value_name="COP")
    # create plot
    sns.catplot(data=df_melted,
                x="Außentemperatur",
                y="COP",
                hue="heat pump type",
                palette="CMRmap_r",
                kind="bar")
    plt.show()

    l1 = sns.catplot(data=df_melted,
                     x="Außentemperatur",
                     y="COP",
                     hue="Warmwassertemperatur",
                     palette="CMRmap_r",
                     kind="bar")
    plt.show()

    number_of_rows = int(np.ceil(len(COP_computed) / 3))
    fig, axes = plt.subplots(number_of_rows, 3, figsize=(15, 15))
    i = 0
    j = 0
    for key in COP_computed.keys():
        axes[j, i].plot(np.arange(-10, 15), COP_computed[key], label="COP fitted")
        axes[j, i].scatter(df_melted.loc[df_melted["Warmwassertemperatur"] == key]["Außentemperatur"],
                           df_melted.loc[df_melted["Warmwassertemperatur"] == key]["COP"],
                           label="manufacturer data",
                           marker="x",
                           color="r")
        axes[j, i].plot(np.arange(-10, 15), COP_computed_thomas[key], label="COP carnot", color="green")

        axes[j, i].set_title("supply temperature: " + str(key) + "°C", fontsize=20)
        axes[j, i].legend(fontsize=16)

        for tick in axes[j, i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in axes[j, i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)

        i += 1
        if i == 3:
            i = 0
            j += 1
    # plt.legend()
    plt.tight_layout()

    plt.show()
    matplotlib.rc("font", size=14)
    data_boxplot = data2plot.loc[data2plot["Warmwassertemperatur"] == 35].drop(columns=["Warmwassertemperatur"]).set_index("Außentemperatur", drop=True).to_numpy()[:-1, :]
    x_axis = outside_temperature[3:7].to_numpy()
    boxplot_list = [i[~np.isnan(i)] for i in data_boxplot]
    fig_35 = plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(-7, 11), COP_air_HP_Thomas(np.arange(-7, 11), 35), label="COP fitted")
    locs, labels = plt.xticks()
    ax.boxplot(boxplot_list, positions=x_axis)
    ax.set_xticklabels(x_axis)

    ax.set_xlabel("Temperature in °C")
    ax.set_ylabel("COP")
    ax.set_title("COP curve for 35°C heating temperature")
    plt.legend()

    plt.savefig("C://Users//mascherbauer//PycharmProjects//NewTrends//Prosumager//_Figures//Aggregated_Results//COP//COP_35.png")
    plt.show()


if __name__ == "__main__":
    main()
