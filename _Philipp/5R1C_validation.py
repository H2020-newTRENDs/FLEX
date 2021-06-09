import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from _Philipp.Radiation import calculate_angels_of_sun
from B_Classes.B2_Building import Building, HeatingCooling_noDR
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS
from _Philipp.Optim_test import create_pyomo_model

def showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR, plot_EPlus=False):
    if plot_EPlus==True:
        B1_EPlus = pd.read_csv("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Prosumager\\_Philipp\\inputdata\\Building1_EPlus.csv", sep=";")
        B1_cooling = B1_EPlus.loc[:, "DistrictCooling:Facility [J](Hourly)"].to_numpy()
        B1_heating = B1_EPlus.loc[:, "DistrictHeating:Facility [J](Hourly) "].to_numpy()
    red = '#F47070'
    dark_red = '#a10606'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    plt.style.use('ggplot')
    colors = [red, blue, green]
    colors2 = [dark_red, dark_blue, dark_green]
    x_achse = np.arange(len(Q_Heating_noDR))
    fig1 = plt.figure()
    for i in range(1):
        plt.plot(x_achse, Q_Heating_noDR[:, i] / 1_000, label="household " + str(i+1), color=colors[i], alpha=0.8)
    plt.plot(x_achse, B1_heating / 3_600 / 1_000, label="EPlus 1", color=colors2[0], linestyle=":")
    plt.legend()
    plt.title("Heating loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig1.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichHeizlast.svg")
    plt.show()

    fig2 = plt.figure()
    for i in [2, 1, 0]:
        plt.plot(x_achse, Q_Cooling_noDR[:, i] / 1_000, label="household " + str(i+1), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Cooling loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichKuehlast.svg")
    plt.show()

    # room temperatures:
    fig3 = plt.figure()
    for i in range(3):
        plt.plot(x_achse, T_Room_noDR[:, i], label="indoor temperature " + str(i+1), color=colors[i], alpha=0.8)
        plt.plot(x_achse, T_thermalMass_noDR[:, i], label="thermal mass temperature " + str(i+1), color=colors2[i], alpha=0.8, linewidth=0.5, linestyle=":")
    plt.legend()
    plt.ylim(T_thermalMass_noDR.min(), T_thermalMass_noDR.max())
    plt.title("temperatures")
    plt.ylabel("temperature in °C")
    plt.xlabel("time in hours")
    fig3.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Temperaturen.svg")
    plt.show()






def showResults_Sprungantwort(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR):
    red = '#F47070'
    dark_red = '#a10606'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    plt.style.use('ggplot')
    colors = [red, blue, green]
    colors2 = [dark_red, dark_blue, dark_green]
    x_achse = np.arange(len(Q_Heating_noDR))
    plt.style.use('ggplot')
    width = 1

    # room temperatures:
    fig3 = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for i in range(3):
        ax1.plot(x_achse, T_thermalMass_noDR[:, i], label="thermal mass temperature " + str(i+1), color=colors2[i], alpha=0.8, linewidth=width, linestyle=":")
        ax2.plot(x_achse, Q_Heating_noDR[:, i] / 1_000, label="heating power " + str(i + 1), color=colors[i], alpha=0.8, linestyle="--", linewidth=width)
    ax1.plot(x_achse, T_Room_noDR[:, 0], label="indoor temperature ", color="black", alpha=0.8, linewidth=width)

    ax1.set_ylim(T_thermalMass_noDR.min(), T_thermalMass_noDR.max())
    ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    plt.title("thermal response 5R1C")
    ax1.set_ylabel("temperature in °C")
    ax2.set_ylabel("heating power in kW")
    plt.xlabel("time in hours")
    plt.tight_layout()
    fig3.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Sprungantwort.svg")
    plt.show()

    fig2 = plt.figure()
    for i in range(3):
        plt.plot(T_thermalMass_noDR[:, i], Q_Heating_noDR[:, i] / 1_000 , color=colors[i])
    plt.show()


def Sprungantwort(base_input_path):
    # # Heizen:
    # define building data
    buildingData = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")
    B = HeatingCooling_noDR(buildingData)

    stunden_anzahl = 24
    outdoorTemp = -5
    Temperature_outside = np.array([outdoorTemp] * stunden_anzahl)
    Q_sol = np.array([[0] * 3] * stunden_anzahl)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                       Q_solar=Q_sol,
                                                                       initial_thermal_mass_temp=21,
                                                                       T_air_min=20,
                                                                       T_air_max=26)
    showResults_Sprungantwort(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR)


def Sprungantwort_Strompreis(base_input_path):
    # define building data
    buildingData = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")
    B = HeatingCooling_noDR(buildingData)

    stunden_anzahl = 24
    outdoorTemp = -5
    Temperature_outside = np.array([outdoorTemp] * stunden_anzahl)
    Q_sol = np.array([[0] * 3] * stunden_anzahl)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                       Q_solar=Q_sol,
                                                                       initial_thermal_mass_temp=21,
                                                                       T_air_min=20,
                                                                       T_air_max=26)
    # Sprungantwort mit Strompreis
    strompreis = 0.2
    elec_price = np.array([strompreis] * stunden_anzahl)
    elec_price[:int(stunden_anzahl/2)] = 0

    # konditionierte Nutzfläche
    Af = buildingData.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af  # 7.2.2.2
    # Airtransfercoefficient
    Hve = buildingData.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = buildingData.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = buildingData.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = buildingData.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = buildingData.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = buildingData.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    # HWB_norm = data.loc[:, "hwb_norm"].to_numpy()
    # Kopplung Temp Luft mit Temp Surface Knoten s
    his = np.float_(3.45)  # 7.2.2.2
    # kopplung zwischen Masse und  zentralen Knoten s (surface)
    hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
    Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
    Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
    # thermischer Kopplungswerte W/K
    Htr_is = his * Atot
    # Equ. C.1
    PHI_ia = 0.5 * Qi

    # Equ. C.6
    Htr_1 = 1 / (1 / Hve + 1 / Htr_is)
    # Equ. C.7
    Htr_2 = Htr_1 + Htr_w
    # Equ.C.8
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)
    # COP is only used in the objetive function (min Q_heating+Q_cooling * elecprice / COP)
    COP = 3
    instance = create_pyomo_model(elec_price, Temperature_outside, Q_sol, Am, Atot, Cm, Hop, Htr_1, Htr_2, Htr_3,
                                     Htr_em, Htr_is, Htr_ms, Htr_w, Hve, PHI_ia, Qi, COP)

    T_room = np.array(list(instance.T_room.extract_values().values()))
    Q_cooling = np.array(list(instance.Q_cooling.extract_values().values())) / 1_000  # kW
    Q_heating = np.array(list(instance.Q_heating.extract_values().values())) / 1_000  # kW
    T_mass_mean = np.array(list(instance.Tm_t.extract_values().values()))

    # # Kühlen:


def compare_solar_radation():
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    Temperature_data = pd.read_csv(base_input_path / "Frankfurt_WeatherData.csv", engine="python", sep=None, header=17)
    Temperature_outside = pd.to_numeric(Temperature_data.loc[:, "DryBulb {C}"].drop(0)).to_numpy()

    # location : Frankfurt
    latitude = 50.11
    longitude = 8.680965
    # year is 2010 even if the radiation is of a representative year..
    timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
                              tz=datetime.timezone.utc)
    GlobalHorizontalRadiation = Temperature_data.loc[:, "GloHorzRad {Wh/m2}"].drop(0).to_numpy().astype(np.float)
    azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west, E_dir = \
        calculate_angels_of_sun(latitude, longitude, timearray, GlobalHorizontalRadiation)
    # define building data
    buildingData = pd.read_excel(base_input_path / "Building_data_for_EPlus.xlsx", engine="openpyxl")

    Q_sol_north = np.outer(E_nord, buildingData.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy())
    Q_sol_south = np.outer(E_sued, buildingData.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy())
    Q_sol_east_west = np.outer((E_ost + E_ost),
                               buildingData.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy())
    Q_sol = Q_sol_north + Q_sol_south + Q_sol_east_west

    # # # radiation EPlus:
    Q_sol_EPlus = pd.to_numeric(
        pd.read_csv(base_input_path / "solarRadiationEPlus.csv", sep=";").loc[:, "solar radiation"])
    Q_sol_EPlus = pd.concat([Q_sol_EPlus, Q_sol_EPlus, Q_sol_EPlus], axis=1).to_numpy()

    B = HeatingCooling_noDR(buildingData)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                                           Q_solar=Q_sol,
                                                                                           initial_thermal_mass_temp=20,
                                                                                           T_air_min=20,
                                                                                           T_air_max=26)

    Q_Heating_noDR_Eplus, Q_Cooling_noDR_Eplus, T_Room_noDR_Eplus, T_thermalMass_noDR_Eplus = B.ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol_EPlus,
        initial_thermal_mass_temp=20,
        T_air_min=20,
        T_air_max=26)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    x_achse = np.arange(len(Q_Heating_noDR))
    ax1.plot(x_achse, Q_sol[:, 0], label="rad gains pysolar")
    ax1.plot(x_achse, Q_sol_EPlus[:, 0], label="rad gains E+", alpha=0.8)
    ax1.legend()

    ax2.plot(x_achse, Q_Heating_noDR[:, 0], label="heating pysolar")
    ax2.plot(x_achse, Q_Heating_noDR_Eplus[:, 0], label="heating E+ solar", alpha=0.8)
    ax2.legend()
    fig.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\SolarRadiationVergleichEPlus.svg")
    plt.show()
    print(f"totale solar Gewinne im Jahr Pysolar: {Q_sol[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale solar Gewinne im Jahr E+: {Q_sol_EPlus[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale Heizlast Pysolar: {Q_Heating_noDR[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale Heitlast E+: {Q_Heating_noDR_Eplus[:, 0].sum() / 1_000_000:.4} MWh")

if __name__=="__main__":
    # Sprungantwort()
    compare_solar_radation()




    # showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR, plot_EPlus=True)