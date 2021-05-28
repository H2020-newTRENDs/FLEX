import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from _Philipp.Radiation import calculate_angels_of_sun
from B_Classes.B2_Building import Building, HeatingCooling_noDR
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS


def showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR):
    red = '#F47070'
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
    x_achse = np.arange(len(Q_Heating_noDR))
    fig1 = plt.figure()
    for i in range(3):
        plt.plot(x_achse, Q_Heating_noDR[:, i] / 1_000, label="household " + str(i), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Heating loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig1.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichHeizlast.svg")
    plt.show()

    fig2 = plt.figure()
    for i in [2, 1, 0]:
        plt.plot(x_achse, Q_Cooling_noDR[:, i] / 1_000, label="household " + str(i), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Cooling loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichKuehlast.svg")
    plt.show()

    # room temperatures:
    fig3 = plt.figure()
    for i in range(3):
        plt.plot(x_achse, T_Room_noDR[:, i], label="household " + str(i), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("indoor temperatures")
    plt.ylabel("temperature in °C")
    plt.xlabel("time in hours")
    fig3.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Temperaturen.svg")
    plt.show()



# path:
project_directory_path = Path(__file__).parent.resolve()
base_input_path = project_directory_path / "inputdata"
# location : Frankfurt
latitude = 50.11
longitude = 8.680965
# year is 2010 even if the radiation is of a representative year..
timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
                          tz=datetime.timezone.utc)

# define building data
buildingData = pd.read_excel(base_input_path / "Building_data_for_EPlus.xlsx", engine="openpyxl")

# define outside temperature
Temperature_data = pd.read_csv(base_input_path / "Frankfurt_WeatherData.csv", engine="python", sep=None, header=17)
Temperature_outside = Temperature_data.loc[:, 'DryBulb {C}'].drop(0).to_numpy().astype(np.float)
# define solar radiation:
GlobalHorizontalRadiation = Temperature_data.loc[:, "GloHorzRad {Wh/m2}"].drop(0).to_numpy().astype(np.float)
azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west, E_dir = \
    calculate_angels_of_sun(latitude, longitude, timearray, GlobalHorizontalRadiation)

Q_sol_north = np.outer(E_nord, buildingData.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy())
Q_sol_south = np.outer(E_sued, buildingData.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy())
Q_sol_east_west = np.outer((E_ost + E_ost), buildingData.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy())
Q_sol = Q_sol_north + Q_sol_south + Q_sol_east_west


B = HeatingCooling_noDR(buildingData)
Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                   Q_solar=Q_sol,
                                                                   initial_thermal_mass_temp=20,
                                                                   T_air_min=20,
                                                                   T_air_max=26)

showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR)
