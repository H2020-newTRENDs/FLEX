import numpy as np
import datetime
import pysolar.solar as pysol
import pandas as pd
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A1_Config.A11_Constants import CONS

def calculate_angels_of_sun(latitude, longitude, timearray, E_dir_horizontal):
    #  Der Azimutwinkel stellt den Horizontalwinkel der Sonne dar und beschreibt ihre Position in horizontaler Richtung
    #  durch die Abweichung von der Himmelsrichtung Süd.
    # Der Höhenwinkel (altitude) beschreibt die Höhedes Sonnenstandes und wird von der Horizontalebene aus in
    # Richtung Zenit gemessen.
    altitude_sun = np.array([])
    azimuth_sun = np.array([])
    for i, time in enumerate(timearray):
        altitude_sun = np.append(altitude_sun, pysol.get_altitude(latitude, longitude, time.to_pydatetime()))
        azimuth_sun = np.append(azimuth_sun, pysol.get_azimuth(latitude, longitude, time.to_pydatetime()))

    # azimuth_sun = azimuth_sun - 180  # weil pysolar bei nord=0 anfängt und nicht süd=0

    # Neigungswinkel aller Fenster gamma:
    gamma_fenster = 90

    # azimuth winkel der Fenster:
    # nord:
    azimuth_nord = 0
    # ost:
    azimuth_ost = 90
    # sued:
    azimuth_sued = 180
    # west:
    azimuth_west = 270

    # Einstrahlungswinkel Theta
    theta_ost = np.arccos(-np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) *
                          np.cos(np.deg2rad(azimuth_sun - azimuth_ost - 180)) +
                          np.sin(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)))

    theta_west = np.arccos(-np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) *
                           np.cos(np.deg2rad(azimuth_sun - azimuth_west - 180)) +
                           np.sin(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)))

    theta_sued = np.arccos(-np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) *
                           np.cos(np.deg2rad(azimuth_sun - azimuth_sued - 180)) +
                           np.sin(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)))

    theta_nord = np.arccos(-np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) *
                           np.cos(np.deg2rad(azimuth_sun - azimuth_nord - 180)) +
                           np.sin(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)))

    # Direkt radiation:
    E_dir_sued = E_dir_horizontal * np.cos(theta_sued) / np.sin(np.deg2rad(altitude_sun))
    E_dir_west = E_dir_horizontal * np.cos(theta_west) / np.sin(np.deg2rad(altitude_sun))
    E_dir_ost = E_dir_horizontal * np.cos(theta_ost) / np.sin(np.deg2rad(altitude_sun))
    E_dir_nord = E_dir_horizontal * np.cos(theta_nord) / np.sin(np.deg2rad(altitude_sun))

    # diffuse Strahlung:
    E_diff_ost = E_dir_horizontal * (1 + np.cos(np.deg2rad(azimuth_ost))) / 2
    E_diff_west = E_dir_horizontal * (1 + np.cos(np.deg2rad(azimuth_west))) / 2
    E_diff_sued = E_dir_horizontal * (1 + np.cos(np.deg2rad(azimuth_sued))) / 2
    E_diff_nord = E_dir_horizontal * (1 + np.cos(np.deg2rad(azimuth_nord))) / 2

    # reflektierte strahlung wird vernachlässigt
    E_nord = E_dir_nord #+ E_diff_nord
    E_sued = E_dir_sued #+ E_diff_sued
    E_ost = E_dir_ost #+ E_diff_ost
    E_west = E_dir_west #+ E_diff_west

    E_sued = E_sued.clip(min=0)
    E_west = E_west.clip(min=0)
    E_ost = E_ost.clip(min=0)
    E_nord = E_nord.clip(min=0)

    return azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west




if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # testing the skript
    latitude = 48.193461
    longitude = 16.352117
    timearray = pd.date_range("01-01-2010", "01-01-2011", freq="H", closed="left",
                              tz=datetime.timezone.utc)

    data = DB().read_DataFrame(REG().Sce_Weather_Radiation, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    E_dir = data.loc[:, "Radiation"].to_numpy()

    azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west = calculate_angels_of_sun(latitude, longitude, timearray, E_dir)

    # create excel
    solar_power = pd.DataFrame(columns=["RadiationNorth", "RadiationEast", "RadiationSouth", "RadiationWest"])
    solar_power["RadiationNorth"] = E_nord
    solar_power["RadiationEast"] = E_ost
    solar_power["RadiationSouth"] = E_sued
    solar_power["RadiationWest"] = E_west
    solar_power.to_csv("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Prosumager\\_Philipp\\inputdata\\directRadiation_himmelsrichtung.csv", sep=";")

    plt_anfang = 3000
    plt_ende = 3024
    x_achse = np.arange(plt_ende-plt_anfang)
    plt.plot(x_achse, E_sued[plt_anfang:plt_ende], "o", label="sued")
    plt.plot(x_achse, E_west[plt_anfang:plt_ende], "x", label="west")
    plt.plot(x_achse, E_ost[plt_anfang:plt_ende], "x",  label="ost")
    plt.plot(x_achse, E_nord[plt_anfang:plt_ende], "x", label="nord")
    plt.plot(x_achse, E_dir[plt_anfang:plt_ende], "x", label="E_dir")
    plt.legend()
    plt.show()


    fig2 = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(x_achse, azimuth_sun[plt_anfang:plt_ende], "x", label="azimuth", color="orange")
    ax2.plot(x_achse, altitude_sun[plt_anfang:plt_ende], "x", label="altitude")
    ax1.set_ylabel("azimuth")
    ax2.set_ylabel("altitude")
    ax1.legend()
    ax2.legend()
    plt.show()



