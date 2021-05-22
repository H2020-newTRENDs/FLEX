import numpy as np
import datetime
import pysolar.solar as pysol
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

    azimuth_sun = azimuth_sun - 180  # weil pysolar bei nord=0 anfängt und nicht süd=0

    # S_incident = S_horizontal / sin(altitude_sun)
    # S_window = S_incident*sin(altitude_sun + 90)

    # considering the northern hemisphere:
    # altitude below 0 = night --> no radiation


    # Neigungswinkel aller Fenster gamma:
    gamma_fenster = 90

    # azimuth winkel der Fenster:
    # nord:
    azimuth_nord = -180
    # ost:
    azimuth_ost = -90
    # sued:
    azimuth_sued = 90
    # west:
    azimuth_west = 0

    # Einstrahlungswinkel Theta
    theta_ost = np.arccos(-np.cos(altitude_sun) * np.sin(gamma_fenster) * np.cos(azimuth_sun - azimuth_ost) +
                          np.sin(altitude_sun) * np.cos(gamma_fenster))

    theta_west = np.arccos(-np.cos(altitude_sun) * np.sin(gamma_fenster) * np.cos(azimuth_sun - azimuth_west) +
                          np.sin(altitude_sun) * np.cos(gamma_fenster))

    theta_sued = np.arccos(-np.cos(altitude_sun) * np.sin(gamma_fenster) * np.cos(azimuth_sun - azimuth_sued) +
                          np.sin(altitude_sun) * np.cos(gamma_fenster))

    theta_nord = np.arccos(-np.cos(altitude_sun) * np.sin(gamma_fenster) * np.cos(azimuth_sun - azimuth_nord) +
                          np.sin(altitude_sun) * np.cos(gamma_fenster))

    # Direkt radiation:
    E_dir_sued = E_dir_horizontal * np.cos(theta_sued) / np.sin(altitude_sun)
    E_dir_west = E_dir_horizontal * np.cos(theta_west) / np.sin(altitude_sun)
    E_dir_ost = E_dir_horizontal * np.cos(theta_ost) / np.sin(altitude_sun)
    E_dir_nord = E_dir_horizontal * np.cos(theta_nord) / np.sin(altitude_sun)

    # diffuse Strahlung:
    E_diff_ost = E_dir_horizontal * (1 + np.cos(azimuth_ost)) / 2
    E_diff_west = E_dir_horizontal * (1 + np.cos(azimuth_west)) / 2
    E_diff_sued = E_dir_horizontal * (1 + np.cos(azimuth_sued)) / 2
    E_diff_nord = E_dir_horizontal * (1 + np.cos(azimuth_nord)) / 2

    # reflektierte strahlung wird vernachlässigt
    E_nord = E_dir_nord + E_diff_nord
    E_sued = E_dir_sued + E_diff_sued
    E_ost = E_dir_ost + E_diff_ost
    E_west = E_dir_west + E_diff_west


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

    azimuth_sun, altitude_sun, E_dir_sued, E_dir_west, E_dir_ost, E_dir_nord = calculate_angels_of_sun(latitude, longitude, timearray, E_dir)
    x_achse= np.arange(8760)
    plt.plot(x_achse[0:24], E_dir_sued[0:24], "x", label="sued")
    plt.plot(x_achse[0:24], E_dir_west[0:24], "x", label="west")
    plt.plot(x_achse[0:24], E_dir_ost[0:24], "x",  label="ost")
    plt.plot(x_achse[0:24], E_dir_nord[0:24], "x", label="nord")
    plt.plot(x_achse[0:24], E_dir[0:24], "x", label="E_dir")
    plt.legend()
    plt.show()


    fig2 = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(x_achse[3000:3024], azimuth_sun[3000:3024], "x", label="azi", color="orange")
    ax2.plot(x_achse[3000:3024], altitude_sun[3000:3024], "x", label="altitude")
    ax1.set_ylabel("azimuth")
    ax2.set_ylabel("altitude")
    ax1.legend()
    ax2.legend()
    plt.show()



