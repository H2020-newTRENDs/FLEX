import numpy as np
import datetime

import pysolar.solar as pysol
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A1_CONS import CONS


def calculate_angels_of_sun(latitude, longitude, timearray, E_dir_horizontal, E_dir_diffuse):
    #  Der Azimutwinkel stellt den Horizontalwinkel der Sonne dar und beschreibt ihre Position in horizontaler Richtung
    #  durch die Abweichung von der Himmelsrichtung Süd.
    # Der Höhenwinkel (altitude) beschreibt die Höhedes Sonnenstandes und wird von der Horizontalebene aus in
    # Richtung Zenit gemessen.
    altitude_sun = np.array([])
    azimuth_sun = np.array([])
    # E_dir_horizontal = np.array([])
    for time in timearray:
        altitude_sun_1 = pysol.get_altitude(latitude, longitude, time.to_pydatetime())
        altitude_sun = np.append(altitude_sun, altitude_sun_1)
        azimuth_sun = np.append(azimuth_sun, pysol.get_azimuth(latitude, longitude, time.to_pydatetime()))
        # E_dir_horizontal = np.append(E_dir_horizontal,
        #                              radiation.get_radiation_direct(time.to_pydatetime(), altitude_sun_1))

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

    # Einstrahlungswinkel Theta https://link.springer.com/content/pdf/10.1007%2F978-3-8348-8237-0_2.pdf
    cos_theta_ost = np.cos(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)) + \
                    np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) * \
                    np.cos(np.deg2rad(azimuth_sun - azimuth_ost))

    cos_theta_west = np.cos(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)) + \
                     np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) * \
                     np.cos(np.deg2rad(azimuth_sun - azimuth_west))

    cos_theta_sued = np.cos(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)) + \
                     np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) * \
                     np.cos(np.deg2rad(azimuth_sun - azimuth_sued))

    cos_theta_nord = np.cos(np.deg2rad(altitude_sun)) * np.cos(np.deg2rad(gamma_fenster)) + \
                     np.cos(np.deg2rad(altitude_sun)) * np.sin(np.deg2rad(gamma_fenster)) * \
                     np.cos(np.deg2rad(azimuth_sun - azimuth_nord))


    # Direkt radiation:
    E_dir_sued = np.nan_to_num(E_dir_horizontal * cos_theta_sued, nan=0) / np.sin(np.deg2rad(altitude_sun))
    E_dir_west = np.nan_to_num(E_dir_horizontal * cos_theta_west, nan=0) / np.sin(np.deg2rad(altitude_sun))
    E_dir_ost = np.nan_to_num(E_dir_horizontal * cos_theta_ost, nan=0) / np.sin(np.deg2rad(altitude_sun))
    E_dir_nord = np.nan_to_num(E_dir_horizontal * cos_theta_nord, nan=0) / np.sin(np.deg2rad(altitude_sun))


    # diffuse Strahlung:
    E_diff_ost = E_dir_diffuse * (1 + np.cos(np.deg2rad(azimuth_ost))) / 2
    E_diff_west = E_dir_diffuse * (1 + np.cos(np.deg2rad(azimuth_west))) / 2
    E_diff_sued = E_dir_diffuse * (1 + np.cos(np.deg2rad(azimuth_sued))) / 2
    E_diff_nord = E_dir_diffuse * (1 + np.cos(np.deg2rad(azimuth_nord))) / 2
    E_nord = E_dir_nord + E_diff_nord
    E_sued = E_dir_sued + E_diff_sued
    E_ost = E_dir_ost + E_diff_ost
    E_west = E_dir_west + E_diff_west


    E_sued = E_sued.clip(min=0)
    E_west = E_west.clip(min=0)
    E_ost = E_ost.clip(min=0)
    E_nord = E_nord.clip(min=0)

    return azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west




if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # testing the skript
    # vienna:
    # latitude = 48.193461
    # longitude = 16.352117
    # würzburg:
    latitude = 49.791183
    longitude = 9.938962
    # Frankfurt
    # latitude = 50.11
    # longitude = 8.680965
    timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
                              tz=datetime.timezone.utc)

    # data = DB().read_DataFrame(REG_Table().Sce_Weather_Radiation, conn=DB().create_Connection(CONS().RootDB), ID_Country=5)
    # E_dir = data.loc[:, "Radiation"].to_numpy()
    E = pd.read_csv("inputdata/Timeseries_49.793_9.936_SA_v0deg_2010_2010.csv", sep=",", header=None, names=range(20)
                    ).dropna(how="all", axis=1)
    # drop rows with nan:
    E = E.dropna().reset_index(drop=True)
    # set header to first column
    header = E.iloc[0]
    E = E.iloc[1:, :]
    E.columns = header
    E_dir = E.loc[:, "Gb(i)"].to_numpy().astype(float)
    E_diff = E.loc[:, "Gd(i)"].to_numpy().astype(float)
    azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west = calculate_angels_of_sun(latitude, longitude, timearray, E_dir, E_diff)

    # create excel
    solar_power = pd.DataFrame(columns=["RadiationNorth", "RadiationEast", "RadiationSouth", "RadiationWest"])
    solar_power["RadiationNorth"] = E_nord
    solar_power["RadiationEast"] = E_ost
    solar_power["RadiationSouth"] = E_sued
    solar_power["RadiationWest"] = E_west
    solar_power = solar_power.fillna(0)
    # solar_power.to_sql("Sce_Weather_Radiation", con=DB().create_Connection(CONS().RootDB), index=False, if_exists='append', chunksize=1000)
    solar_power.to_csv("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Prosumager\\_Philipp\\inputdata\\directRadiation_himmelsrichtung_GERmitPVGIS.csv", sep=";")

    plt_anfang = 0
    plt_ende = 24
    x_achse = np.arange(plt_ende-plt_anfang)
    plt.plot(x_achse, E_sued[plt_anfang:plt_ende], "o", label="sued")
    plt.plot(x_achse, E_west[plt_anfang:plt_ende], "x", label="west")
    plt.plot(x_achse, E_ost[plt_anfang:plt_ende], "x",  label="ost")
    plt.plot(x_achse, E_nord[plt_anfang:plt_ende], "x", label="nord")
    plt.plot(x_achse, E_dir[plt_anfang:plt_ende], "x", label="E_dir", alpha=0.5)
    plt.legend()
    plt.grid()
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



