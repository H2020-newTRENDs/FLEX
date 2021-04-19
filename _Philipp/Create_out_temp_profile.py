import scipy.io
import pandas as pd
import numpy as np
from scipy import interpolate


def Create_out_temp_profile(input_dir_constant, OUTPUT_PATH_TEMP, RN, OUTPUT_PATH, YEAR, ds_hourly):
    # TODO soll die importdatei auf csv umgestellt werden??
    # M["x"] und M["Y"] liefern die zwei numpy arrays aus dem file:
    M = scipy.io.loadmat(input_dir_constant + "dstr_hskd.mat")

    # monthly temperature per climate zone
    # TODO Es gibt 2 Temperatur Profile, welches nehme ich? eines wird erstellt aus TEMP und eines ist gegeben mit te_obs
    temp_file = str(RN) + '___mean_temperatur_' + str(YEAR) + '.csv'
    TEMP = pd.read_csv(OUTPUT_PATH_TEMP / temp_file)
    TEMP = TEMP.set_index("month")
    # import Solar radiation
    sol_rad_file = str(RN) + '__climate_data_solar_rad_' + str(YEAR) + '.csv'
    SOL_RAD = pd.read_csv(OUTPUT_PATH / sol_rad_file)
    num_clreg = len(SOL_RAD)

    # dividing solar radiation into celestial directions:
    sol_rad_north_clreg = SOL_RAD.iloc[:, 1: 13]
    sol_rad_east_west_clreg = SOL_RAD.iloc[:, 13: 25]
    sol_rad_south_clreg = SOL_RAD.iloc[:, 25: 37]

    # Days per month
    DpM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    T_e_HSKD_8760 = ds_hourly.loc[:, "te_obs"]
    T_e_8760_clreg = np.zeros((num_clreg, 8760))
    T_e_HSKD_8760_clreg = np.tile(T_e_HSKD_8760, (num_clreg, 1))

    # Temperature and Climate:
    Te_month_mean2 = np.zeros((num_clreg, 12))
    dT_nextday = np.zeros(num_clreg)
    dT_thisday = np.zeros(num_clreg)

    # Differenz between daily min temperature and daily avg. temperature per month:
    Te1 = np.array([1.90, 0.30, 5.40, 8.30, 14.00, 17.60, 19.10, 18.50, 15.00, 10.20, 4.40, 2.40])
    Te2 = np.array([3.1, 2.1, 8.1, 11.7, 18, 21.2, 23, 22.1, 18.6, 13, 6.4, 3.4])
    DeltaT05 = Te2 - Te1

    # convert TEMP to np array:
    Te_mean_month_clreg = TEMP.to_numpy()

    cum_hours = 0
    for month in np.arange(12): # achtung monat fängt bei 0 an!
        days_this_month = DpM[month]
        num_hours = 24 * days_this_month

        # Erstellen des DailyTfacotrs mit std=0.0172 und mean=1
        Daily_T_factor2 = np.random.normal(loc=1, scale=0.0172, size=days_this_month)
        #Daily_T_factor = interpolate.splrep(M["Y"], M["x"])

        # daily temperatures:
        Temin = Te_mean_month_clreg[month, :] - DeltaT05[month]
        Temax = Te_mean_month_clreg[month, :] + DeltaT05[month]
        angle_offset = -9  # TODO was ist das? ändert er sich für andere länder?
        # TODO Wieso werden die Matrizen auf 24 aufgeblasen und nicht auf 12 (monate)?
        sinus_fkt = np.sin((np.arange(0.5, 24) + angle_offset) / (24 / 2 / 3.1415))
        Te_month_konst = np.tile((Temax - Temin) / 2 + Temin, (24, 1)).T + \
                         np.outer((Temax - Temin) / 2, sinus_fkt)

        if month == 0:
            Tm_0 = np.tile((Te_month_konst[:, 0] * 5 + 20) / 6, (24,1)).T
            Tm_10 = np.tile((Te_month_konst[:, 0] * 2 + 20) / 3, (24,1)).T

        if month > 0:
            delta_Tmonth_prev = Te_mean_month_clreg[month, :] - Te_mean_month_clreg[month-1, :]
        else:
            delta_Tmonth_prev = Te_mean_month_clreg[month, :] - Te_mean_month_clreg[11, :] # TODO -temp von Dez?

        if month < 11:
            delta_Tmonth_next = Te_mean_month_clreg[month + 1, :] - Te_mean_month_clreg[month, :]
        else:
            delta_Tmonth_next = Te_mean_month_clreg[0, :] - Te_mean_month_clreg[month, :]

        # describes the relative change of the month compared to next and last month times all days??
        days_prev_trend_prev_month = days_this_month * (delta_Tmonth_next / (delta_Tmonth_next + delta_Tmonth_prev))

        for day in np.arange(days_this_month): # achtung tage fangen bei 0 an!
            # Variation der Tagesmitteltemperaturen innerhalb eines Monats
            curr_Daily_T_factor = (Te_mean_month_clreg[month, :] + 273) * Daily_T_factor2[day] - \
                                  273 - Te_mean_month_clreg[month, :]
            Te = Te_month_konst + np.tile(curr_Daily_T_factor, (24,1)).T

            # Trend der Tagestemperatur zur durchschnittlichen Tagestemp des nächsten monats
            idx = np.where(days_prev_trend_prev_month>day)
            dT_thisday[idx] = - ((days_prev_trend_prev_month[idx] - day) / (
                    days_prev_trend_prev_month[idx] - 1) * delta_Tmonth_prev[idx] / 2)
            idx = np.where(days_prev_trend_prev_month<day)
            # TODO warum 30- und nicht monatstag?
            dT_thisday[idx] = ((day - days_prev_trend_prev_month[idx]) /
                                np.maximum(1, np.maximum(0, 30 - days_prev_trend_prev_month[idx])) *
                                delta_Tmonth_next[idx] / 2)
            Te = Te + np.tile(dT_thisday, (24, 1)).T

            idx = np.where(days_prev_trend_prev_month > (day + 1))
            dT_nextday[idx] = - ((days_prev_trend_prev_month[idx] - day) / (
                    days_prev_trend_prev_month[idx] - 1) * delta_Tmonth_prev[idx] / 2)
            idx = np.where(days_prev_trend_prev_month < (day + 1))
            dT_nextday[idx] = ((day - days_prev_trend_prev_month[idx]) /
                                np.maximum(1, np.maximum(0, 30 - days_prev_trend_prev_month[idx])) *
                                delta_Tmonth_next[idx] / 2)


            # Trend innerhalb des Tages zur Tagestemperatur des nächsten Tages
            delta_Te = (Te_month_konst[:, 0] + 273) * Daily_T_factor2[min(day + 1, days_this_month-1)] - \
                        (Te_month_konst[:,1] + 273) * Daily_T_factor2[day] + dT_nextday - dT_thisday
            Te = Te + np.outer(delta_Te, (np.arange(24) / 23))
            # tag muss +1 gerechnet werden weil sie bei 0 beginnen
            time_vector = np.arange(cum_hours + day * 24, cum_hours + (day+1) * 24)
            T_e_8760_clreg[:, time_vector] = Te

        time_vector = np.arange(cum_hours, cum_hours + num_hours)

        # Correct my algorithm towards (corrected) HSKD Data ???
        Delta_T_e_avg = np.mean(T_e_8760_clreg[:, time_vector], axis=1) - \
                        np.mean(T_e_HSKD_8760_clreg[:, time_vector], axis=1)
        T_e_8760_clreg[:, time_vector] = T_e_8760_clreg[:, time_vector] - np.tile(Delta_T_e_avg, (num_hours,1)).T
        cum_hours = cum_hours + num_hours
    # TODO warum wird T_e_8760_clreg nicht mehr verwendet in weiterer folge??
    return T_e_8760_clreg, T_e_HSKD_8760_clreg


