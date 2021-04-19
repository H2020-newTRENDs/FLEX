import numpy as np
import h5py


def create_matrix_after_day(num_bc):
    Q_H_RC_day = np.zeros((num_bc, 24))
    Q_C_RC_day = np.zeros((num_bc, 24))
    Q_DHW_RC_day = np.zeros((num_bc, 24))
    Q_H_EB_day = np.zeros((num_bc, 24))
    Q_C_EB_day = np.zeros((num_bc, 24))
    Q_HC = np.zeros((num_bc, 24))

    return Q_H_RC_day, Q_C_RC_day, Q_DHW_RC_day, Q_H_EB_day, Q_C_EB_day, Q_HC


def create_matrix_before_day(num_bc, days_this_month):
    Top_month = np.zeros((num_bc, 24 * days_this_month))
    Tair_month = np.zeros((num_bc, 24 * days_this_month))
    Ts_month = np.zeros((num_bc, 24 * days_this_month))

    # Zero additional Heating
    X_0 = np.zeros((num_bc, 24))
    PHIm_tot_0 = np.zeros((num_bc, 24))
    Tm_0 = np.zeros((num_bc, 24))
    Ts_0 = np.zeros((num_bc, 24))
    Tair_0 = np.zeros((num_bc, 24))
    Top_0 = np.zeros((num_bc, 24))
    # 10 W / m2 additional Heating
    X_10 = np.zeros((num_bc, 24))
    PHIm_tot_10 = np.zeros((num_bc, 24))
    Tm_10 = np.zeros((num_bc, 24))
    Ts_10 = np.zeros((num_bc, 24))
    Tair_10 = np.zeros((num_bc, 24))
    Top_10 = np.zeros((num_bc, 24))
    # actual Heating and Cooling Loads
    X_HC = np.zeros((num_bc, 24))
    PHIm_tot_HC = np.zeros((num_bc, 24))
    Tm_HC = np.zeros((num_bc, 24))
    Ts_HC = np.zeros((num_bc, 24))
    Tair_HC = np.zeros((num_bc, 24))
    Top_HC = np.zeros((num_bc, 24))

    PHIm = np.zeros((num_bc, 24))
    PHIst = np.zeros((num_bc, 24))
    PHIia = np.zeros((num_bc, 24))
    Qsol = np.zeros((num_bc, 24))

    Q_H_RC = np.zeros((num_bc, 24))
    Q_C_RC = np.zeros((num_bc, 24))

    Q_H_EB = np.zeros((num_bc, 24))
    Q_C_EB = np.zeros((num_bc, 24))
    return Top_month, Tair_month, Ts_month, X_0, PHIm_tot_0, Tm_0, Ts_0, Tair_0, Top_0, X_10, PHIm_tot_10, Tm_10, \
           Ts_10, Tair_10, Top_10, X_HC, PHIm_tot_HC, Tm_HC, Ts_HC, Tair_HC, Top_HC, PHIm, PHIst, PHIia, Qsol, Q_H_RC, \
           Q_C_RC, Q_H_EB, Q_C_EB


def create_matrix_before_month(num_bc):
    Q_H_month_RC = np.zeros((num_bc, 12))
    Q_H_month_EB = np.zeros((num_bc, 12))
    Q_C_month_RC = np.zeros((num_bc, 12))
    Q_C_month_EB = np.zeros((num_bc, 12))

    T_op_0_hourly = np.zeros((num_bc, 8760))
    T_op_10_hourly = np.zeros((num_bc, 8760))
    T_op_HC_hourly = np.zeros((num_bc, 8760))
    T_s_0_hourly = np.zeros((num_bc, 8760))
    T_s_10_hourly = np.zeros((num_bc, 8760))
    T_s_HC_hourly = np.zeros((num_bc, 8760))
    T_air_0_hourly = np.zeros((num_bc, 8760))
    T_air_10_hourly = np.zeros((num_bc, 8760))
    T_air_HC_hourly = np.zeros((num_bc, 8760))
    T_m_0_hourly = np.zeros((num_bc, 8760))
    T_m_10_hourly = np.zeros((num_bc, 8760))
    T_m_HC_hourly = np.zeros((num_bc, 8760))

    Q_H_LOAD_8760 = np.zeros((num_bc, 8760))
    Q_C_LOAD_8760 = np.zeros((num_bc, 8760))
    T_Set_8760 = np.zeros((num_bc, 8760))
    return Q_H_month_RC, Q_H_month_EB, Q_C_month_RC, Q_C_month_EB, T_op_0_hourly, T_op_10_hourly, T_op_HC_hourly, \
           T_s_0_hourly, T_s_10_hourly, T_s_HC_hourly, T_air_0_hourly, T_air_10_hourly, T_air_HC_hourly, T_m_0_hourly, \
           T_m_10_hourly, T_m_HC_hourly, Q_H_LOAD_8760, Q_C_LOAD_8760, T_Set_8760


def core_rc_model(sol_rad, data, DHW_need_day_m2_8760_up, DHW_loss_Circulation_040_day_m2_8760_up,
                  share_Circulation_DHW, temp_8760, Tset_heating_8760_up, Tset_cooling_8760_up,
                  bc_num_building_not_Zero_vctr):

    # Climate properties:
    # convert necessary variables to numpy for faster calculation:
    sol_rad = sol_rad.drop(columns=["climate_region_index"]).to_numpy()
    # climate region index - 1 because python starts to count at zero:
    climate_region_index = data.loc[:, "climate_region_index"].to_numpy().astype(int) - 1
    # user Profile index -1 because python starts indexing at zero compared to matlab
    UserProfile_idx = data.loc[:, "user_profile"].to_numpy().astype(int) - 1
    unique_climate_region_index = np.unique(climate_region_index).astype(int)

    sol_rad_north = np.empty((1, 12), float)
    sol_rad_east_west = np.empty((1, 12))
    sol_rad_south = np.empty((1, 12))
    for climate_region in unique_climate_region_index:
        anzahl = len(np.where(climate_region_index == climate_region)[0])
        sol_rad_north = np.append(sol_rad_north,
                                  np.tile(sol_rad[int(climate_region), np.arange(12)], (anzahl, 1)),
                                  axis=0)  # weil von 0 gezählt
        sol_rad_east_west = np.append(sol_rad_east_west,
                                      np.tile(sol_rad[int(climate_region), np.arange(12, 24)], (anzahl, 1)), axis=0)
        sol_rad_south = np.append(sol_rad_south,
                                  np.tile(sol_rad[int(climate_region), np.arange(24, 36)], (anzahl, 1)), axis=0)

    # delete first row in solar radiation as its just zeros
    sol_rad_north = np.delete(sol_rad_north, 0, axis=0)
    sol_rad_east_west = np.delete(sol_rad_east_west, 0, axis=0)
    sol_rad_south = np.delete(sol_rad_south, 0, axis=0)

    # Building properties:
    # konditionierte Nutzfläche
    Af = data.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af # 7.2.2.2
    # Airtransfercoefficient
    Hve = data.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = data.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = data.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = data.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = data.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = data.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    # Days per Month
    DpM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # hours per day
    h = np.arange(1, 25)
    # sol_hx = min(14, max(0, h + 0.5 - 6.5))
    # Verteilung der Sonneneinstrahlung auf die einzelnen Stunden mit fixem Profil:
    # Minimalwert = 0 und Maximalwert = 14
    sol_hx = []
    for i in h:
        if h[i - 1] + 0.5 - 6.5 < 0:
            sol_hx.append(0)
        elif h[i - 1] + 0.5 - 6.5 > 14:
            sol_hx.append(14)
        else:
            sol_hx.append(h[i - 1] + 0.5 - 6.5)

    # sol_rad_norm = max(0, np.sin(3.1415 * sol_hx / 13)) / 8.2360
    # Sinusprofil für die Sonneneinstrahlung:
    sol_rad_norm = []
    for i in sol_hx:
        if np.sin(3.1415 * i / 13) / 8.2360 < 0:
            sol_rad_norm.append(0)
        else:
            sol_rad_norm.append(np.sin(3.1415 * i / 13) / 8.2360)

    # Number of building classes:
    num_bc = len(Qi)
    print("Number of building classes: " + str(num_bc))

    # =================================================================================
    # Kopplung Temp Luft mit Temp Surface Knoten s
    his = 3.45  # 7.2.2.2
    # thermischer Kopplungswerte W/K
    Htr_is = his * Atot
    Htr_1 = 1 / (1. / Hve + 1. / Htr_is)  # Equ. C.6
    Htr_2 = Htr_1 + Htr_w  # Equ. C.7

    # kopplung zwischen Masse und  zentralen Knoten s (surface)
    hms = 9.1  # W / m2K from Equ.C.3 (from 12.2.2)
    Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
    Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8
    subVar1 = Cm / 3600 - 0.5 * (Htr_3 + Htr_em)  # Part of Equ.C.4
    subVar2 = Cm / 3600 + 0.5 * (Htr_3 + Htr_em)  # Part of Equ.C.4


    # DHW Profile: TODO warum 0.3?? vielleicht 13.2.2.1; share_CirculationDHW = 0 (weichstätten) warum?
    DHW_losses_m2_8760_up = share_Circulation_DHW * DHW_loss_Circulation_040_day_m2_8760_up + 0.3 * \
                            DHW_need_day_m2_8760_up
    # DHW losses multiplied by Area are losses for respective Buildings:
    Q_DHW_losses = DHW_losses_m2_8760_up[UserProfile_idx, :] * np.tile(Af, (8760, 1)).T
    # DHW load is DHW need multiplied by Area + DHW losses:
    Q_DHW_LOAD_8760 = DHW_need_day_m2_8760_up[UserProfile_idx, :] * np.tile(Af, (8760, 1)).T + Q_DHW_losses

    # create empty numpy frames for the following calculations:
    Q_H_month_RC, Q_H_month_EB, Q_C_month_RC, Q_C_month_EB, T_op_0_hourly, T_op_10_hourly, T_op_HC_hourly, \
        T_s_0_hourly, T_s_10_hourly, T_s_HC_hourly, T_air_0_hourly, T_air_10_hourly, T_air_HC_hourly, T_m_0_hourly, \
        T_m_10_hourly, T_m_HC_hourly, Q_H_LOAD_8760, Q_C_LOAD_8760, T_Set_8760 = \
        create_matrix_before_month(num_bc)
    cum_hours = 0
    # iterate through months:
    for month in range(12):
        days_this_month = DpM[month]
        num_hours = 24 * days_this_month

        if month == 0:
            # Estimate starting value of first temperature sets
            Tm_prev_hour = (5 * temp_8760[climate_region_index, 0] + 1 * 20) / 6
            Q_HC_prev_hour = 10

        #  Heating and Cooling Heat flow rate (Thermal power) need
        PHIHC_nd = np.zeros((num_bc, num_hours))
        # Like PHIHC_nd but with 10 W/m2 internal load
        PHIHC_nd10 = PHIHC_nd + np.tile(Af * 10, (num_hours, 1)).T

        # create empty numpy frames for the following calculations:
        Top_month, Tair_month, Ts_month, X_0, PHIm_tot_0, Tm_0, Ts_0, Tair_0, Top_0, X_10, PHIm_tot_10, Tm_10, \
        Ts_10, Tair_10, Top_10, X_HC, PHIm_tot_HC, Tm_HC, Ts_HC, Tair_HC, Top_HC, PHIm, PHIst, PHIia, Qsol, Q_H_RC, \
        Q_C_RC, Q_H_EB, Q_C_EB = create_matrix_before_day(num_bc, days_this_month)
        # iterate through days in month:
        for day in range(days_this_month):
            # create empty numpy frames for the following calculations:
            Q_H_RC_day, Q_C_RC_day, Q_DHW_RC_day, Q_H_EB_day, Q_C_EB_day, Q_HC = create_matrix_after_day(num_bc)
            # time_day_vector represents the index for the hours of one day (eg: 24-48 for second day in year)
            time_day_vector = np.arange(cum_hours + day * 24, cum_hours + (day + 1) * 24)

            # outdoor temperature
            Te = temp_8760[np.ix_(climate_region_index, time_day_vector)]

            # TODO maybe iterate over the first day twice??
            # for the first day if the year the "prev_hour" has to be set manually:
            # if month == 0 and day == 0:

            # iterate through hours:
            for hour in range(24):
                if hour == 0:
                    prev_hour = 23
                else:
                    prev_hour = hour - 1

                # solar radiation: Norm profile multiplied by typical radiation of month times 24 hours for one day
                # TODO warum *24?
                sol_rad_n = sol_rad_norm[hour] * sol_rad_north[:, month] * 24
                sol_rad_ea = sol_rad_norm[hour] * sol_rad_east_west[:, month] * 24
                sol_rad_s = sol_rad_norm[hour] * sol_rad_south[:, month] * 24

                # calculate energy needs:
                # get desired Heating and Cooling Set Point Temperature
                Tset_h = Tset_heating_8760_up[UserProfile_idx, cum_hours + day * 24 + hour]

                # TODO vielleicht User Profile Einbauen??
                # estimate User behaviour: if outdoor temp drops, user will also reduce indoor temp slightly
                Tset_h = Tset_h + (Te[:, hour] - Tset_h + 12) / 8
                # TODO Cooling temp is not adjusted to outside temp?
                Tset_c = Tset_cooling_8760_up[UserProfile_idx, cum_hours + day * 24 + hour]

                # Hourly Losses DHW supply
                Qloss_DWH = Q_DHW_losses[:, cum_hours + day * 24 + hour]

                # solar gains through windows
                Qsol[:, hour] = Awindows_rad_north * sol_rad_n + Awindows_rad_east_west * sol_rad_ea + \
                                Awindows_rad_south * sol_rad_s

                # Gains Air Note [W] TODO werden Qloss_DHW als internal gains gesehen?
                PHIia[:, hour] = 0.5 * (Qi + Qloss_DWH)  # Equ.C.1
                PHIm[:, hour] = Am / Atot * (0.5 * (Qi + Qloss_DWH) + Qsol[:, hour])  # Equ.C.2
                PHIst[:, hour] = (1 - Am / Atot - Htr_w / (hms * Atot)) * (
                        0.5 * (Qi + Qloss_DWH) + Qsol[:, hour])  # Equ.C.3

                # supply air temperature equal to outdoor temperature if no heat recovery system is considered
                # TODO implement a some households that have heat recovery
                T_air_supply = 1 * Te[:, hour] + 0 * Tair_HC[:, prev_hour]
                # Bestimmung der Lufttemperatur: Kapitel C.3
                # X_0 ist variable um Gleichung C.5 auf zu teilen (große Klammer)
                X_0[:, hour] = (PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 *
                                (((PHIia[:, hour] + PHIHC_nd[:, day * 24 + hour]) / Hve) + T_air_supply))
                PHIm_tot_0[:, hour] = PHIm[:, hour] + Htr_em * Te[:, hour] + Htr_3 * X_0[:, hour] / Htr_2 # Equ. C.5

                # Berechnung der operativen Temperatur: Kapitel C.3
                Tm_0[:, hour] = (Tm_prev_hour * subVar1 + PHIm_tot_0[:, hour]) / subVar2  # Equ. C.4
                Tm = (Tm_0[:, hour] + Tm_prev_hour) / 2  # Equ. C.9
                Ts_0[:, hour] = (Htr_ms * Tm + PHIst[:, hour] + Htr_w * Te[:, hour] +
                                 Htr_1 * (T_air_supply + (PHIia[:, hour] + PHIHC_nd[:, day * 24 + hour]) / Hve)) / \
                                (Htr_ms + Htr_w + Htr_1)  # Equ. C.10
                Tair_0[:, hour] = (Htr_is * Ts_0[:, hour] + Hve * T_air_supply + PHIia[:, hour] +
                                   PHIHC_nd[:, day * 24 + hour]) / (Htr_is + Hve)  # Equ. C.11
                Top_0[:, hour] = 0.3 * Tair_0[:, hour] + 0.7 * Ts_0[:, hour]  # Euq. C.12

                # selbe Berechnung wie oben nur für 10 W/m^2 interne Load!:
                X_10[:, hour] = PHIst[:, hour] + Htr_w * Te[:, hour] + \
                                Htr_1 * (((PHIia[:, hour] + PHIHC_nd10[:, day * 24 + hour]) / Hve) +
                                         T_air_supply)  # Part of Equ.C.5
                PHIm_tot_10[:, hour] = PHIm[:, hour] + Htr_em * Te[:, hour] + Htr_3 * X_10[:, hour] / Htr_2  # Equ.C.5
                Tm_10[:, hour] = (Tm_prev_hour * subVar1 + PHIm_tot_10[:, hour]) / subVar2  # Equ.C.4
                Tm = (Tm_10[:, hour] + Tm_prev_hour) / 2  # Equ.C.9
                Ts_10[:, hour] = (Htr_ms * Tm + PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 * (
                        T_air_supply + (PHIia[:, hour] + PHIHC_nd10[:, (day - 1) * 24 + hour]) / Hve)) / (
                                         Htr_ms + Htr_w + Htr_1)  # Equ.C.10
                Tair_10[:, hour] = (Htr_is * Ts_10[:, hour] + Hve * T_air_supply + PHIia[:, hour] +
                                    PHIHC_nd10[:, day * 24 + hour]) / (Htr_is + Hve)  # Equ.C.11
                Top_10[:, hour] = 0.3 * Tair_10[:, hour] + 0.7 * Ts_10[:, hour]  # Equ.C.12

                # Heating and cooling needs: (werden hier mit 10 W/m^2 internal gains berechnet:
                # kann nicht kleiner 0 werden..
                Q_H_RC_day[:, hour] = np.maximum(0, PHIHC_nd10[:, day * 24 + hour] * (Tset_h - Tair_0[:, hour]) /
                                                 (Tair_10[:, hour] - Tair_0[:, hour]))  # Equ. C.13
                Q_C_RC_day[:, hour] = np.maximum(0, PHIHC_nd10[:, day * 24 + hour] * (Tair_0[:, hour] - Tset_c) /
                                                 (Tair_10[:, hour] - Tair_0[:, hour]))  # Equ. C.13

                # actual heating and cooling loads
                # Q_HC combines Q_H_RC_day and Q_C_RC_day. When one of them has a value, the other one equals 0:
                # Q_HC is the mean between the heating/cooling need of the prev. hour and the actual hour.
                # TODO welche der Berechnungen für Q_HC stimmt?? ich glaube die die nicht ausgegraut ist (es war keine einzige ausgegraut)
                # Q_HC[:, hour] = (Q_H_RC_day[:, hour] + Q_H_RC_day[:, prev_hour]) / 2 - (Q_C_RC_day[:, hour] +
                #                                                                         Q_C_RC_day[:, prev_hour]) / 2
                # Heizen ist ein positiver Wert, Kühlen ein negativer:
                Q_HC[:, hour] = Q_H_RC_day[:, hour] - Q_C_RC_day[:, hour]

                # Q_HC[:, hour] = np.maximum(0, Q_H_RC_day[:, hour] - Q_C_RC_day[:, hour])
                # Q_HC[:, hour] = (Q_HC[:, hour] + Q_HC_prev_hour) / 2

                # Schritt 4: TODO muss hier nicht vielleciht eine if abfrage statt finden ob solltemp erreicht wurde?
                X_HC[:, hour] = PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 * \
                                (((PHIia[:, hour] + Q_HC[:, hour]) / Hve) + T_air_supply)  # teil von Equ. C.5
                PHIm_tot_HC[:, hour] = PHIm[:, hour] + Htr_em * Te[:, hour] + Htr_3 * X_HC[:, hour] / Htr_2  # Equ. C.5
                Tm_HC[:, hour] = (Tm_prev_hour * subVar1 + PHIm_tot_HC[:, hour]) / subVar2  # Equ.C.4
                Tm = (Tm_HC[:, hour] + Tm_prev_hour) / 2  # Equ.C.9
                Ts_HC[:, hour] = (Htr_ms * Tm + PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 * (
                        T_air_supply + (PHIia[:, hour] + Q_HC[:, hour]) / Hve)) / (Htr_ms + Htr_w + Htr_1)  # Equ.C.10
                Tair_HC[:, hour] = (Htr_is * Tm_HC[:, hour] + Hve * T_air_supply + PHIia[:, hour] + Q_HC[:, hour]) / (
                        Htr_is + Hve)  # Equ.C.11
                Top_HC[:, hour] = 0.3 * Tair_HC[:, hour] + 0.7 * Ts_HC[:, hour]  # Equ.C.12

                # heating and cooling needs, energy balance
                Q_H_EB_day[:, hour] = (Htr_w + Hop + Hve) * np.maximum(0, (Tset_h - Te[:, hour]))  # *3.600
                Q_C_EB_day[:, hour] = (Htr_w + Hop + Hve) * np.maximum(0, (Te[:, hour] - Tset_c))  # *3.600

                Tm_prev_hour = Tm_HC[:, hour]
                Q_HC_prev_hour = Q_HC[:, hour]

                # END day 1
            # save results in output vektor (cooling and heating are seperated)
            Q_H_LOAD_8760[:, time_day_vector] = Q_H_RC_day
            Q_C_LOAD_8760[:, time_day_vector] = Q_C_RC_day
            Q_H_LOAD_8760[:, time_day_vector] = np.maximum(0, Q_HC)
            Q_C_LOAD_8760[:, time_day_vector] = np.maximum(0, - Q_HC)

            T_op_HC_hourly[:, time_day_vector] = Top_HC
            T_air_HC_hourly[:, time_day_vector] = Tair_HC
            T_s_HC_hourly[:, time_day_vector] = Ts_HC
            T_m_HC_hourly[:, time_day_vector] = Tm_HC

            time_month_vector = np.arange(cum_hours,
                                          cum_hours + num_hours)  # TODO checken ob derselbe vektor raus kommt
            # aufaddieren der Heizleistung auf ein monat TODO warum tageswert/monatstage ? wird ja in schleife gemacht
            Q_H_RC = Q_H_RC + Q_H_RC_day / days_this_month
            Q_C_RC = Q_C_RC + Q_C_RC_day / days_this_month

            Q_H_EB = Q_H_EB + Q_H_EB_day / days_this_month
            Q_C_EB = Q_C_EB + Q_C_EB_day / days_this_month

            # calculate heat days
            Top_0_mean = Top_0.mean(axis=1)
            # TODO Andi fragen was dieser threshold bedeuted? (Bei mean temp über 26,28,30,32 +1 heat day? wird nicht gebraucht
            # heat_day_treshold = np.arange(26, 33, 2)
            # heat_days = np.zeros((num_bc, len(heat_day_treshold)))
            # WIRD NICHT GEBRAUCHT! Top_month und Tair_month nur für plot zur überprüfung, TODO heat days komplett unnötig?!!
            # heat_days_idx = (Top_0_mean * ones(1, length(heat_day_treshold))) > (ones(num_bc, 1) * heat_day_treshold)
            # heat_days(heat_days_idx) = heat_days(heat_days_idx) + 1
            # Top_month(:, (day - 1) * 24 + 1: day * 24) = Top_0;
            # Tair_month(:, (day - 1) * 24 + 1: day * 24) = Tair_0;

            # END MONTH
        # TODO warum /1000 ? Ist das nicht auch unnötig? ich brauch die 8760 werte..
        Q_H_month_RC[:, month] = Q_H_RC.sum(axis=1) / 1000 * DpM[month]
        Q_H_month_EB[:, month] = (Q_H_EB - np.tile(Qi, (24, 1)).T).sum(axis=1) / 1000 * DpM[month]

        Q_C_month_RC[:, month] = Q_C_RC.sum(axis=1) / 1000 * DpM[month]
        Q_C_month_EB[:, month] = (Q_C_EB - np.tile(Qi, (24, 1)).T).sum(axis=1) / 1000 * DpM[month]

        cum_hours = cum_hours + num_hours
        # END YEAR



    return Q_H_LOAD_8760, Q_C_LOAD_8760, Q_DHW_LOAD_8760, Af, bc_num_building_not_Zero_vctr, climate_region_index
