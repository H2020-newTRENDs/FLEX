import numpy as np
import timeit
from numba import njit


# @njit()
def create_matrix_after_day(daylist):
    Q_H_RC_day = np.zeros_like(daylist)
    Q_C_RC_day = np.zeros_like(daylist)
    Q_DHW_RC_day = np.zeros_like(daylist)
    Q_H_EB_day = np.zeros_like(daylist)
    Q_C_EB_day = np.zeros_like(daylist)
    Q_HC = np.zeros_like(daylist)

    return Q_H_RC_day, Q_C_RC_day, Q_DHW_RC_day, Q_H_EB_day, Q_C_EB_day, Q_HC


# @njit()
def create_matrix_before_day(daylist):
    # Zero additional Heating
    X_0 = np.zeros_like(daylist)
    PHIm_tot_0 = np.zeros_like(daylist)
    Tm_0 = np.zeros_like(daylist)
    Ts_0 = np.zeros_like(daylist)
    Tair_0 = np.zeros_like(daylist)
    Top_0 = np.zeros_like(daylist)
    # 10 W / m2 additional Heating
    X_10 = np.zeros_like(daylist)
    PHIm_tot_10 = np.zeros_like(daylist)
    Tm_10 = np.zeros_like(daylist)
    Ts_10 = np.zeros_like(daylist)
    Tair_10 = np.zeros_like(daylist)
    Top_10 = np.zeros_like(daylist)
    # actual Heating and Cooling Loads
    X_HC = np.zeros_like(daylist)
    PHIm_tot_HC = np.zeros_like(daylist)
    Tm_HC = np.zeros_like(daylist)
    Ts_HC = np.zeros_like(daylist)
    Tair_HC = np.zeros_like(daylist)
    Top_HC = np.zeros_like(daylist)

    PHIm = np.zeros_like(daylist)
    PHIst = np.zeros_like(daylist)
    PHIia = np.zeros_like(daylist)
    Qsol = np.zeros_like(daylist)

    Q_H_RC = np.zeros_like(daylist)
    Q_C_RC = np.zeros_like(daylist)

    Q_H_EB = np.zeros_like(daylist)
    Q_C_EB = np.zeros_like(daylist)
    return X_0, PHIm_tot_0, Tm_0, Ts_0, Tair_0, Top_0, X_10, PHIm_tot_10, Tm_10, \
           Ts_10, Tair_10, Top_10, X_HC, PHIm_tot_HC, Tm_HC, Ts_HC, Tair_HC, Top_HC, PHIm, PHIst, PHIia, Qsol, Q_H_RC, \
           Q_C_RC, Q_H_EB, Q_C_EB


# @njit()
def create_matrix_before_month(monthlist, yearlist):
    Q_H_month_RC = np.zeros_like(monthlist)
    Q_H_month_EB = np.zeros_like(monthlist)
    Q_C_month_RC = np.zeros_like(monthlist)
    Q_C_month_EB = np.zeros_like(monthlist)

    T_op_0_hourly = np.zeros_like(yearlist)
    T_op_10_hourly = np.zeros_like(yearlist)
    T_op_HC_hourly = np.zeros_like(yearlist)
    T_s_0_hourly = np.zeros_like(yearlist)
    T_s_10_hourly = np.zeros_like(yearlist)
    T_s_HC_hourly = np.zeros_like(yearlist)
    T_air_0_hourly = np.zeros_like(yearlist)
    T_air_10_hourly = np.zeros_like(yearlist)
    T_air_HC_hourly = np.zeros_like(yearlist)
    T_m_0_hourly = np.zeros_like(yearlist)
    T_m_10_hourly = np.zeros_like(yearlist)
    T_m_HC_hourly = np.zeros_like(yearlist)

    Q_H_LOAD_8760 = np.zeros_like(yearlist)
    Q_C_LOAD_8760 = np.zeros_like(yearlist)
    T_Set_8760 = np.zeros_like(yearlist)
    return Q_H_month_RC, Q_H_month_EB, Q_C_month_RC, Q_C_month_EB, T_op_0_hourly, T_op_10_hourly, T_op_HC_hourly, \
           T_s_0_hourly, T_s_10_hourly, T_s_HC_hourly, T_air_0_hourly, T_air_10_hourly, T_air_HC_hourly, T_m_0_hourly, \
           T_m_10_hourly, T_m_HC_hourly, Q_H_LOAD_8760, Q_C_LOAD_8760, T_Set_8760


# @njit()
def core_rc_model(DHW_need_day_m2_8760_up, DHW_loss_Circulation_040_day_m2_8760_up,
                  share_Circulation_DHW, temp_8760, Tset_heating_8760_up, Tset_cooling_8760_up, sol_rad_north,
                  sol_rad_south, sol_rad_east_west, sol_rad_norm, Af, Atot, Hve, Htr_w, Hop, Cm, Am, Qi,
                  Awindows_rad_east_west, Awindows_rad_north, Awindows_rad_south, DpM, UserProfile_idx, num_bc,
                  daylist, monthlist, yearlist):
    # Kopplung Temp Luft mit Temp Surface Knoten s
    his = np.float_(3.45)  # 7.2.2.2
    # thermischer Kopplungswerte W/K
    Htr_is = his * Atot
    Htr_1 = np.float_(1) / (np.float_(1) / Hve + np.float_(1) / Htr_is)  # Equ. C.6
    Htr_2 = Htr_1 + Htr_w  # Equ. C.7

    # kopplung zwischen Masse und  zentralen Knoten s (surface)
    hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
    Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
    Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8
    subVar1 = Cm / 3600 - 0.5 * (Htr_3 + Htr_em)  # Part of Equ.C.4
    subVar2 = Cm / 3600 + 0.5 * (Htr_3 + Htr_em)  # Part of Equ.C.4

    # insert frames that will be needed later? matlab zeile 95-111

    # DHW Profile: TODO warum 0.3?? vielleicht 13.2.2.1; share_CirculationDHW = 0 (weichstätten) warum?
    DHW_losses_m2_8760_up = share_Circulation_DHW * DHW_loss_Circulation_040_day_m2_8760_up + 0.3 * \
                            DHW_need_day_m2_8760_up
    # DHW losses multiplied by Area are losses for respective Buildings:
    Q_DHW_losses = DHW_losses_m2_8760_up[UserProfile_idx, :] * Af
    # DHW load is DHW need multiplied by Area + DHW losses:
    Q_DHW_LOAD_8760 = DHW_need_day_m2_8760_up[UserProfile_idx, :] * Af + Q_DHW_losses

    # create empty numpy frames for the following calculations:
    Q_H_month_RC, Q_H_month_EB, Q_C_month_RC, Q_C_month_EB, T_op_0_hourly, T_op_10_hourly, T_op_HC_hourly, \
    T_s_0_hourly, T_s_10_hourly, T_s_HC_hourly, T_air_0_hourly, T_air_10_hourly, T_air_HC_hourly, T_m_0_hourly, \
    T_m_10_hourly, T_m_HC_hourly, Q_H_LOAD_8760, Q_C_LOAD_8760, T_Set_8760 = \
        create_matrix_before_month(monthlist, yearlist)
    cum_hours = 0
    # iterate through months:
    for month in range(12):
        days_this_month = DpM[month]
        num_hours = 24 * days_this_month
        hours_up_to_this_month = np.sum(DpM[:month] * 24)

        if month == 0:
            # Estimate starting value of first temperature sets
            Tm_prev_hour = (5 * temp_8760[:, 0] + 1 * 20) / 6
            Q_HC_prev_hour = 10

        #  Heating and Cooling Heat flow rate (Thermal power) need
        PHIHC_nd = np.zeros((num_bc, num_hours))
        # Like PHIHC_nd but with 10 W/m2 internal load
        PHIHC_nd10 = PHIHC_nd + Af[:, hours_up_to_this_month:num_hours + hours_up_to_this_month] * 10

        # create empty numpy frames for the following calculations:
        X_0, PHIm_tot_0, Tm_0, Ts_0, Tair_0, Top_0, X_10, PHIm_tot_10, Tm_10, \
        Ts_10, Tair_10, Top_10, X_HC, PHIm_tot_HC, Tm_HC, Ts_HC, Tair_HC, Top_HC, PHIm, PHIst, PHIia, Qsol, Q_H_RC, \
        Q_C_RC, Q_H_EB, Q_C_EB = create_matrix_before_day(daylist)
        # iterate through days in month:
        for day in range(days_this_month):
            # create empty numpy frames for the following calculations:
            Q_H_RC_day, Q_C_RC_day, Q_DHW_RC_day, Q_H_EB_day, Q_C_EB_day, Q_HC = create_matrix_after_day(daylist)
            # time_day_vector represents the index for the hours of one day (eg: 24-48 for second day in year)
            time_day_vector = np.arange(cum_hours + day * 24, cum_hours + (day + 1) * 24)

            # outdoor temperature
            Te = temp_8760[:, time_day_vector]

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
                # geTet desired Heating and Cooling Set Point Temperature
                Tset_h = Tset_heating_8760_up[UserProfile_idx, cum_hours + day * 24 + hour]

                # TODO vielleicht User Profile Einbauen?
                # estimate User behaviour: if outdoor temp drops, user will also reduce indoor temp slightly
                Tset_h = Tset_h + (Te[:, hour] - Tset_h + 12) / 8
                # TODO Cooling temp is not adjusted to outside temp?
                Tset_c = Tset_cooling_8760_up[UserProfile_idx, cum_hours + day * 24 + hour]

                # Hourly Losses DHW supply
                Qloss_DWH = Q_DHW_losses[:, cum_hours + day * 24 + hour]

                # solar gains through windows
                Qsol[:, hour] = Awindows_rad_north * sol_rad_n + Awindows_rad_east_west * sol_rad_ea + \
                                Awindows_rad_south * sol_rad_s

                # Gains Air Note [W]
                PHIia[:, hour] = 0.5 * (Qi + Qloss_DWH)  # Equ.C.1
                PHIm[:, hour] = Am / Atot * (0.5 * (Qi + Qloss_DWH) + Qsol[:, hour])  # Equ.C.2
                PHIst[:, hour] = (1 - Am / Atot - Htr_w / (hms * Atot)) * (
                        0.5 * (Qi + Qloss_DWH) + Qsol[:, hour])  # Equ.C.3

                # supply air temperature equal to outdoor temperature if no heat recovery system is considered
                # TODO implement some households that have heat recovery
                T_air_supply = 1 * Te[:, hour] + 0 * Tair_HC[:, prev_hour]
                # Bestimmung der Lufttemperatur: Kapitel C.3
                # X_0 ist variable um Gleichung C.5 auf zu teilen (große Klammer)
                X_0[:, hour] = (PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 *
                                (((PHIia[:, hour] + PHIHC_nd[:, day * 24 + hour]) / Hve) + T_air_supply))
                PHIm_tot_0[:, hour] = PHIm[:, hour] + Htr_em * Te[:, hour] + Htr_3 * X_0[:, hour] / Htr_2  # Equ. C.5

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
                        T_air_supply + (PHIia[:, hour] + PHIHC_nd10[:, day * 24 + hour]) / Hve)) / (
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

                # Schritt 4:
                X_HC[:, hour] = PHIst[:, hour] + Htr_w * Te[:, hour] + Htr_1 * \
                                (((PHIia[:, hour] + Q_HC[:, hour]) / Hve) + T_air_supply)  # part of Equ. C.5
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

                # END day
            # save results in output vektor (cooling and heating are seperated)
            Q_H_LOAD_8760[:, time_day_vector] = Q_H_RC_day
            Q_C_LOAD_8760[:, time_day_vector] = Q_C_RC_day
            Q_H_LOAD_8760[:, time_day_vector] = np.maximum(0, Q_HC)
            Q_C_LOAD_8760[:, time_day_vector] = np.maximum(0, - Q_HC)

            T_op_HC_hourly[:, time_day_vector] = Top_HC
            T_air_HC_hourly[:, time_day_vector] = Tair_HC
            T_s_HC_hourly[:, time_day_vector] = Ts_HC
            T_m_HC_hourly[:, time_day_vector] = Tm_HC

            time_month_vector = np.arange(cum_hours, cum_hours + num_hours)

            # aufaddieren der Heizleistung auf ein monat TODO warum tageswert/monatstage ? wird ja in schleife gemacht
            Q_H_RC = Q_H_RC + Q_H_RC_day / days_this_month
            Q_C_RC = Q_C_RC + Q_C_RC_day / days_this_month

            Q_H_EB = Q_H_EB + Q_H_EB_day / days_this_month
            Q_C_EB = Q_C_EB + Q_C_EB_day / days_this_month

            # calculate heat days
            # Top_0_mean = Top_0.mean(axis=1)
            # TODO Andi fragen was dieser threshold bedeuted? (Bei mean temp über 26,28,30,32 +1 heat day? wird nicht gebraucht
            # heat_day_treshold = np.arange(26, 33, 2)
            # heat_days = np.zeros((num_bc, len(heat_day_treshold)))
            # WIRD NICHT GEBRAUCHT! Top_month und Tair_month nur für plot zur überprüfung, TODO heat days komplett unnötig?
            # heat_days_idx = (Top_0_mean * ones(1, length(heat_day_treshold))) > (ones(num_bc, 1) * heat_day_treshold)
            # heat_days(heat_days_idx) = heat_days(heat_days_idx) + 1
            # Top_month(:, (day - 1) * 24 + 1: day * 24) = Top_0;
            # Tair_month(:, (day - 1) * 24 + 1: day * 24) = Tair_0;

            # END MONTH
        # TODO warum /1000 ? Ist das nicht auch unnötig? ich brauch die 8760 werte..
        # Q_H_month_RC[:, month] = Q_H_RC.sum(axis=1) / 1000 * DpM[month]
        # Q_H_month_EB[:, month] = (Q_H_EB - np.tile(Qi, (24, 1)).T).sum(axis=1) / 1000 * DpM[month]
        #
        # Q_C_month_RC[:, month] = Q_C_RC.sum(axis=1) / 1000 * DpM[month]
        # Q_C_month_EB[:, month] = (Q_C_EB - np.tile(Qi, (24, 1)).T).sum(axis=1) / 1000 * DpM[month]

        cum_hours = cum_hours + num_hours
        # END YEAR

    return Q_H_LOAD_8760, Q_C_LOAD_8760, Q_DHW_LOAD_8760, Af


def rc_heating_cooling(Q_solar, Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Af, T_outside,
                       initial_thermal_mass_temp=20, T_air_min=20, T_air_max=28):
    # check if vectors are the same length:
    def check_length(*args):
        length_1 = len(args[0])
        for vector in args:
            if len(vector) != length_1:
                raise ValueError("Arrays must have the same size")

    check_length(Q_solar, T_outside)

    timesteps = np.arange(len(T_outside))
    At = 4.5  # 7.2.2.2

    # Kopplung Temp Luft mit Temp Surface Knoten s
    his = np.float_(3.45)  # 7.2.2.2
    # kopplung zwischen Masse und  zentralen Knoten s (surface)
    hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
    Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
    Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
    # thermischer Kopplungswerte W/K
    Htr_is = his * Atot
    Htr_1 = np.float_(1) / (np.float_(1) / Hve + np.float_(1) / Htr_is)  # Equ. C.6
    Htr_2 = Htr_1 + Htr_w  # Equ. C.7
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

    # Equ. C.1
    PHI_ia = 0.5 * Qi

    Tm_t = np.empty(shape=(len(timesteps), len(Hve)))
    T_sup = np.empty(shape=(len(timesteps),))
    Q_H_real = np.empty(shape=(len(timesteps), len(Hve)))
    Q_C_real = np.empty(shape=(len(timesteps), len(Hve)))
    T_air_real = np.empty(shape=(len(timesteps), len(Hve)))
    heating_power_10 = Af * 10
    for t in timesteps:  # t is the index for each timestep
        for i in range(len(Hve)):  # i is the index for each individual building

            # Equ. C.2
            PHI_m = Am[i] / Atot[i] * (0.5 * Qi[i] + Q_solar[t])
            # Equ. C.3
            PHI_st = (1 - Am[i] / Atot[i] - Htr_w[i] / 9.1 / Atot[i]) * (0.5 * Qi[i] + Q_solar[t])

            # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                    PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] * (((PHI_ia[i] + 0) / Hve[i]) + T_sup[t])) / \
                         Htr_2[i]

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                    PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] * (
                    ((PHI_ia[i] + heating_power_10[i]) / Hve[i]) + T_sup[t])) / Htr_2[i]

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                    PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] * (
                    ((PHI_ia[i] - heating_power_10[i]) / Hve[i]) + T_sup[t])) / Htr_2[i]

            if t == 0:
                Tm_t_prev = initial_thermal_mass_temp
            else:
                Tm_t_prev = Tm_t[t - 1, i]

            # Equ. C.4
            Tm_t_0 = (Tm_t_prev * (Cm[i] / 3600 - 0.5 * (Htr_3[i] + Htr_em[i])) + PHI_mtot_0) / \
                     (Cm[i] / 3600 + 0.5 * (Htr_3[i] + Htr_em[i]))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (Tm_t_prev * (Cm[i] / 3600 - 0.5 * (Htr_3[i] + Htr_em[i])) + PHI_mtot_10) / \
                      (Cm[i] / 3600 + 0.5 * (Htr_3[i] + Htr_em[i]))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (Tm_t_prev * (Cm[i] / 3600 - 0.5 * (Htr_3[i] + Htr_em[i])) + PHI_mtot_10_c) / \
                        (Cm[i] / 3600 + 0.5 * (Htr_3[i] + Htr_em[i]))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (Htr_ms[i] * T_m_0 + PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] *
                     (T_sup[t] + (PHI_ia[i] + 0) / Hve[i])) / (Htr_ms[i] + Htr_w[i] + Htr_1[i])

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (Htr_ms[i] * T_m_10 + PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] *
                      (T_sup[t] + (PHI_ia[i] + heating_power_10[i]) / Hve[i])) / (Htr_ms[i] + Htr_w[i] + Htr_1[i])

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (Htr_ms[i] * T_m_10_c + PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] *
                        (T_sup[t] + (PHI_ia[i] - heating_power_10[i]) / Hve[i])) / (Htr_ms[i] + Htr_w[i] + Htr_1[i])

            # Equ. C.11
            T_air_0 = (Htr_is[i] * T_s_0 + Hve[i] * T_sup[t] + PHI_ia[i] + 0) / \
                      (Htr_is[i] + Hve[i])

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (Htr_is[i] * T_s_10 + Hve[i] * T_sup[t] + PHI_ia[i] + heating_power_10[i]) / \
                       (Htr_is[i] + Hve[i])

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (Htr_is[i] * T_s_10_c + Hve[i] * T_sup[t] + PHI_ia[i] - heating_power_10[i]) / \
                         (Htr_is[i] + Hve[i])

            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0 >= T_air_min and T_air_0 <= T_air_max:
                Q_H_real[t, i] = 0
            elif T_air_0 < T_air_min:  # heating is required
                Q_H_real[t, i] = heating_power_10[i] * (T_air_min - T_air_0) / (T_air_10 - T_air_0)
            elif T_air_0 > T_air_max:  # cooling is required
                Q_C_real[t, i] = heating_power_10[i] * (T_air_max - T_air_0) / (T_air_10_c - T_air_0)

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                    PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] * (
                    ((PHI_ia[i] + Q_H_real[t, i] - Q_C_real[t, i]) / Hve[i]) + T_sup[t])) / Htr_2[i]
            # Equ. C.4
            Tm_t[t, i] = (Tm_t_prev * (Cm[i] / 3600 - 0.5 * (Htr_3[i] + Htr_em[i])) + PHI_mtot_real) / \
                         (Cm[i] / 3600 + 0.5 * (Htr_3[i] + Htr_em[i]))

            # Equ. C.9
            T_m_real = (Tm_t[t, i] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (Htr_ms[i] * T_m_real + PHI_st + Htr_w[i] * T_outside[t] + Htr_1[i] *
                       (T_sup[t] + (PHI_ia[i] + Q_H_real[t, i] - Q_C_real[t, i]) / Hve[i])) / \
                       (Htr_ms[i] + Htr_w[i] + Htr_1[i])

            # Equ. C.11 for 10 W/m^2 heating
            T_air_real[t, i] = (Htr_is[i] * T_s_real + Hve[i] * T_sup[t] + PHI_ia[i] + Q_H_real[t, i] - Q_C_real[t, i]) / \
                                (Htr_is[i] + Hve[i])

    # fill nan
    Q_C_real = np.nan_to_num(Q_C_real, nan=0)
    Q_H_real = np.nan_to_num(Q_H_real, nan=0)
    T_air_real = np.nan_to_num(T_air_real, nan=0)

    return Q_H_real, Tm_t, Q_C_real, T_air_real
