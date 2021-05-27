from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
import numpy as np
import pandas as pd
from pathlib import Path


class Building:

    def __init__(self, para_series):
        # self.ID_BuildingType = para_series["ID_BuildingType"]
        # self.ID_BuildingAgeClass = para_series["ID_BuildingAgeClass"]
        # self.FloorArea = para_series["FloorArea"]

        self.ID = para_series['ID']
        self.f1 = para_series['f1']
        self.index = para_series['index']
        self.name = para_series['name']
        self.construction_period_start = para_series['construction_period_start']
        self.construction_period_end = para_series['construction_period_end']
        self.building_categories_index = para_series['building_categories_index']
        self.number_of_dwellings_per_building = para_series['number_of_dwellings_per_building']
        self.horizontal_shading_building = para_series['horizontal_shading_building']  #
        self.areawindows = para_series['areawindows']
        self.area_suitable_solar = para_series['area_suitable_solar']
        self.ued_dhw = para_series['ued_dhw']
        self.average_effective_area_wind_west_east_red_cool \
            = para_series['average_effective_area_wind_west_east_red_cool']
        self.average_effective_area_wind_south_red_cool \
            = para_series['average_effective_area_wind_south_red_cool']
        self.average_effective_area_wind_north_red_cool \
            = para_series['average_effective_area_wind_north_red_cool']
        self.spec_int_gains_cool_watt = para_series['spec_int_gains_cool_watt']
        self.building_categories_index = para_series['building_categories_index']
        self.Af = para_series['Af']
        self.Hop = para_series['Hop']
        self.Htr_w = para_series['Htr_w']
        self.Hve = para_series['Hve']
        self.CM_factor = para_series['CM_factor']
        self.Am_factor = para_series['Am_factor']
        self.country_ID = para_series['country_ID']

        def calc_IndoorTemperature(self, Energy_TankToRoom_t, OutdoorTemperature_t):
            """
            :param Energy_TankToRoom_t: float
            :param OutdoorTemperature_t: float
            :return: IndoorTemperature_tp1: float
            """


class HeatingCooling_noDR:

    def __init__(self, ID_BuildingOption_dataframe):
        self.index = self.index['index']
        self.name = self.ID_Household['name']
        self.building_categories_index = self.ID_Household['building_categories_index']
        self.number_of_dwellings_per_building = self.ID_Household['number_of_dwellings_per_building']
        self.areawindows = self.ID_Household['areawindows']
        self.area_suitable_solar = self.ID_Household['area_suitable_solar']
        self.ued_dhw = self.ID_Household['ued_dhw']
        self.AreaWindowEastWest = self.ID_Household['average_effective_area_wind_west_east_red_cool'].to_numpy()
        self.AreaWindowSouth = self.ID_Household['average_effective_area_wind_south_red_cool'].to_numpy()
        self.AreaWindowNorth = self.ID_Household['average_effective_area_wind_north_red_cool'].to_numpy()
        self.InternalGains = self.ID_Household['spec_int_gains_cool_watt'].to_numpy()
        self.building_categories_index = self.ID_Household['building_categories_index']
        self.Af = self.ID_Household['Af'].to_numpy()
        self.Hop = self.ID_Household['Hop'].to_numpy()
        self.Htr_w = self.ID_Household['Htr_w'].to_numpy()
        self.Hve = self.ID_Household['Hve'].to_numpy()
        self.CM_factor = self.ID_Household['CM_factor'].to_numpy()
        self.Am_factor = self.ID_Household['Am_factor'].to_numpy()
        self.country_ID = self.ID_Household['country_ID'].to_numpy()


    def ref_HeatingCooling(self, T_outside, Q_solar=None,
                           initial_thermal_mass_temp=15, T_air_min=20, T_air_max=28):
        """
        This function calculates the heating and cooling demand as well as the indoor temperature for every building
        category based in the 5R1C model. The results are hourls vectors for one year. Q_solar is imported from a CSV
        at the time!
        """
        # check if vectors are the same length:
        def check_length(*args):
            length_1 = len(args[0])
            for vector in args:
                if len(vector) != length_1:
                    raise ValueError("Arrays must have the same size")
        project_directory_path = Path(__file__).parent.resolve()
        base_results_path = project_directory_path / "inputdata"
        check_length(Q_solar, T_outside)
        Q_sol_all = pd.read_csv(base_results_path / "directRadiation_himmelsrichtung_GER.csv", sep=";")
        Q_sol_north = np.outer(Q_sol_all.loc[:, "RadiationNorth"].to_numpy(), self.AreaWindowNorth)
        Q_sol_south = np.outer(Q_sol_all.loc[:, "RadiationSouth"].to_numpy(), self.AreaWindowSouth)
        Q_sol_east_west = np.outer((Q_sol_all.loc[:, "RadiationEast"].to_numpy() +
                                    Q_sol_all.loc[:, "RadiationWest"].to_numpy()), self.AreaWindowEastWest)
        Q_solar = Q_sol_north + Q_sol_south + Q_sol_east_west
        # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
        Atot = 4.5 * self.Af
        # Speicherkapazität J/K
        Cm = self.CM_factor * self.Af
        # wirksame Massenbezogene Fläche [m^2]
        Am = self.Am_factor * self.Af
        # internal gains
        Q_InternalGains = self.InternalGains * self.Af
        timesteps = np.arange(len(T_outside))

        # Kopplung Temp Luft mit Temp Surface Knoten s
        his = np.float_(3.45)  # 7.2.2.2
        # kopplung zwischen Masse und  zentralen Knoten s (surface)
        hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / self.Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte W/K
        Htr_is = his * Atot
        Htr_1 = np.float_(1) / (np.float_(1) / self.Hve + np.float_(1) / Htr_is)  # Equ. C.6
        Htr_2 = Htr_1 + self.Htr_w  # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

        # Equ. C.1
        PHI_ia = 0.5 * Q_InternalGains

        Tm_t = np.empty(shape=(len(timesteps), len(self.Hve)))
        T_sup = np.empty(shape=(len(timesteps),))
        Q_Heating_noDR = np.empty(shape=(len(timesteps), len(self.Hve)))
        Q_Cooling_noDR = np.empty(shape=(len(timesteps), len(self.Hve)))
        T_Room_noDR = np.empty(shape=(len(timesteps), len(self.Hve)))
        heating_power_10 = self.Af * 10
        for t in timesteps:  # t is the index for each timestep
            for i in range(len(self.Hve)):  # i is the index for each individual building
                # Equ. C.2
                PHI_m = Am[i] / Atot[i] * (0.5 * Q_InternalGains[i] + Q_solar[t, i])
                # Equ. C.3
                PHI_st = (1 - Am[i] / Atot[i] - self.Htr_w[i] / 9.1 / Atot[i]) * \
                         (0.5 * Q_InternalGains[i] + Q_solar[t, i])

                # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
                T_sup[t] = T_outside[t]

                # Equ. C.5
                PHI_mtot_0 = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                        PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] * (((PHI_ia[i] + 0) / self.Hve[i]) + T_sup[t])) / \
                             Htr_2[i]

                # Equ. C.5 with 10 W/m^2 heating power
                PHI_mtot_10 = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                        PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] * (
                        ((PHI_ia[i] + heating_power_10[i]) / self.Hve[i]) + T_sup[t])) / Htr_2[i]

                # Equ. C.5 with 10 W/m^2 cooling power
                PHI_mtot_10_c = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                        PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] * (
                        ((PHI_ia[i] - heating_power_10[i]) / self.Hve[i]) + T_sup[t])) / Htr_2[i]

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
                T_s_0 = (Htr_ms[i] * T_m_0 + PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] *
                         (T_sup[t] + (PHI_ia[i] + 0) / self.Hve[i])) / (Htr_ms[i] + self.Htr_w[i] + Htr_1[i])

                # Euq. C.10 for 10 W/m^2 heating
                T_s_10 = (Htr_ms[i] * T_m_10 + PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] *
                          (T_sup[t] + (PHI_ia[i] + heating_power_10[i]) / self.Hve[i])) / (Htr_ms[i] + self.Htr_w[i] + Htr_1[i])

                # Euq. C.10 for 10 W/m^2 cooling
                T_s_10_c = (Htr_ms[i] * T_m_10_c + PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] *
                            (T_sup[t] + (PHI_ia[i] - heating_power_10[i]) / self.Hve[i])) / (Htr_ms[i] + self.Htr_w[i] + Htr_1[i])

                # Equ. C.11
                T_air_0 = (Htr_is[i] * T_s_0 + self.Hve[i] * T_sup[t] + PHI_ia[i] + 0) / \
                          (Htr_is[i] + self.Hve[i])

                # Equ. C.11 for 10 W/m^2 heating
                T_air_10 = (Htr_is[i] * T_s_10 + self.Hve[i] * T_sup[t] + PHI_ia[i] + heating_power_10[i]) / \
                           (Htr_is[i] + self.Hve[i])

                # Equ. C.11 for 10 W/m^2 cooling
                T_air_10_c = (Htr_is[i] * T_s_10_c + self.Hve[i] * T_sup[t] + PHI_ia[i] - heating_power_10[i]) / \
                             (Htr_is[i] + self.Hve[i])

                # Check if air temperature without heating is in between boundaries and calculate actual HC power:
                if T_air_0 >= T_air_min and T_air_0 <= T_air_max:
                    Q_Heating_noDR[t, i] = 0
                elif T_air_0 < T_air_min:  # heating is required
                    Q_Heating_noDR[t, i] = heating_power_10[i] * (T_air_min - T_air_0) / (T_air_10 - T_air_0)
                elif T_air_0 > T_air_max:  # cooling is required
                    Q_Cooling_noDR[t, i] = heating_power_10[i] * (T_air_max - T_air_0) / (T_air_10_c - T_air_0)

                # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
                # Equ. C.5 with actual heating power
                PHI_mtot_real = PHI_m + Htr_em[i] * T_outside[t] + Htr_3[i] * (
                        PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] * (
                        ((PHI_ia[i] + Q_Heating_noDR[t, i] - Q_Cooling_noDR[t, i]) / self.Hve[i]) + T_sup[t])) / Htr_2[i]
                # Equ. C.4
                Tm_t[t, i] = (Tm_t_prev * (Cm[i] / 3600 - 0.5 * (Htr_3[i] + Htr_em[i])) + PHI_mtot_real) / \
                             (Cm[i] / 3600 + 0.5 * (Htr_3[i] + Htr_em[i]))

                # Equ. C.9
                T_m_real = (Tm_t[t, i] + Tm_t_prev) / 2

                # Euq. C.10
                T_s_real = (Htr_ms[i] * T_m_real + PHI_st + self.Htr_w[i] * T_outside[t] + Htr_1[i] *
                            (T_sup[t] + (PHI_ia[i] + Q_Heating_noDR[t, i] - Q_Cooling_noDR[t, i]) / self.Hve[i])) / \
                           (Htr_ms[i] + self.Htr_w[i] + Htr_1[i])

                # Equ. C.11 for 10 W/m^2 heating
                T_Room_noDR[t, i] = (Htr_is[i] * T_s_real + self.Hve[i] * T_sup[t] + PHI_ia[i] +
                                     Q_Heating_noDR[t, i] - Q_Cooling_noDR[t, i]) / (Htr_is[i] + self.Hve[i])

        # fill nan
        Q_Cooling_noDR = np.nan_to_num(Q_Cooling_noDR, nan=0)
        Q_Heating_noDR = np.nan_to_num(Q_Heating_noDR, nan=0)
        T_Room_noDR = np.nan_to_num(T_Room_noDR, nan=0)

        return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR
