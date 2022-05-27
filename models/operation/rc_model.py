import numpy as np
from models.operation.scenario import OperationScenario


class R5C1Model:

    def __init__(self, scenario: 'OperationScenario'):
        self.scenario = scenario
        self.building = scenario.building
        self.init_rc_params()

    def init_rc_params(self):
        # TODO: add the norm name and link, so the users can find the equations following their numbers.
        # 7.2.2.2: Area of all surfaces facing the building zone
        self.Atot = 4.5 * self.building.Af
        self.Cm = self.building.CM_factor * self.building.Af
        # Effective mass related area [m^2]
        self.Am = self.building.Am_factor * self.building.Af
        # internal gains
        self.Qi = self.building.internal_gains * self.building.Af
        # Coupling Temp Air with Temp Surface Node s
        self.his = np.float_(3.45)  # 7.2.2.2
        # coupling between mass and central node s (surface)
        self.hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
        self.Htr_ms = self.hms * self.Am  # from 12.2.2 Equ. (64)
        self.Htr_em = 1 / (1 / self.building.Hop - 1 / self.Htr_ms)  # from 12.2.2 Equ. (63)
        # Thermal coupling value W/K
        self.Htr_is = self.his * self.Atot
        self.Htr_1 = np.float_(1) / (np.float_(1) / self.building.Hve + np.float_(1) / self.Htr_is)  # Equ. C.6
        self.Htr_2 = self.Htr_1 + self.building.Htr_w  # Equ. C.7
        self.Htr_3 = 1 / (1 / self.Htr_2 + 1 / self.Htr_ms)  # Equ.C.8
        # Equ. C.1
        self.PHI_ia = 0.5 * self.Qi
        self.Q_solar = self.calculate_solar_gains()

    def calculate_solar_gains(self) -> np.array:
        """return: 8760h solar gains, calculated with solar radiation and the effective window area."""
        area_window_east_west = self.building.effective_window_area_west_east
        area_window_south = self.building.effective_window_area_south
        area_window_north = self.building.effective_window_area_north

        Q_solar_north = np.outer(np.array(self.scenario.region.radiation_north), area_window_north)
        Q_solar_east = np.outer(np.array(self.scenario.region.radiation_east), area_window_east_west / 2)
        Q_solar_south = np.outer(np.array(self.scenario.region.radiation_south), area_window_south)
        Q_solar_west = np.outer(np.array(self.scenario.region.radiation_west), area_window_east_west / 2)
        Q_solar = ((Q_solar_north + Q_solar_south + Q_solar_east + Q_solar_west).squeeze())
        return Q_solar

    def calculate_heating_and_cooling_demand(self,
                                             thermal_start_temperature: float = 15,
                                             static=False) -> (np.array, np.array, np.array, np.array):
        """
        if "static" is True, then the RC model will calculate a static heat demand calculation for the first hour of
        the year by using this hour 100 times. This way a good approximation of the thermal mass temperature of the
        building in the beginning of the calculation is achieved. Solar gains are set to 0.
        Returns: heating demand, cooling demand, indoor air temperature, temperature of the thermal mass
        """
        heating_power_10 = self.building.Af * 10

        if self.scenario.space_cooling_technology.power == 0:
            T_air_max = np.full((8760,), 100)  # if no cooling is adopted --> raise max air temperature to 100 so it will never cool:
        else:
            T_air_max = self.scenario.behavior.target_temperature_array_max

        if static:
            Q_solar = np.array([0] * 100)
            T_outside = np.array([self.scenario.region.temperature[0]] * 100)
            T_air_min = np.array([self.scenario.behavior.target_temperature_array_min[0]] * 100)
            time = np.arange(100)

            Tm_t = np.zeros(shape=(100,))  # thermal mass temperature
            T_sup = np.zeros(shape=(100,))
            heating_demand = np.zeros(shape=(100,))
            cooling_demand = np.zeros(shape=(100,))
            room_temperature = np.zeros(shape=(100,))

        else:
            Q_solar = self.Q_solar
            T_outside = self.scenario.region.temperature
            T_air_min = self.scenario.behavior.target_temperature_array_min
            time = np.arange(8760)

            Tm_t = np.zeros(shape=(8760,))  # thermal mass temperature
            T_sup = np.zeros(shape=(8760,))
            heating_demand = np.zeros(shape=(8760,))
            cooling_demand = np.zeros(shape=(8760,))
            room_temperature = np.zeros(shape=(8760,))

        # RC-Model
        for t in time:  # t is the index for each time step
            # Equ. C.2
            PHI_m = self.Am / self.Atot * (0.5 * self.Qi + Q_solar[t])
            # Equ. C.3
            PHI_st = (1 - self.Am / self.Atot - self.building.Htr_w / 9.1 / self.Atot) * \
                     (0.5 * self.Qi + Q_solar[t])

            # (T_sup = T_outside because incoming air is not preheated)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = PHI_m + self.Htr_em * T_outside[t] + self.Htr_3 * (
                    PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 * (((self.PHI_ia + 0) / self.building.Hve) + T_sup[t])
            ) / self.Htr_2

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = PHI_m + self.Htr_em * T_outside[t] + self.Htr_3 * (
                    PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 * (
                    ((self.PHI_ia + heating_power_10) / self.building.Hve) + T_sup[t])) / self.Htr_2

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = PHI_m + self.Htr_em * T_outside[t] + self.Htr_3 * (
                    PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 * (
                    ((self.PHI_ia - heating_power_10) / self.building.Hve) + T_sup[t])) / self.Htr_2

            if t == 0:
                Tm_t_prev = thermal_start_temperature
            else:
                Tm_t_prev = Tm_t[t - 1]

            # Equ. C.4
            Tm_t_0 = (Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot_0) / \
                     (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot_10) / \
                      (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot_10_c) / \
                        (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (self.Htr_ms * T_m_0 + PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 *
                     (T_sup[t] + (self.PHI_ia + 0) / self.building.Hve)) / (self.Htr_ms + self.building.Htr_w + self.Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (self.Htr_ms * T_m_10 + PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 *
                      (T_sup[t] + (self.PHI_ia + heating_power_10) / self.building.Hve)) / (
                                 self.Htr_ms + self.building.Htr_w + self.Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (self.Htr_ms * T_m_10_c + PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 *
                        (T_sup[t] + (self.PHI_ia - heating_power_10) / self.building.Hve)) / (
                                   self.Htr_ms + self.building.Htr_w + self.Htr_1)

            # Equ. C.11
            T_air_0 = (self.Htr_is * T_s_0 + self.building.Hve * T_sup[t] + self.PHI_ia + 0) / \
                      (self.Htr_is + self.building.Hve)

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (self.Htr_is * T_s_10 + self.building.Hve * T_sup[t] + self.PHI_ia + heating_power_10) / \
                       (self.Htr_is + self.building.Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (self.Htr_is * T_s_10_c + self.building.Hve * T_sup[t] + self.PHI_ia - heating_power_10) / \
                         (self.Htr_is + self.building.Hve)


            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0 >= T_air_min[t] and T_air_0 <= T_air_max[t]:
                heating_demand[t] = 0
            elif T_air_0 < T_air_min[t]:  # heating is required
                heating_demand[t] = heating_power_10 * (T_air_min[t] - T_air_0) / (
                        T_air_10 - T_air_0)
            elif T_air_0 > T_air_max[t]:  # cooling is required
                cooling_demand[t] = heating_power_10 * (T_air_max[t] - T_air_0) / (
                        T_air_10_c - T_air_0)

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = PHI_m + self.Htr_em * T_outside[t] + self.Htr_3 * (
                    PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 * (
                    ((self.PHI_ia + heating_demand[t] - cooling_demand[t]) / self.building.Hve) + T_sup[t])) / self.Htr_2
            # Equ. C.4
            Tm_t[t] = (Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot_real) / \
                         (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (self.Htr_ms * T_m_real + PHI_st + self.building.Htr_w * T_outside[t] + self.Htr_1 *
                        (T_sup[t] + (self.PHI_ia + heating_demand[t] - cooling_demand[t]) / self.building.Hve)) / \
                       (self.Htr_ms + self.building.Htr_w + self.Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            room_temperature[t] = (self.Htr_is * T_s_real + self.building.Hve * T_sup[t] + self.PHI_ia +
                                   heating_demand[t] - cooling_demand[t]) / (self.Htr_is + self.building.Hve)

        return heating_demand, cooling_demand, room_temperature, Tm_t


if __name__ == "__main__":
    scenario = OperationScenario(scenario_id=0)
    RC_model = R5C1Model(scenario)
    heating, cooling, air_temp, mass_temp = RC_model.calculate_heating_and_cooling_demand()

