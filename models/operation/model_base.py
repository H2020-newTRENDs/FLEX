from abc import ABC

import numpy as np
from typing import Union
from models.operation.scenario import OperationScenario


class OperationModel(ABC):

    def __init__(self, scenario: "OperationScenario"):
        self.scenario = scenario
        self.CPWater = 4200 / 3600
        self.setup_operation_model_params()

    def setup_operation_model_params(self):
        self.setup_time_params()
        self.setup_region_params()
        self.setup_building_params()
        self.setup_space_heating_params()
        self.setup_hot_water_params()
        self.setup_space_cooling_params()
        self.setup_pv_params()
        self.setup_battery_params()
        self.setup_ev_params()
        self.setup_energy_price_params()
        self.setup_behavior_params()
        self.setup_heating_element_params()

    def setup_time_params(self):
        self.Hour = np.arange(1, 8761)
        self.DayHour = np.tile(np.arange(1, 25), 365)

    def setup_region_params(self):
        self.T_outside = self.scenario.region.temperature  # °C
        self.Q_Solar = self.calculate_solar_gain()  # W

    def setup_building_params(self):
        """
        The thermal dynamics of building is modeled following...
        provide the link, so the equation numbers can make sense
        """
        self.Am = (self.scenario.building.Am_factor * self.scenario.building.Af)  # Effective mass related area [m^2]
        self.Cm = self.scenario.building.CM_factor * self.scenario.building.Af
        self.Atot = (4.5 * self.scenario.building.Af)  # 7.2.2.2: Area of all surfaces facing the building zone
        self.Qi = self.scenario.building.internal_gains * self.scenario.building.Af
        self.Htr_w = self.scenario.building.Htr_w
        self.Htr_ms = np.float_(9.1) * self.Am  # from 12.2.2 Equ. (64)
        self.Htr_is = np.float_(3.45) * self.Atot
        self.Htr_em = 1 / (1 / self.scenario.building.Hop - 1 / self.Htr_ms)  # from 12.2.2 Equ. (63)
        self.Htr_1 = np.float_(1) / (np.float_(1) / self.scenario.building.Hve + np.float_(1) / self.Htr_is)  # Equ. C.6
        self.Htr_2 = self.Htr_1 + self.scenario.building.Htr_w  # Equ. C.7
        self.Htr_3 = 1 / (1 / self.Htr_2 + 1 / self.Htr_ms)  # Equ.C.8
        self.Hve = self.scenario.building.Hve
        self.PHI_ia = 0.5 * self.Qi  # Equ. C.1
        _, _, _, mass_temperature = self.calculate_heating_and_cooling_demand(static=True)
        self.BuildingMassTemperatureStartValue = mass_temperature[-1]

        self.Q_RoomHeating, self.Q_RoomCooling, self.T_Room, self.T_BuildingMass = \
            self.calculate_heating_and_cooling_demand(thermal_start_temperature=self.BuildingMassTemperatureStartValue,
                                                      static=False)

    def setup_space_heating_params(self):
        self.fuel_boiler_efficiency = self.scenario.boiler.fuel_boiler_efficiency
        self.SpaceHeatingHourlyCOP = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.building.supply_temperature,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type,
        )
        self.SpaceHeatingHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.building.supply_temperature + 10,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type,
        )
        self.T_TankStart_heating = self.scenario.space_heating_tank.temperature_start
        self.U_LossTank_heating = self.scenario.space_heating_tank.loss
        self.T_TankSurrounding_heating = (self.scenario.space_heating_tank.temperature_surrounding)
        self.SpaceHeating_MaxBoilerPower = self.generate_maximum_electric_or_thermal_power()
        self.Q_TankEnergyMin_heating = self.CPWater * self.scenario.space_heating_tank.size * \
                                       (273.15 + self.scenario.space_heating_tank.temperature_min)
        self.Q_TankEnergyMax_heating = self.CPWater * self.scenario.space_heating_tank.size * \
                                       (273.15 + self.scenario.space_heating_tank.temperature_max)
        self.M_WaterTank_heating = self.setup_space_heating_tank_size()
        self.A_SurfaceTank_heating = self.calculate_surface_area_from_volume(self.scenario.space_heating_tank.size)


    def setup_space_heating_tank_size(self):
        # Rule: 30L per kW thermal power output
        thermal_power = self.SpaceHeating_MaxBoilerPower * self.worst_COP
        # overwrite whatever was given as input
        tanksize = thermal_power / 1_000 * 30
        self.scenario.space_heating_tank.size = tanksize[0]
        return tanksize[0]

    def setup_heating_element_params(self):
        self.HeatingElement_efficiency = self.scenario.heating_element.efficiency
        self.HeatingElement_power = self.scenario.heating_element.power

    def setup_hot_water_params(self):
        self.HotWaterHourlyCOP = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.building.supply_temperature,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type,
        )
        self.HotWaterHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.building.supply_temperature + 10,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type,
        )
        self.T_TankStart_DHW = self.scenario.hot_water_tank.temperature_start
        self.M_WaterTank_DHW = self.scenario.hot_water_tank.size
        self.U_LossTank_DHW = self.scenario.hot_water_tank.loss
        self.T_TankSurrounding_DHW = self.scenario.hot_water_tank.temperature_surrounding
        self.A_SurfaceTank_DHW = self.calculate_surface_area_from_volume(self.scenario.hot_water_tank.size)
        self.Q_TankEnergyMin_DHW = self.CPWater * self.scenario.hot_water_tank.size * \
                                   (273.15 + self.scenario.hot_water_tank.temperature_min)
        self.Q_TankEnergyMax_DHW = self.CPWater * self.scenario.hot_water_tank.size * \
                                   (273.15 + self.scenario.hot_water_tank.temperature_max)

    def calculate_surface_area_from_volume(self, volume: float) -> float:
        # V = h * pi * r2 -> h = V/(r2*pi), A = 2r*pi*h + 2pi*r2 -> A = 2r*pi*V/(r2*pi) + 2pi*r2 = 2V/r + 2pi*r2
        # derivative of area: 0 = 4pi*r - 4V/r2  -> V = pi*r3  -> r = sqrt^3(V/pi)
        if volume == 0:
            return 0
        V = volume / 1_000  # from l into m^3
        r = np.cbrt(V / np.pi)
        h = V / (r ** 2 * np.pi)
        return np.round(2 * r * np.pi * h + 2 * np.pi * r ** 2, 2)

    def setup_space_cooling_params(self):
        self.CoolingCOP = self.scenario.space_cooling_technology.efficiency
        self.CoolingHourlyCOP = (np.ones(8760, ) * self.CoolingCOP)
        self.scenario.space_cooling_technology.power = self.calculate_max_cooling_power()

    def calculate_max_cooling_power(self):
        if self.scenario.space_cooling_technology.power > 0:
            # calculate the cooling demand in reference mode:
            max_cooling_demand = self.Q_RoomCooling.max()
            # round to the next 500 W
            max_cooling_power = np.ceil(max_cooling_demand / 500) * 500

            return max_cooling_power
        else:
            return 0
        
    def setup_pv_params(self):
        self.PhotovoltaicProfile = self.scenario.pv.generation

    def setup_battery_params(self):
        self.ChargeEfficiency = self.scenario.battery.charge_efficiency
        self.DischargeEfficiency = self.scenario.battery.discharge_efficiency

    def setup_ev_params(self):
        self.EVDemandProfile = self.scenario.behavior.vehicle_demand
        self.EVAtHomeProfile = self.scenario.behavior.vehicle_at_home
        self.EVChargeEfficiency = 0
        self.EVDischargeEfficiency = 0
        self.EVOptionV2B = 0
        self.EVcapacity = 0

        if self.scenario.vehicle.capacity > 0:
            self.test_vehicle_profile()  # test if vehicle makes model infeasible

    def setup_energy_price_params(self):
        self.ElectricityPrice = self.scenario.energy_price.electricity  # C/Wh
        self.FiT = self.scenario.energy_price.electricity_feed_in  # C/Wh

    def setup_behavior_params(self):
        self.HotWaterProfile = self.scenario.behavior.hot_water_demand
        self.BaseLoadProfile = self.scenario.behavior.appliance_electricity_demand

    def calculate_heating_and_cooling_demand(
            self, thermal_start_temperature: float = 15, static=False
    ) -> (np.array, np.array, np.array, np.array):
        """
        if "static" is True, then the RC model will calculate a static heat demand calculation for the first hour of
        the year by using this hour 100 times. This way a good approximation of the thermal mass temperature of the
        building in the beginning of the calculation is achieved. Solar gains are set to 0.
        Returns: heating demand, cooling demand, indoor air temperature, temperature of the thermal mass
        """
        heating_power_10 = self.scenario.building.Af * 10

        if self.scenario.space_cooling_technology.power == 0:
            T_air_max = np.full((8760,), 100)
            # if no cooling is adopted --> raise max air temperature to 100 so it will never cool:
        else:
            T_air_max = self.scenario.behavior.target_temperature_array_max

        if static:
            Q_solar = np.array([0] * 100)
            T_outside = np.array([self.scenario.region.temperature[0]] * 100)
            T_air_min = np.array(
                [self.scenario.behavior.target_temperature_array_min[0]] * 100
            )
            time = np.arange(100)

            Tm_t = np.zeros(shape=(100,))  # thermal mass temperature
            T_sup = np.zeros(shape=(100,))
            heating_demand = np.zeros(shape=(100,))
            cooling_demand = np.zeros(shape=(100,))
            room_temperature = np.zeros(shape=(100,))

        else:
            Q_solar = self.Q_Solar
            T_outside = self.T_outside
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
            PHI_st = (1 - self.Am / self.Atot - self.Htr_w / 9.1 / self.Atot) * (
                    0.5 * self.Qi + Q_solar[t]
            )

            # (T_sup = T_outside because incoming air is not preheated)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = (
                    PHI_m
                    + self.Htr_em * T_outside[t]
                    + self.Htr_3
                    * (
                            PHI_st
                            + self.Htr_w * T_outside[t]
                            + self.Htr_1 * (((self.PHI_ia + 0) / self.Hve) + T_sup[t])
                    )
                    / self.Htr_2
            )

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = (
                    PHI_m
                    + self.Htr_em * T_outside[t]
                    + self.Htr_3
                    * (
                            PHI_st
                            + self.Htr_w * T_outside[t]
                            + self.Htr_1
                            * (((self.PHI_ia + heating_power_10) / self.Hve) + T_sup[t])
                    )
                    / self.Htr_2
            )

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = (
                    PHI_m
                    + self.Htr_em * T_outside[t]
                    + self.Htr_3
                    * (
                            PHI_st
                            + self.Htr_w * T_outside[t]
                            + self.Htr_1
                            * (((self.PHI_ia - heating_power_10) / self.Hve) + T_sup[t])
                    )
                    / self.Htr_2
            )

            if t == 0:
                Tm_t_prev = thermal_start_temperature
            else:
                Tm_t_prev = Tm_t[t - 1]

            # Equ. C.4
            Tm_t_0 = (
                             Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em))
                             + PHI_mtot_0
                     ) / (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (
                              Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em))
                              + PHI_mtot_10
                      ) / (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (
                                Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em))
                                + PHI_mtot_10_c
                        ) / (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (
                            self.Htr_ms * T_m_0
                            + PHI_st
                            + self.Htr_w * T_outside[t]
                            + self.Htr_1 * (T_sup[t] + (self.PHI_ia + 0) / self.Hve)
                    ) / (self.Htr_ms + self.Htr_w + self.Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (
                             self.Htr_ms * T_m_10
                             + PHI_st
                             + self.Htr_w * T_outside[t]
                             + self.Htr_1 * (T_sup[t] + (self.PHI_ia + heating_power_10) / self.Hve)
                     ) / (self.Htr_ms + self.Htr_w + self.Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (
                               self.Htr_ms * T_m_10_c
                               + PHI_st
                               + self.Htr_w * T_outside[t]
                               + self.Htr_1 * (T_sup[t] + (self.PHI_ia - heating_power_10) / self.Hve)
                       ) / (self.Htr_ms + self.Htr_w + self.Htr_1)

            # Equ. C.11
            T_air_0 = (self.Htr_is * T_s_0 + self.Hve * T_sup[t] + self.PHI_ia + 0) / (
                    self.Htr_is + self.Hve
            )

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (
                               self.Htr_is * T_s_10
                               + self.Hve * T_sup[t]
                               + self.PHI_ia
                               + heating_power_10
                       ) / (self.Htr_is + self.Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (
                                 self.Htr_is * T_s_10_c
                                 + self.Hve * T_sup[t]
                                 + self.PHI_ia
                                 - heating_power_10
                         ) / (self.Htr_is + self.Hve)

            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0 >= T_air_min[t] and T_air_0 <= T_air_max[t]:
                heating_demand[t] = 0
            elif T_air_0 < T_air_min[t]:  # heating is required
                heating_demand[t] = (
                        heating_power_10 * (T_air_min[t] - T_air_0) / (T_air_10 - T_air_0)
                )
            elif T_air_0 > T_air_max[t]:  # cooling is required
                cooling_demand[t] = (
                        heating_power_10 * (T_air_max[t] - T_air_0) / (T_air_10_c - T_air_0)
                )

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = (
                    PHI_m
                    + self.Htr_em * T_outside[t]
                    + self.Htr_3
                    * (
                            PHI_st
                            + self.Htr_w * T_outside[t]
                            + self.Htr_1
                            * (
                                    (
                                            (self.PHI_ia + heating_demand[t] - cooling_demand[t])
                                            / self.Hve
                                    )
                                    + T_sup[t]
                            )
                    )
                    / self.Htr_2
            )
            # Equ. C.4
            Tm_t[t] = (
                              Tm_t_prev * (self.Cm / 3600 - 0.5 * (self.Htr_3 + self.Htr_em))
                              + PHI_mtot_real
                      ) / (self.Cm / 3600 + 0.5 * (self.Htr_3 + self.Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (
                               self.Htr_ms * T_m_real
                               + PHI_st
                               + self.Htr_w * T_outside[t]
                               + self.Htr_1
                               * (
                                       T_sup[t]
                                       + (self.PHI_ia + heating_demand[t] - cooling_demand[t]) / self.Hve
                               )
                       ) / (self.Htr_ms + self.Htr_w + self.Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            room_temperature[t] = (
                                          self.Htr_is * T_s_real
                                          + self.Hve * T_sup[t]
                                          + self.PHI_ia
                                          + heating_demand[t]
                                          - cooling_demand[t]
                                  ) / (self.Htr_is + self.Hve)

        return heating_demand, cooling_demand, room_temperature, Tm_t

    def generate_target_indoor_temperature(self, temperature_offset: Union[int, float]) -> (np.array, np.array):
        """
        This function modifies the exogenous target temperature range, so that
        1. the building will not be pre-heated (or pre-cooled) to too-high (or too-low) temperature;
        2. the optimization will not be infeasible if the building is heated above maximum temperature by radiation.

        Args:
            temperature_offset: int or float, temperature the optimization can pre-heat or pre-cool the building

        Returns: array of the maximum temperature, array of minimum temperature
        """
        max_temperature_list = []
        min_temperature_list = []
        for i, indoor_temp in enumerate(self.T_Room):
            temperature_max_winter = self.scenario.behavior.target_temperature_array_min[i] + temperature_offset
            if indoor_temp < temperature_max_winter:
                min_temperature_list.append(self.scenario.behavior.target_temperature_array_min[i])
                max_temperature_list.append(indoor_temp + temperature_offset)
            elif temperature_max_winter <= indoor_temp <= self.scenario.behavior.target_temperature_array_max[
                i] + 0.01:  # 0.01 is added to ignore floating point errors
                if self.scenario.space_cooling_technology.power == 0:
                    max_temperature_list.append(indoor_temp + temperature_offset)
                    min_temperature_list.append(indoor_temp - temperature_offset)
                else:
                    max_temperature_list.append(self.scenario.behavior.target_temperature_array_max[i])
                    min_temperature_list.append(indoor_temp - temperature_offset)
            else:  # if no cooling than temperature can be higher than max:
                max_temperature_list.append(indoor_temp + temperature_offset)
                min_temperature_list.append(indoor_temp - temperature_offset)
        return np.array(max_temperature_list), np.array(
            min_temperature_list)

    @staticmethod
    def calc_cop(
            outside_temperature: np.array,
            supply_temperature: float,
            efficiency: float,
            source: str,
    ) -> np.array:
        """
        Args:
            outside_temperature:
            supply_temperature: temperature that the heating system needs (eg. 35 low temperature, 65 hot)
            efficiency: carnot efficiency factor
            source: refers to the heat source (air, ground...) is saved in the boiler class as "name"

        Returns: numpy array of the hourly COP
        """
        if source == "Air_HP":
            COP = np.array(
                [
                    efficiency
                    * (supply_temperature + 273.15)
                    / (supply_temperature - temp)
                    for temp in outside_temperature
                ]
            )
        elif source == "Ground_HP":
            # for the ground source heat pump the temperature of the source is 10°C
            COP = np.array(
                [
                    efficiency
                    * (supply_temperature + 273.15)
                    / (supply_temperature - 10)
                    for temp in outside_temperature
                ]
            )
        elif source == "Electric":
            COP = np.full(8760, 1)
        else:
            # for fuel-based boiler, we initialize the COP with "Air_HP"
            COP = np.array(
                [
                    efficiency
                    * (supply_temperature + 273.15)
                    / (supply_temperature - temp)
                    for temp in outside_temperature
                ]
            )
        # check maximum COP, COP should not go to infinity for high outside temperatures
        COP[COP > 18] = 18
        # check minimum COP, COP can not be lower than 1
        COP[COP < 1] = 1
        return COP

    def generate_maximum_electric_or_thermal_power(self) -> float:
        """
        Calculates the necessary HP power for each building through the 5R1C reference model.
        The maximum heating power then will be rounded to the next 500 W.
        Then we divide the thermal power by the worst COP of the HP which is calculated at design conditions
        (coldest day) and the respective source and supply temperature. For conventional boilers the thermal power
        is provided as output
        """
        # calculate the heating demand in reference mode:
        max_heating_demand = self.Q_RoomHeating.max()
        max_dhw_demand = self.scenario.behavior.hot_water_demand.max()
        # round to the next 500 W
        max_thermal_power = np.ceil((max_heating_demand + max_dhw_demand) / 500) * 500
        # distinguish between electric heating systems and conventional ones:
        if self.scenario.boiler.type in ["Air_HP", "Ground_HP", "Electric"]:
            # calculate the design condition COP
            self.worst_COP = OperationModel.calc_cop(
                outside_temperature=[min(self.scenario.region.temperature)],  # get coldest hour of the year
                supply_temperature=self.scenario.building.supply_temperature,
                efficiency=self.scenario.boiler.carnot_efficiency_factor,
                source=self.scenario.boiler.type,
            )
            max_electric_power_float = max_thermal_power / self.worst_COP
            # round the maximum electric power to the next 100 W:
            max_electric_power = np.ceil(max_electric_power_float[0] / 100) * 100
            return max_electric_power
        else:
            # if the heating system is conventional the carnot efficiency factor of the system is treated as the
            # efficiency of the system itself:
            max_thermal_power_float = max_thermal_power / self.scenario.boiler.carnot_efficiency_factor
            # round the maximum electric power to the next 100 W:
            max_thermal_power = np.ceil(max_thermal_power_float / 100) * 100
            self.worst_COP = self.scenario.boiler.carnot_efficiency_factor
            return max_thermal_power

    def generate_solar_gain_rate(self):
        outside_temperature = self.scenario.region.temperature
        shading_threshold_temperature = self.scenario.behavior.shading_threshold_temperature
        shading_solar_reduction_rate = self.scenario.behavior.shading_solar_reduction_rate
        solar_gain_rate = np.ones(len(outside_temperature), )
        for i in range(0, len(outside_temperature)):
            if outside_temperature[i] >= shading_threshold_temperature:
                solar_gain_rate[i] = 1 - shading_solar_reduction_rate
        return solar_gain_rate

    def calculate_solar_gain(self) -> np.array:
        """return: 8760h solar gains, calculated with solar radiation and the effective window area."""

        area_window_east_west = self.scenario.building.effective_window_area_west_east
        area_window_south = self.scenario.building.effective_window_area_south
        area_window_north = self.scenario.building.effective_window_area_north

        Q_solar_north = np.outer(np.array(self.scenario.region.radiation_north), area_window_north)
        Q_solar_east = np.outer(np.array(self.scenario.region.radiation_east), area_window_east_west / 2)
        Q_solar_south = np.outer(np.array(self.scenario.region.radiation_south), area_window_south)
        Q_solar_west = np.outer(np.array(self.scenario.region.radiation_west), area_window_east_west / 2)

        solar_gain_rate = self.generate_solar_gain_rate()
        Q_solar = (Q_solar_north + Q_solar_south + Q_solar_east + Q_solar_west).squeeze() * solar_gain_rate
        return Q_solar

    def create_upper_bound_ev_discharge(self) -> np.array:
        """
        Returns: array that limits the discharge of the EV when it is at home and can use all capacity in one hour
        when not at home (unlimited if not at home because this is endogenously derived)
        """
        upper_discharge_bound_array = []
        for i, status in enumerate(self.scenario.behavior.vehicle_at_home):
            if round(status) == 1:  # when vehicle at home, discharge power is limited
                upper_discharge_bound_array.append(
                    self.scenario.vehicle.discharge_power_max
                )
            else:  # when vehicle is not at home discharge power is limited to max capacity of vehicle
                upper_discharge_bound_array.append(self.scenario.vehicle.capacity)
        return np.array(upper_discharge_bound_array)

    def test_vehicle_profile(self) -> None:
        # test if the vehicle driving profile can be achieved by vehicle capacity:
        counter = 0
        for i, status in enumerate(self.scenario.behavior.vehicle_at_home):
            # add vehicle demand while driving up:
            if round(status) == 0:
                counter += self.scenario.behavior.vehicle_demand[i]
                # check if counter exceeds capacity:
                assert counter <= self.scenario.vehicle.capacity, (
                    "the driving profile exceeds the total capacity of "
                    "the EV. The EV will run out of electricity before "
                    "returning home."
                )
            else:  # vehicle returns home -> counter is set to 0
                counter = 0
