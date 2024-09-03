from abc import ABC
from collections.abc import Iterable
import numpy as np
from typing import Union
from models.operation.scenario import OperationScenario
import copy
import pandas as pd

building_id_to_name = {
    4: "EZFH 1 B" ,
    5: "EZFH 1 S" ,
    1: "EZFH 5 B" ,
    2: "EZFH 5 S" ,
    3: "EZFH 9 B" ,
    8: "MFH 1 B",
    9: "MFH 1 S",
    6: "MFH 5 B",
    7: "MFH 5 S",
}


class OperationModel(ABC):

    def __init__(self, scenario: "OperationScenario"):
        self.CPWater = 4200 / 3600
        # save all the scenario variables that need to be used later as self variables so the scenario does not have to be copied explicitly because its large:
        self.project_name = scenario.config.project_name
        self.scenario_id = scenario.scenario_id

        # behviour parameters:
        self.HotWaterProfile = scenario.behavior.hot_water_demand
        self.BaseLoadProfile = scenario.behavior.appliance_electricity_demand
        self.target_temperature_array_min = scenario.behavior.target_temperature_array_min
        self.target_temperature_array_max = scenario.behavior.target_temperature_array_max

        # region paramters
        self.T_outside = scenario.region.temperature  # °C
        self.effective_window_area_north = scenario.building.effective_window_area_north
        self.effective_window_area_west_east = scenario.building.effective_window_area_west_east
        self.effective_window_area_sout = scenario.building.effective_window_area_south
        self.shading_threshold_temperature = scenario.behavior.shading_threshold_temperature
        self.shading_solar_reduction_rate = scenario.behavior.shading_solar_reduction_rate
        self.radiation_north = scenario.region.radiation_north
        self.radiation_east = scenario.region.radiation_east
        self.radiation_west = scenario.region.radiation_west
        self.radiation_south = scenario.region.radiation_south

        self.Q_Solar = self.calculate_solar_gain()
        
        # EV parameters
        self.EVDemandProfile = scenario.behavior.vehicle_demand
        self.EVAtHomeProfile = scenario.behavior.vehicle_at_home
        self.EVChargeEfficiency = scenario.vehicle.charge_efficiency
        self.EVDischargeEfficiency = scenario.vehicle.discharge_efficiency
        self.EVOptionV2B = scenario.vehicle.charge_bidirectional
        self.vehicle_charge_power_max = scenario.vehicle.charge_power_max
        self.vehicle_discharge_power_max = scenario.vehicle.discharge_power_max
        self.EVCapacity = scenario.vehicle.capacity
        self.vehicle_demand = scenario.behavior.vehicle_demand

        # Battery parameters
        self.battery_charge_efficiency = scenario.battery.charge_efficiency
        self.battery_discharge_efficiency = scenario.battery.discharge_efficiency
        self.battery_charge_power_max = scenario.battery.charge_power_max
        self.battery_discharge_power_max = scenario.battery.discharge_power_max
        self.battery_capacity = scenario.battery.capacity
        self.BatterySOCStartValue = 0

        # Heating Element
        self.HeatingElement_efficiency = scenario.heating_element.efficiency
        self.HeatingElement_power = scenario.heating_element.power

        # boiler parameters
        self.fuel_boiler_efficiency = scenario.boiler.fuel_boiler_efficiency
        self.supply_temperature = scenario.building.supply_temperature
        self.carnot_efficiency_factor = scenario.boiler.carnot_efficiency_factor
        self.boiler_type = scenario.boiler.type

        # Hot water parameters
        self.HotWaterHourlyCOP = self.calc_cop(
            outside_temperature=self.T_outside,
            supply_temperature=self.supply_temperature,
            efficiency=self.carnot_efficiency_factor,
            source=self.boiler_type,
        )
        self.HotWaterHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.T_outside,
            supply_temperature=self.supply_temperature + 10,
            efficiency=self.carnot_efficiency_factor,
            source=self.boiler_type,
        )
        self.T_TankStart_DHW = scenario.hot_water_tank.temperature_start
        self.M_WaterTank_DHW = scenario.hot_water_tank.size
        self.U_LossTank_DHW = scenario.hot_water_tank.loss
        self.T_TankSurrounding_DHW = scenario.hot_water_tank.temperature_surrounding
        self.hot_water_tank_temperature_min = scenario.hot_water_tank.temperature_min
        self.hot_water_tank_temperature_max = scenario.hot_water_tank.temperature_max
        self.A_SurfaceTank_DHW = self.calculate_surface_area_from_volume(scenario.hot_water_tank.size)
        self.Q_TankEnergyMin_DHW = self.CPWater * scenario.hot_water_tank.size * \
                                   (273.15 + self.hot_water_tank_temperature_min)
        self.Q_TankEnergyMax_DHW = self.CPWater * scenario.hot_water_tank.size * \
                                   (273.15 + self.hot_water_tank_temperature_max)

        # Thermal tank parameters and space heating
        self.SpaceHeatingHourlyCOP = self.calc_cop(
            outside_temperature=self.T_outside,
            supply_temperature=self.supply_temperature,
            efficiency=self.carnot_efficiency_factor,
            source=self.boiler_type,
        )
        self.SpaceHeatingHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.T_outside,
            supply_temperature=self.supply_temperature + 10,
            efficiency=self.carnot_efficiency_factor,
            source=self.boiler_type,
        )
        self.T_TankStart_heating = scenario.space_heating_tank.temperature_start
        self.M_WaterTank_heating = scenario.space_heating_tank.size
        self.U_LossTank_heating = scenario.space_heating_tank.loss
        self.T_TankSurrounding_heating = scenario.space_heating_tank.temperature_surrounding
        self.space_heating_tank_temperature_min = scenario.space_heating_tank.temperature_min
        self.space_heating_tank_temperature_max = scenario.space_heating_tank.temperature_max
        self.A_SurfaceTank_heating = self.calculate_surface_area_from_volume(scenario.space_heating_tank.size)
        self.Q_TankEnergyMin_heating = self.CPWater * scenario.space_heating_tank.size * \
                                       (273.15 + self.space_heating_tank_temperature_min)
        self.Q_TankEnergyMax_heating = self.CPWater * scenario.space_heating_tank.size * \
                                       (273.15 + self.space_heating_tank_temperature_max)
        
        # Cooling parameters:
        self.CoolingCOP = scenario.space_cooling_technology.efficiency
        self.CoolingHourlyCOP = (np.ones(8760, ) * self.CoolingCOP)
        self.cooling_power = scenario.space_cooling_technology.power

        # PV parameters:
        self.PhotovoltaicProfile = scenario.pv.generation
        self.pv_size = scenario.pv.size

        # energy price parameters:
        self.ElectricityPrice = scenario.energy_price.electricity  # C/Wh
        self.FiT = scenario.energy_price.electricity_feed_in  # C/Wh
        self.fuel_price = scenario.energy_price.fuel

        # building parameters:
        self.Am_factor = scenario.building.Am_factor
        self.Af = scenario.building.Af
        self.Cm_factor = scenario.building.CM_factor
        self.internal_gains = scenario.building.internal_gains
        self.Htr_w = scenario.building.Htr_w
        self.Hop = scenario.building.Hop
        self.Hve = scenario.building.Hve
        self.Am = (self.Am_factor * self.Af)  # Effective mass related area [m^2]
        self.Cm = self.Cm_factor * self.Af
        self.Atot = (4.5 * self.Af)  # 7.2.2.2: Area of all surfaces facing the building zone
        self.Qi = self.internal_gains * self.Af
        self.Htr_w = self.Htr_w
        self.Htr_ms = np.float_(9.1) * self.Am  # from 12.2.2 Equ. (64)
        self.Htr_is = np.float_(3.45) * self.Atot
        self.Htr_em = 1 / (1 / self.Hop - 1 / self.Htr_ms)  # from 12.2.2 Equ. (63)
        self.Htr_1 = np.float_(1) / (np.float_(1) / self.Hve + np.float_(1) / self.Htr_is)  # Equ. C.6
        self.Htr_2 = self.Htr_1 + self.Htr_w  # Equ. C.7
        self.Htr_3 = 1 / (1 / self.Htr_2 + 1 / self.Htr_ms)  # Equ.C.8
        self.PHI_ia = 0.5 * self.Qi  # Equ. C.1
        _, _, _, mass_temperature = self.calculate_heating_and_cooling_demand(static=True)
        self.BuildingMassTemperatureStartValue = mass_temperature[-1]

        self.Q_RoomHeating, self.Q_RoomCooling, self.T_Room, self.T_BuildingMass = \
            self.calculate_heating_and_cooling_demand(thermal_start_temperature=self.BuildingMassTemperatureStartValue,
                                                      static=False)
        self.SpaceHeating_MaxBoilerPower = self.generate_maximum_electric_or_thermal_power()
        self.max_target_temperature, self.min_target_temperature = self.generate_target_indoor_temperature(temperature_offset=3)

        self.reference_Q_RoomHeating_binary = [1 if x>0 else 0 for x in self.Q_RoomHeating]

        self.load_dual_variables(scenario)

        self.calculate_thermal_mass_loss_factor_based_on_outside_temp(scenario)
        self.add_thermal_heating_tank_loss_factor(scenario)
        self.add_thermal_dhw_tank_loss_factor(scenario)
        self.add_price_quantile_values_for_thermal_mass(scenario)

    def load_dual_variables(self, scenario):
        # df = pd.read_csv(scenario.config.input / "dual_variables.csv", sep=";").query(f"ID_Building == {self.get_building_id(scenario=scenario)}")
        self.dual_battery = 0  #df["dual variable Battery SOC"].values
        self.dual_heating_tank = 0  #df["dual variable heating tank SOC"].values
        self.dual_dhw_tank = 0  # df["dual variable DHW tank SOC"].values
        self.dual_thermal_mass = 0  #df["dual variable thermal mass"].values        

    def calculate_thermal_mass_loss_factor_based_on_outside_temp(self, scenario):
        df = pd.read_csv(scenario.config.input / "thermal_mass_loss_based_on_outside_temperature.csv", sep=";")

        scen_table = scenario.input_tables["OperationScenario"]
        building_id = scen_table.loc[scen_table["ID_Scenario"]==self.scenario_id, "ID_Building"].values[0]
        
        b_df = df.loc[df["Building ID"]==building_id_to_name[building_id], :].copy()
        loss_array = b_df["loss"].values
        max_temp = b_df["outside temperature"].max()
        loss_max_temp = b_df.loc[b_df["outside temperature"]==max_temp, "loss"].values[0]
        min_temp = b_df["outside temperature"].min()
        loss_min_temp = b_df.loc[b_df["outside temperature"]==min_temp, "loss"].values[0]

        loss_vector = []
        for x in self.T_outside:
            if x > max_temp:
                loss_vector.append(loss_max_temp + (x-max_temp)/100 + 0.02)  # plus a penalty if the temperature is much higher
            elif x < min_temp:
                loss_vector.append(loss_min_temp)
            else:
            # interpolate
                diff_array = np.abs(b_df["outside temperature"].values - x)
                smallest_indices = np.argpartition(diff_array, 2)[:2]
                closest_losses = loss_array[smallest_indices]
                closest_diffs = diff_array[smallest_indices]
                weighted_sum = (closest_losses * closest_diffs).sum()
                loss_vector.append(weighted_sum)

        self.thermal_mass_loss_factor = loss_vector
    
    def get_building_id(self, scenario) -> int:
        scen_table = scenario.input_tables["OperationScenario"]
        building_id = scen_table.loc[scen_table["ID_Scenario"]==self.scenario_id, "ID_Building"].values[0]
        return building_id

    def add_price_quantile_values_for_thermal_mass(self, scenario):
        df = pd.read_csv(scenario.config.input / "price_quantile_values_thermal_mass_unweighted.csv", sep=";")
        building_id = self.get_building_id(scenario)
        self.price_quantile_for_charging_thermal_mass = df.loc[df["ID_Building"]==building_id_to_name[building_id], "price quantile for charging"].values[0]

    def add_thermal_heating_tank_loss_factor(self, scenario):
        if self.M_WaterTank_heating != 0:
            df = pd.read_csv(scenario.config.input / "Heating_tank_loss_values.csv", sep=";")
            building_id = self.get_building_id(scenario)
            self.heating_tank_loss_factor = df.loc[df["ID_Building"]==building_id_to_name[building_id], "xi_heating_tank"].values[0]
        else:
            self.heating_tank_loss_factor = 1

    def add_thermal_dhw_tank_loss_factor(self, scenario):
        if self.M_WaterTank_DHW != 0:
            df = pd.read_csv(scenario.config.input / "DHW_tank_loss_values.csv", sep=";")
            building_id = self.get_building_id(scenario)
            self.dhw_tank_loss_factor = df.loc[df["ID_Building"]==building_id_to_name[building_id], "xi_dhw_tank"].values[0]  
        else:
            self.dhw_tank_loss_factor = 1


    def calculate_surface_area_from_volume(self, volume: float) -> float:
        # V = h * pi * r2 -> h = V/(r2*pi), A = 2r*pi*h + 2pi*r2 -> A = 2r*pi*V/(r2*pi) + 2pi*r2 = 2V/r + 2pi*r2
        # derivative of area: 0 = 4pi*r - 4V/r2  -> V = pi*r3  -> r = sqrt^3(V/pi)
        if volume == 0:
            return 0
        V = volume / 1_000  # from l into m^3
        r = np.cbrt(V / np.pi)
        h = V / (r ** 2 * np.pi)
        return np.round(2 * r * np.pi * h + 2 * np.pi * r ** 2, 2)
        

    def calculate_heating_and_cooling_demand(
            self, thermal_start_temperature: float = 15, static=False
    ) -> (np.array, np.array, np.array, np.array):
        """
        if "static" is True, then the RC model will calculate a static heat demand calculation for the first hour of
        the year by using this hour 100 times. This way a good approximation of the thermal mass temperature of the
        building in the beginning of the calculation is achieved. Solar gains are set to 0.
        Returns: heating demand, cooling demand, indoor air temperature, temperature of the thermal mass
        """
        heating_power_10 = self.Af * 10

        if self.cooling_power  == 0:
            T_air_max = np.full((8760,), 100)
            # if no cooling is adopted --> raise max air temperature to 100 so it will never cool:
        else:
            T_air_max = self.target_temperature_array_max

        if static:
            Q_solar = np.array([0] * 100)
            T_outside = np.array([self.T_outside[0]] * 100)
            T_air_min = np.array(
                [self.target_temperature_array_min[0]] * 100
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
            T_air_min = self.target_temperature_array_min
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
            temperature_max_winter = self.target_temperature_array_min[i] + temperature_offset
            if indoor_temp < temperature_max_winter:
                min_temperature_list.append(self.target_temperature_array_min[i])
                max_temperature_list.append(indoor_temp + temperature_offset)
            elif temperature_max_winter <= indoor_temp <= self.target_temperature_array_max[
                i] + 0.01:  # 0.01 is added to ignore floating point errors
                if self.cooling_power  == 0:
                    max_temperature_list.append(indoor_temp + temperature_offset)
                    min_temperature_list.append(indoor_temp - temperature_offset)
                else:
                    max_temperature_list.append(self.target_temperature_array_max[i])
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
        max_dhw_demand = self.HotWaterProfile.max()
        # round to the next 500 W
        max_thermal_power = np.ceil((max_heating_demand + max_dhw_demand) / 500) * 500
        # distinguish between electric heating systems and conventional ones:
        if self.boiler_type in ["Air_HP", "Ground_HP", "Electric"]:
            # calculate the design condition COP
            worst_COP = OperationModel.calc_cop(
                outside_temperature=[min(self.T_outside)],  # get coldest hour of the year
                supply_temperature=self.supply_temperature,
                efficiency=self.carnot_efficiency_factor,
                source=self.boiler_type,
            )
            max_electric_power_float = max_thermal_power / worst_COP
            # round the maximum electric power to the next 100 W:
            max_electric_power = np.ceil(max_electric_power_float[0] / 100) * 100
            return max_electric_power
        else:
            # if the heating system is conventional the carnot efficiency factor of the system is treated as the
            # efficiency of the system itself:
            max_thermal_power_float = max_thermal_power / self.carnot_efficiency_factor
            # round the maximum electric power to the next 100 W:
            max_thermal_power = np.ceil(max_thermal_power_float / 100) * 100
            return max_thermal_power

    def generate_solar_gain_rate(self):
        solar_gain_rate = np.ones(len(self.T_outside), )
        for i in range(0, len(self.T_outside)):
            if self.T_outside[i] >= self.shading_threshold_temperature:
                solar_gain_rate[i] = 1 - self.shading_solar_reduction_rate
        return solar_gain_rate

    def calculate_solar_gain(self) -> np.array:
        """return: 8760h solar gains, calculated with solar radiation and the effective window area."""
        Q_solar_north = np.outer(np.array(self.radiation_north), self.effective_window_area_north)
        Q_solar_east = np.outer(np.array(self.radiation_east), self.effective_window_area_west_east / 2)
        Q_solar_south = np.outer(np.array(self.radiation_south), self.effective_window_area_sout)
        Q_solar_west = np.outer(np.array(self.radiation_west), self.effective_window_area_west_east / 2)

        solar_gain_rate = self.generate_solar_gain_rate()
        Q_solar = (Q_solar_north + Q_solar_south + Q_solar_east + Q_solar_west).squeeze() * solar_gain_rate
        return Q_solar

    def create_upper_bound_ev_discharge(self) -> np.array:
        """
        Returns: array that limits the discharge of the EV when it is at home and can use all capacity in one hour
        when not at home (unlimited if not at home because this is endogenously derived)
        """
        upper_discharge_bound_array = []
        for i, status in enumerate(self.EVAtHomeProfile):
            if round(status) == 1:  # when vehicle at home, discharge power is limited
                upper_discharge_bound_array.append(self.vehicle_discharge_power_max)
            else:  # when vehicle is not at home discharge power is limited to max capacity of vehicle
                upper_discharge_bound_array.append(self.EVCapacity)
        return np.array(upper_discharge_bound_array)

    def test_vehicle_profile(self) -> None:
        # test if the vehicle driving profile can be achieved by vehicle capacity:
        counter = 0
        for i, status in enumerate(self.vehicle_at_home):
            # add vehicle demand while driving up:
            if round(status) == 0:
                counter += self.vehicle_demand[i]
                # check if counter exceeds capacity:
                assert counter <= self.vehicle.capacity, (
                    "the driving profile exceeds the total capacity of "
                    "the EV. The EV will run out of electricity before "
                    "returning home."
                )
            else:  # vehicle returns home -> counter is set to 0
                counter = 0

    @staticmethod
    def split_data(vector)->list:
        # split the vector into 36 hour pieces however the "new data" will always be available at 12o'clock
        # therefore the perfect foresight is effectively 36 hours, being updated every 24 hours
        # split the data into 36 hour pieces starting at hour 0, 12, 36, 60, 84, ...
        splited_vector = [vector[0:24]] + [vector[i:i+36] for i in np.arange(12, 8760, 24)]
        return splited_vector

    def return_splitted_scenarios(self):
        """This function returns a list containing OperationModel
        This operation is very inefficient. It would be better to have seperated classes for all the input data of the model and classes only for the logic."""
        list_of_classes = [copy.deepcopy(self) for _ in range(366)]
        for key, value in self.__dict__.items():
                if isinstance(value, Iterable) and not isinstance(value, str):
                    # slice it into 366 pieces:
                    splitted_value = self.split_data(value)
                    for i, split_class in enumerate(list_of_classes):
                        setattr(split_class, key, splitted_value[i])
        return list_of_classes
    
