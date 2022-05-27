from abc import ABC
import numpy as np
from models.operation.scenario import OperationScenario


class OperationModel(ABC):

    def __init__(self, scenario: 'OperationScenario'):
        self.scenario = scenario
        self.building = scenario.building
        self.init_rc_params()
        self.init_pyomo_params()


    def init_pyomo_params(self):
        self.SpaceHeatingHourlyCOP = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.boiler.heating_supply_temperature,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type
        )
        # COP for space heating tank charging (10°C increase in supply temperature):
        self.SpaceHeatingHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.boiler.heating_supply_temperature + 10,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type
        )
        # COP DHW:
        self.HotWaterHourlyCOP = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.boiler.hot_water_supply_temperature,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type
        )
        # COP DHW tank charging (10°C increase in supply temperature):
        self.HotWaterHourlyCOP_tank = self.calc_cop(
            outside_temperature=self.scenario.region.temperature,
            supply_temperature=self.scenario.boiler.hot_water_supply_temperature + 10,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type
        )

        # heat pump maximal electric power
        self.SpaceHeating_HeatPumpMaximalElectricPower = self.generate_maximum_electric_heat_pump_power()
        self.DayHour = np.tile(np.arange(1, 25), 365)
        self.cp_water = 4200 / 3600
        _, _, _, mass_temperature = self.calculate_heating_and_cooling_demand(static=True)
        self.thermal_mass_start_temperature = mass_temperature[-1]
        if self.scenario.vehicle.capacity > 0:
            self.test_vehicle_profile()  # test if vehicle makes model infeasible





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

    def generate_maximum_target_indoor_temperature_no_cooling(self, temperature_max: int) -> np.array:
        """
        calculates an array of the maximum temperature that will be equal to the provided max temperature if
        heating is needed at the current hour, but it will be "unlimited" (60°C) if no heating is needed. This
        way it is ensured that the model does not pre-heat the building above the maximum temperature and at the same
        time the model will not be infeasible when the household is heated above maximum temperature by external
        radiation.
        Args:
            temperature_max: int

        Returns: array of the maximum temperature

        """
        heating_demand, _, T_room, _ = self.calculate_heating_and_cooling_demand()
        # create max temperature array:
        max_temperature_list = []
        for i, heat_demand in enumerate(heating_demand):
            # if heat demand == 0 and the heat demand in the following 8 hours is also 0 and the heat demand of 3
            # hours before that is also 0, then the max temperature is raised so model does not become infeasible:
            if heat_demand == 0 and \
                   heating_demand[i - 3] == 0 and \
                   heating_demand[i - 2] == 0 and \
                   heating_demand[i - 1] == 0 and \
                   heating_demand[i + 1] == 0 and \
                   heating_demand[i + 2] == 0 and \
                   heating_demand[i + 3] == 0 and \
                   heating_demand[i + 4] == 0 and \
                   heating_demand[i + 5] == 0 and \
                   heating_demand[i + 6] == 0 and \
                   heating_demand[i + 7] == 0 and \
                   heating_demand[i + 8] == 0:
                # append the temperature of the reference model + 0.5°C to make sure it is feasible
                max_temperature_list.append(T_room[i] + 1)
            else:
                max_temperature_list.append(temperature_max)
        return np.array(max_temperature_list)
















    @staticmethod
    def calc_cop(outside_temperature: np.array,
                 supply_temperature: float,
                 efficiency: float,
                 source: str) -> np.array:
        """
        Args:
            outside_temperature:
            supply_temperature: temperature that the heating system needs (eg. 35 low temperature, 65 hot)
            efficiency: carnot efficiency factor
            source: refers to the heat source (air, ground...) is saved in the boiler class as "name"

        Returns: numpy array of the hourly COP
        """
        if source == "Air_HP":
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - temp) for temp in
                            outside_temperature])
        elif source == "Ground_HP":
            # for the ground source heat pump the temperature of the source is 10°C
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - 10) for temp in
                            outside_temperature])
        else:
            assert "only >>air<< and >>ground<< are valid arguments"
        return COP

    def generate_maximum_electric_heat_pump_power(self):
        # TODO we could add different supply temperatures for different buildings to make COP more accurate
        """
        Calculates the necessary HP power for each building through the 5R1C reference model. The maximum heating power
        then will be rounded to the next 500 W. Then we divide the thermal power by the worst COP of the HP
        which is calculated at design conditions (-12°C) and the respective source and supply temperature.

        Args:
            scenario: AbstractScenario

        Returns: maximum heat pump electric power (float)
        """
        # calculate the heating demand in reference mode:
        heating_demand, _, _, _ = self.calculate_heating_and_cooling_demand()
        max_heating_demand = heating_demand.max()
        # round to the next 500 W
        max_thermal_power = np.ceil(max_heating_demand / 500) * 500
        # calculate the design condition COP (-12°C)
        worst_COP = OperationModel.calc_cop(
            outside_temperature=[-12],
            supply_temperature=self.scenario.boiler.heating_supply_temperature,
            efficiency=self.scenario.boiler.carnot_efficiency_factor,
            source=self.scenario.boiler.type
        )
        max_electric_power_float = max_thermal_power / worst_COP
        # round the maximum electric power to the next 100 W:
        max_electric_power = np.ceil(max_electric_power_float[0] / 100) * 100
        return max_electric_power

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

    def create_upper_bound_EV_discharge(self) -> np.array:
        """
        Returns: array that limits the discharge of the EV when it is at home and can use all capacity in one hour
        when not at home (unlimited if not at home because this is endogenously derived)
        """
        upper_discharge_bound_array = []
        for i, status in enumerate(self.scenario.behavior.vehicle_at_home):
            if round(status) == 1:  # when vehicle at home, discharge power is limited
                upper_discharge_bound_array.append(self.scenario.vehicle.discharge_power_max)
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
                assert counter <= self.scenario.vehicle.capacity, "the driving profile exceeds the total capacity of " \
                                                                  "the EV. The EV will run out of electricity before " \
                                                                  "returning home."
            else:  # vehicle returns home -> counter is set to 0
                counter = 0

    def calculate_Atot(self, Af: float) -> float:
        return 4.5 * Af  # 7.2.2.2: Area of all surfaces facing the building zone

    def calculate_Cm(self, CM_factor: float, Af: float) -> float:
        return CM_factor * Af

    def calculate_Am(self, Am_factor: float, Af: float) -> float:
        # (Effective mass related area) [m^2]
        return Am_factor * Af

    def calculate_Qi(self, specific_internal_gains: float, Af: float) -> float:
        # internal gains
        return specific_internal_gains * Af

    # Coupling values
    def calculate_his(self) -> float:
        return np.float_(3.45)  # 7.2.2.2 Coupling Temp Air with Temp Surface node s

    def calculate_hms(self):
        return np.float_(9.1)  # [W / m2K] from Equ.C.3 (from 12.2.2) - (coupling between mass and central node s)

    def calculate_Htr_ms(self,  Am_factor: float, Af: float) -> float:
        return self.calculate_hms() * self.calculate_Am(Am_factor, Af)  # from 12.2.2 Equ. (64)

    def calculate_Htr_em(self, Hop: float, Am_factor: float, Af: float) -> float:
        return 1 / (1 / Hop - 1 / self.calculate_Htr_ms(Am_factor, Af))  # from 12.2.2 Equ. (63)

    # thermal coupling values [W/K]
    def calculate_Htr_is(self, Af) -> float:
        return self.calculate_his() * self.calculate_Atot(Af=Af)

    def calculate_PHI_ia(self,specific_internal_gains: float, Af: float) -> float:
        return 0.5 * self.calculate_Qi(specific_internal_gains, Af)  # Equ. C.1

    def calculate_Htr_1(self, Hve: float, Af: float) -> float:
        return 1 / (1 / Hve + 1 / self.calculate_Htr_is(Af))  # Equ. C.6

    def calculate_Htr_2(self, Hve: float, Af: float, Htr_w: float) -> float:
        return self.calculate_Htr_1(Hve, Af) + Htr_w  # Equ. C.7

    def calculate_Htr_3(self,  Hve: float, Af: float, Htr_w: float, Am_factor: float) -> float:
        return 1 / (1 / self.calculate_Htr_2(Hve, Af, Htr_w) + 1 / self.calculate_Htr_ms(Am_factor, Af))  # Equ.C.8

    def set_result_variables(self):
        # Result variables: CAREFUL, THESE NAMES HAVE TO BE IDENTICAL TO THE ONES IN THE PYOMO OPTIMIZATION
        # -----------
        # PARAMETERS:
        # -----------
        # price
        self.electricity_price = None  # C/Wh
        # Feed in Tariff of Photovoltaic
        self.FiT = None  # C/Wh
        # solar gains:
        self.Q_Solar = None  # W
        # outside temperature
        self.T_outside = None  # °C
        # COP of cooling
        self.CoolingCOP = None  # single value because it is not dependent on time
        # electricity load profile
        self.BaseLoadProfile = None
        # PV profile
        self.PhotovoltaicProfile = None
        # HotWater
        self.HotWaterProfile = None

        # building source: not saved to DB

        # Heating Tank source
        # Mass of water in tank
        self.M_WaterTank_heating = None
        # Surface of Tank in m2
        self.A_SurfaceTank_heating = None
        # insulation of tank, for calc of losses
        self.U_ValueTank_heating = None
        self.T_TankStart_heating = None
        # surrounding temp of tank
        self.T_TankSurrounding_heating = None

        # DHW Tank source
        # Mass of water in tank
        self.M_WaterTank_DHW = None
        # Surface of Tank in m2
        self.A_SurfaceTank_DHW = None
        # insulation of tank, for calc of losses
        self.U_ValueTank_DHW = None
        self.T_TankStart_DHW = None
        # surrounding temp of tank
        self.T_TankSurrounding_DHW = None

        # Battery source
        self.ChargeEfficiency = None
        self.DischargeEfficiency = None

        # EV
        self.EVDemandProfile = None
        self.EVAtHomeStatus = None
        self.EVChargeEfficiency = None
        self.EVDischargeEfficiency = None

        # ----------------------------
        # Variables
        # ----------------------------
        # Variables SpaceHeating
        self.Q_HeatingTank_in = None
        self.Q_HeatingElement = None
        self.Q_HeatingTank_out = None
        self.E_HeatingTank = None  # energy in the tank
        self.Q_HeatingTank_bypass = None
        self.E_Heating_HP_out = None
        self.Q_room_heating = None

        # Variables DHW
        self.Q_DHWTank_out = None
        self.E_DHWTank = None  # energy in the tank
        self.Q_DHWTank_in = None
        self.E_DHW_HP_out = None
        self.Q_DHWTank_bypass = None

        # Variable space cooling
        self.Q_RoomCooling = None

        # Temperatures (room and thermal mass)
        self.T_room = None
        self.Tm_t = None

        # Grid variables
        self.Grid = None
        self.Grid2Load = None
        self.Grid2Bat = None
        self.Grid2EV = None

        # PV variables
        self.PV2Load = None
        self.PV2Bat = None
        self.PV2Grid = None
        self.PV2EV = None

        # Electric Load and Electricity fed back to the grid
        self.Load = None
        self.Feed2Grid = None

        # Battery variables
        self.BatSoC = None
        self.BatCharge = None
        self.BatDischarge = None
        self.Bat2Load = None
        self.Bat2EV = None

        # electric vehicle (EV)
        self.EVSoC = None
        self.EVCharge = None
        self.EVDischarge = None

        self.EV2Bat = None
        self.EV2Grid = None
        self.EV2Load = None

        # Objective:
        self.total_operation_cost = None