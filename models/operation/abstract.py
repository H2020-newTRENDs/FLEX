from abc import ABC
import numpy as np
from models.operation.scenario import OperationScenario
from models.operation.rc_model import R5C1Model


class OperationModel(ABC):

    def __init__(self, scenario: 'OperationScenario'):
        self.scenario = scenario
        self.DayHour = np.tile(np.arange(1, 25), 365)
        self.cp_water = 4200 / 3600
        _, _, _, mass_temperature = R5C1Model(scenario).calculate_heating_and_cooling_demand(static=True)
        self.thermal_mass_start_temperature = mass_temperature[-1]
        # COP for space heating:
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
        heating_demand, _, _, _ = R5C1Model(self.scenario).calculate_heating_and_cooling_demand()
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
        """calculates the solar gains through solar radiation through the effective window area
        returns the solar gains for all considered building IDs with 8760 rows. Each columns represents a building ID"""
        AreaWindowEastWest = self.scenario.building.effective_window_area_west_east
        AreaWindowSouth = self.scenario.building.effective_window_area_south
        AreaWindowNorth = self.scenario.building.effective_window_area_north

        Q_sol_north = np.outer(np.array(self.scenario.region.radiation_north), AreaWindowNorth)
        Q_sol_east = np.outer(np.array(self.scenario.region.radiation_east), AreaWindowEastWest / 2)
        Q_sol_south = np.outer(np.array(self.scenario.region.radiation_south), AreaWindowSouth)
        Q_sol_west = np.outer(np.array(self.scenario.region.radiation_west), AreaWindowEastWest / 2)
        Q_solar = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())
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
