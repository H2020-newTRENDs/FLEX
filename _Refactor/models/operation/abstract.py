from typing import Type
from abc import ABC, abstractmethod
import numpy as np

from _Refactor.core.household.abstract_scenario import AbstractScenario

"""
abstract operation model
"""


class AbstractOperationModel(ABC):

    def __init__(self,
                 scenario: 'AbstractScenario'):

        self.scenario = scenario
        self.day_hour = np.tile(np.arange(1, 25), 365)
        self.cp_water = 4200 / 3600

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
        # COP of heatpump
        self.SpaceHeatingHourlyCOP = None
        # COP of cooling
        self.CoolingCOP = None  # single value because it is not dependent on time
        # electricity load profile
        self.BaseLoadProfile = None
        # PV profile
        self.PhotovoltaicProfile = None
        # HotWater
        self.HotWaterProfile = None
        self.HotWaterHourlyCOP = None

        # Smart Technologies
        self.DayHour = None

        # building data:
        self.Am = None
        self.Atot = None
        self.Qi = None
        self.Htr_w = None
        self.Htr_em = None
        self.Htr_3 = None
        self.Htr_1 = None
        self.Htr_2 = None
        self.Hve = None
        self.Htr_ms = None
        self.Htr_is = None
        self.PHI_ia = None
        self.Cm = None
        self.BuildingMassTemperatureStartValue = None

        # Heating Tank data
        # Mass of water in tank
        self.M_WaterTank_heating = None
        # Surface of Tank in m2
        self.A_SurfaceTank_heating = None
        # insulation of tank, for calc of losses
        self.U_ValueTank_heating = None
        self.T_TankStart_heating = None
        # surrounding temp of tank
        self.T_TankSurrounding_heating = None

        # DHW Tank data
        # Mass of water in tank
        self.M_WaterTank_DHW = None
        # Surface of Tank in m2
        self.A_SurfaceTank_DHW = None
        # insulation of tank, for calc of losses
        self.U_ValueTank_DHW = None
        self.T_TankStart_DHW = None
        # surrounding temp of tank
        self.T_TankSurrounding_DHW = None

        # heat pump
        self.SpaceHeating_HeatPumpMaximalThermalPower = None

        # Battery data
        self.ChargeEfficiency = None
        self.DischargeEfficiency = None

        # ----------------------------
        # Variables
        # ----------------------------
        # Variables SpaceHeating
        self.Q_HeatingTank_in = None
        self.Q_HeatingElement = None
        self.Q_HeatingTank_out = None
        self.E_HeatingTank = None  # energy in the tank
        self.Q_HeatingTank_bypass = None
        self.Q_Heating_HP_out = None
        self.Q_room_heating = None

        # Variables DHW
        self.Q_DHWTank_out = None
        self.E_DHWTank = None  # energy in the tank
        self.Q_DHWTank_in = None
        self.Q_DHW_HP_out = None
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

        # PV variables
        self.PV2Load = None
        self.PV2Bat = None
        self.PV2Grid = None

        # Electric Load and Electricity fed back to the grid
        self.Load = None
        self.Feed2Grid = None

        # Battery variables
        self.BatSoC = None
        self.BatCharge = None
        self.BatDischarge = None
        self.Bat2Load = None

        # electric vehicle (EV)

    def COP_HP(self,
               outside_temperature: np.array,
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

    def calculate_solar_gains(self) -> np.array:
        """calculates the solar gains through solar radiation through the effective window area
        returns the solar gains for all considered building IDs with 8760 rows. Each columns represents a building ID"""
        AreaWindowEastWest = self.scenario.building_class.effective_window_area_west_east
        AreaWindowSouth = self.scenario.building_class.effective_window_area_south
        AreaWindowNorth = self.scenario.building_class.effective_window_area_north

        Q_sol_north = np.outer(np.array(self.scenario.region_class.north), AreaWindowNorth)
        Q_sol_east = np.outer(np.array(self.scenario.region_class.east), AreaWindowEastWest / 2)
        Q_sol_south = np.outer(np.array(self.scenario.region_class.south), AreaWindowSouth)
        Q_sol_west = np.outer(np.array(self.scenario.region_class.west), AreaWindowEastWest / 2)
        Q_solar = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())
        return Q_solar



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
