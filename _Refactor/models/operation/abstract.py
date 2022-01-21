from typing import Type
from abc import ABC, abstractmethod
import numpy as np

from _Refactor.core.environment.abstract_environment import AbstractEnvironment
from _Refactor.core.household.abstract_household import AbstractHousehold

"""
abstract operation model
"""


class AbstractOperationModel(ABC):

    def __init__(self,
                 household: 'AbstractHousehold'):

        self.household = household
        self.day_hour = np.tile(np.arange(1, 25), 365)

        # Result variables: CAREFUL, THESE NAMES HAVE TO BE IDENTICAL TO THE ONES IN THE PYOMO OPTIMIZATION
        # space heating
        self.Q_HeatingTank_in = None
        self.Q_HeatingElement = None
        self.Q_HeatingTank_out = None
        self.E_HeatingTank = None
        self.Q_HeatingTank_bypass = None
        self.Q_Heating_HP_out = None
        self.Q_room_heating = None

        # DHW
        self.Q_DHWTank_out = None
        self.E_DHWTank = None
        self.Q_DHWTank_in = None
        self.Q_DHW_HP_out = None
        self.Q_DHWTank_bypass = None

        # space cooling
        self.Q_RoomCooling = None

        # temperatures
        self.T_room = None
        self.Tm_t = None

        # grid variables
        self.Grid = None
        self.Grid2Load = None
        self.Grid2Bat = None

        # PV variables
        self.PV2Load = None
        self.PV2Bat = None
        self.PV2Grid = None

        # electric load
        self.Load = None

        # electricity fed back to the grid
        self.Feedin = None

        # Battery
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
        calculates the COP based on a performance factor and the massflow temperatures
        """
        if source == "Air_HP":
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - temp) for temp in
                            outside_temperature])
        elif source == "Water_HP":
            # for the ground source heat pump the temperature of the source is 10°C
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - 10) for temp in
                            outside_temperature])
        else:
            assert "only >>air<< and >>ground<< are valid arguments"
        return COP

    def calculate_solar_gains(self) -> np.array:
        """calculates the solar gains through solar radiation through the effective window area
        returns the solar gains for all considered building IDs with 8760 rows. Each columns represents a building ID"""
        AreaWindowEastWest = self.household.building_class.effective_window_area_west_east
        AreaWindowSouth = self.household.building_class.effective_window_area_south
        AreaWindowNorth = self.household.building_class.effective_window_area_north

        Q_sol_north = np.outer(np.array(self.household.region_class.north), AreaWindowNorth)
        Q_sol_east = np.outer(np.array(self.household.region_class.east), AreaWindowEastWest / 2)
        Q_sol_south = np.outer(np.array(self.household.region_class.south), AreaWindowSouth)
        Q_sol_west = np.outer(np.array(self.household.region_class.west), AreaWindowEastWest / 2)
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

    def calculate_Htr_ms(self, hms: float, Am: float) -> float:
        return hms * Am  # from 12.2.2 Equ. (64)

    def calculate_Htr_em(self, Hop: float, Htr_ms: float) -> float:
        return 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)

    # thermal coupling values [W/K]
    def calculate_Htr_is(self, his: float, Atot: float) -> float:
        return his * Atot

    def calculate_PHI_ia(self, internal_gains: float) -> float:
        """
        Args:
            internal_gains: total internal gains, not specific
        """
        return 0.5 * internal_gains  # Equ. C.1

    def calculate_Htr_1(self, Hve: float, Htr_is: float) -> float:
        return 1 / (1 / Hve + 1 / Htr_is)  # Equ. C.6

    def calculate_Htr_2(self, Htr_1: float, Htr_w: float) -> float:
        return Htr_1 + Htr_w  # Equ. C.7

    def calculate_Htr_3(self, Htr_2: float, Htr_ms: float) -> float:
        return 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8
