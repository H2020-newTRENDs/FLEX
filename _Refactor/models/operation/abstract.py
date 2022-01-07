
from typing import Type
from abc import ABC, abstractmethod

from _Refactor.core.environment.abstract_environment import AbstractEnvironment
from _Refactor.core.household.abstract_household import AbstractHousehold

"""
abstract operation model
"""
class AbstractOperationModel(ABC):

    def __init__(self,
                 household: 'AbstractHousehold',
                 environment: 'AbstractEnvironment'):

        self.household = household
        self.environment = environment

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


    # @abstractmethod
    # def run(self):
    #     data_collector(some_dict)
    #     pass


