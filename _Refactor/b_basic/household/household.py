
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .components import Region, PersonList, Building, ApplianceList, Boiler, Tank,\
                            AirConditioner, PV, Battery, Vehicle, Behavior, Demand, Policy

class Household(ABC):

    def __init__(self,
                 region: 'Region' = None,
                 person_list: 'PersonList' = None,
                 building: 'Building' = None,
                 appliance_list: 'ApplianceList' = None,
                 boiler: 'Boiler' = None,
                 tank: 'Tank' = None,
                 air_conditioner: 'AirConditioner' = None,
                 pv: 'PV' = None,
                 battery: 'Battery' = None,
                 vehicle: 'Vehicle' = None,
                 behavior: 'Behavior' = None,
                 demand: 'Demand' = None,
                 policy: 'Policy' = None):

        self.region = region
        self.person_list = person_list
        self.building = building
        self.appliance_list = appliance_list
        self.boiler = boiler
        self.tank = tank
        self.air_conditioner = air_conditioner
        self.pv = pv
        self.battery = battery
        self.vehicle = vehicle
        self.behavior = behavior
        self.demand = demand
        self.policy = policy

    @abstractmethod
    def setup(self):
        pass





