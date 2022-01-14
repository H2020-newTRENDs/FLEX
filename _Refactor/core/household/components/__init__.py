
from .region import Region
from .person import Person, PersonList
from .building import Building
from .appliance import Appliance, ApplianceList
from .boiler import Boiler
from .space_heating_tank import SpaceHeatingTank
from .air_conditioner import AirConditioner
from .hot_water_tank import HotWaterTank
from .pv import PV
from .battery import Battery
from .vehicle import Vehicle
from .behavior import Behavior
from .demand import HotWaterDemand, ElectricityDemand
from .electricity_price import ElectricityPrice
from .feedin_tariff import FeedInTariff

component_list = [Region, Person, PersonList, Building, Appliance, ApplianceList, Boiler, SpaceHeatingTank,
                  AirConditioner, HotWaterTank, PV, Battery, Vehicle, Behavior, HotWaterDemand,
                  ElectricityDemand, ElectricityPrice, FeedInTariff]


