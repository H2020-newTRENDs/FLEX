
class Table:

    def __init__(self):

        """
        Root database
        """
        # Household
        self.scenarios = "Scenarios"
        self.region = "Region"
        self.person_list = "PersonList"
        self.building = "Building"
        self.appliance_list = "ApplianceList"
        self.boiler = "Boiler"
        self.space_heating_tank = "SpaceHeatingTank"
        self.air_conditioner = "AirConditioner"
        self.hot_water_tank = "HotWaterTank"
        self.pv = "HouseholdPV"
        self.battery = "Battery"
        self.vehicle = "Vehicle"
        self.behavior = "Behavior"
        self.hot_water_demand = "HotWaterDemand"
        self.electricity_demand = "ElectricityDemand"

        # Environment
        self.electricity_price = "EnvironmentElectricityPrice"
        self.feedin_tariff = "EnvironmentFeedinTariff"

        # tables with downloaded data for countries:
        self.temperature = "Temperature"
        self.radiation = "Radiation"
        self.pv_generation = "PV_generation"


class Column:

    pass


