
class Table:

    def __init__(self):

        """
        Root database
        """
        # Household
        self.scenarios = "Scenarios"
        self.region = "HouseholdRegion"
        self.person_list = "HouseholdPersonList"
        self.building = "HouseholdBuilding"
        self.appliance_list = "HouseholdApplianceList"
        self.boiler = "HouseholdBoiler"
        self.space_heating_tank = "HouseholdSpaceHeatingTank"
        self.air_conditioner = "HouseholdAirConditioner"
        self.hot_water_tank = "HouseholdHotWaterTank"
        self.pv = "HouseholdPV"
        self.battery = "HouseholdBattery"
        self.vehicle = "HouseholdVehicle"
        self.behavior = "HouseholdBehavior"
        self.demand = "HouseholdDemand"

        # Environment
        self.electricity_price = "EnvironmentElectricityPrice"
        self.feedin_tariff = "EnvironmentFeedinTariff"

        # tables with downloaded data for countries:
        self.temperature = "Temperature"
        self.radiation = "Radiation"
        self.pv_generation = "PVGeneration"


class Column:

    pass


