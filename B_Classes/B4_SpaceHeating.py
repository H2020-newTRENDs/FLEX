class SpaceHeating:

    def __init__(self, para_series):

        # self.ID_SpaceHeatingBoilerType = para_series["ID_SpaceHeatingBoilerType"]
        self.Name_SpaceHeatingPumpType = para_series['Name_SpaceHeatingPumpType']
        # self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.HeatPumpMaximalThermalPower = para_series["HeatPumpMaximalThermalPower"]
        self.CarnotEfficiencyFactor = para_series['CarnotEfficiencyFactor']
        self.HeatingElementPower = para_series['HeatingElementPower']

        self.ID_SpaceHeatingTankType = para_series["ID_SpaceHeatingTankType"]
        self.TankSize = para_series["TankSize"]
        self.TankSurfaceArea = para_series["TankSurfaceArea"]
        self.TankLoss = para_series["TankLoss"]

        self.TankStartTemperature = para_series["TankStartTemperature"]
        self.TankMaximalTemperature = para_series["TankMaximalTemperature"]
        self.TankMinimalTemperature = para_series["TankMinimalTemperature"]
        self.TankSurroundingTemperature = para_series["TankSurroundingTemperature"]

        self.ID_SpaceHeatingPumpType = para_series["ID_SpaceHeatingPumpType"]
        # self.PumpConsumptionRate = para_series["PumpConsumptionRate"]
        # self.MaximalPowerFloorHeating = para_series["MaximalPowerFloorHeating"]






