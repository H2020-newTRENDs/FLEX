from C_Model_Operation.C1_REG import REG_Var


class SpaceHeatingSystem:

    def __init__(self, para_series):
        # TODO change the strings in para_series[STRING] to REG_VAR()? and then also in table generator, so we are consistent in every part
        # self.ID_SpaceHeatingBoilerType = para_series["ID_SpaceHeatingBoilerType"]
        self.Name_SpaceHeatingPumpType = para_series['Name_SpaceHeatingPumpType']
        # self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.HeatPumpMaximalThermalPower = para_series["HeatPumpMaximalThermalPower"]
        self.CarnotEfficiencyFactor = para_series['CarnotEfficiencyFactor']
        self.HeatingElementPower = para_series['HeatingElementPower']
        self.ID_SpaceHeatingSystem = para_series["ID_SpaceHeatingSystem"]
        # self.PumpConsumptionRate = para_series["PumpConsumptionRate"]
        # self.MaximalPowerFloorHeating = para_series["MaximalPowerFloorHeating"]


class SpaceHeatingTank:
    def __init__(self, para_series):
        self.ID_SpaceHeatingTank = para_series[REG_Var().ID_SpaceHeatingTank]
        self.TankSize = para_series["TankSize"]
        self.TankSurfaceArea = para_series["TankSurfaceArea"]
        self.TankLoss = para_series["TankLoss"]

        self.TankStartTemperature = para_series["TankStartTemperature"]
        self.TankMaximalTemperature = para_series["TankMaximalTemperature"]
        self.TankMinimalTemperature = para_series["TankMinimalTemperature"]
        self.TankSurroundingTemperature = para_series["TankSurroundingTemperature"]


class DHWTank:
    def __init__(self, para_series):
        self.ID_DHWTankType = para_series[REG_Var().ID_DHWTank]
        self.DHWTankSize = para_series["DHWTankSize"]
        self.DHWTankSurfaceArea = para_series["DHWTankSurfaceArea"]
        self.DHWTankLoss = para_series["DHWTankLoss"]

        self.DHWTankStartTemperature = para_series["DHWTankStartTemperature"]
        self.DHWTankMaximalTemperature = para_series["DHWTankMaximalTemperature"]
        self.DHWTankMinimalTemperature = para_series["DHWTankMinimalTemperature"]
        self.DHWTankSurroundingTemperature = para_series["DHWTankSurroundingTemperature"]






