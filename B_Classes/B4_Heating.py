from C_Model_Operation.C1_REG import REG_Var


class SpaceHeatingSystem:

    def __init__(self, para_series):
        # self.ID_SpaceHeatingBoilerType = para_series["ID_SpaceHeatingBoilerType"]
        self.Name_SpaceHeatingPumpType = para_series[REG_Var().Name_SpaceHeatingPumpType]
        # self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.HeatPumpMaximalThermalPower = para_series[REG_Var().HeatPumpMaximalThermalPower]
        self.CarnotEfficiencyFactor = para_series[REG_Var().CarnotEfficiencyFactor]
        self.HeatingElementPower = para_series[REG_Var().HeatingElementPower]
        self.ID_SpaceHeatingSystem = para_series[REG_Var().ID_SpaceHeatingSystem]
        # self.PumpConsumptionRate = para_series["PumpConsumptionRate"]
        # self.MaximalPowerFloorHeating = para_series["MaximalPowerFloorHeating"]


class SpaceHeatingTank:
    def __init__(self, para_series):
        self.ID_SpaceHeatingTank = para_series[REG_Var().ID_SpaceHeatingTank]
        self.TankSize = para_series[REG_Var().TankSize]
        self.TankSurfaceArea = para_series[REG_Var().TankSurfaceArea]
        self.TankLoss = para_series[REG_Var().TankLoss]

        self.TankStartTemperature = para_series[REG_Var().TankStartTemperature]
        self.TankMaximalTemperature = para_series[REG_Var().TankMaximalTemperature]
        self.TankMinimalTemperature = para_series[REG_Var().TankMinimalTemperature]
        self.TankSurroundingTemperature = para_series[REG_Var().TankSurroundingTemperature]


class DHWTank:
    def __init__(self, para_series):
        self.ID_DHWTankType = para_series[REG_Var().ID_DHWTank]
        self.DHWTankSize = para_series[REG_Var().DHWTankSize]
        self.DHWTankSurfaceArea = para_series[REG_Var().DHWTankSurfaceArea]
        self.DHWTankLoss = para_series[REG_Var().DHWTankLoss]

        self.DHWTankStartTemperature = para_series[REG_Var().DHWTankStartTemperature]
        self.DHWTankMaximalTemperature = para_series[REG_Var().DHWTankMaximalTemperature]
        self.DHWTankMinimalTemperature = para_series[REG_Var().DHWTankMinimalTemperature]
        self.DHWTankSurroundingTemperature = para_series[REG_Var().DHWTankSurroundingTemperature]






