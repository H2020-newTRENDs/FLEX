
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class SpaceHeating:

    """
    Data type:
    (1)
    """

    def __init__(self, household, para_series, conn):

        self.Household = household
        self.Conn = conn
        self.SpaceHeatingAdoptionStatus = bool(para_series["ID_SpaceHeatingAdoptionStatus"])
        self.SpaceHeatingType = para_series["ID_SpaceHeatingType"]
        [self.ID_EnergyCarrier, self.Efficiency] = self.init_SpaceHeating()
        self.SpaceHeatingTankType = para_series["ID_SpaceHeatingTankType"]
        self.TankSize = DB().read_ExoTableValue(REG().Exo_SpaceHeatingTankSize, self.Conn, ID_SpaceHeatingTankType=self.SpaceHeatingTankType)
        self.TankSurfaceArea = DB().read_ExoTableValue(REG().Exo_SpaceHeatingTankSurfaceArea, self.Conn, ID_SpaceHeatingTankType=self.SpaceHeatingTankType)

    def init_SpaceHeating(self):

        SpaceHeatingTable = DB().read_DataFrame(REG().Exo_SpaceHeatingEfficiency, self.Conn, ID_SpaceHeatingType=self.SpaceHeatingType)
        ID_EnergyCarrier = SpaceHeatingTable.iloc[0]["ID_EnergyCarrier"]
        Efficiency = SpaceHeatingTable.iloc[0]["Value"]

        return [ID_EnergyCarrier, Efficiency]







