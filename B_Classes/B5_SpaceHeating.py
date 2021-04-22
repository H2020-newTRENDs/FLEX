
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
        self.SpaceHeatingTankType = para_series["ID_SpaceHeatingTankType"]
