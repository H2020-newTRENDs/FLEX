
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class Building:

    def __init__(self, para_series):
        self.ID_BuildingType = para_series["ID_BuildingType"]
        self.ID_BuildingAgeClass = para_series["ID_BuildingAgeClass"]
        self.FloorArea = para_series["FloorArea"]
        # other attributes to be added
        pass

    def calc_IndoorTemperature(self, Energy_TankToRoom_t, OutdoorTemperature_t):

        """
        :param Energy_TankToRoom_t: float
        :param OutdoorTemperature_t: float
        :return: IndoorTemperature_tp1: float
        """

        IndoorTemperature = []
        return IndoorTemperature