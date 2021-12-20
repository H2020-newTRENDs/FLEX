from C_Model_Operation.C1_REG import REG_Var


class PV:

    def __init__(self, para_series):
        self.ID_PV = para_series[REG_Var().ID_PV]
        self.PVPower = para_series[REG_Var().PVPower]
        self.PVPower_unit = para_series["PVPower_unit"]

