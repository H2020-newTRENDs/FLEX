from C_Model_Operation.C1_REG import REG_Var


class SpaceCooling:

    def __init__(self, para_series):
        self.ID_SpaceCooling = para_series[REG_Var().ID_SpaceCooling]
        # self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.SpaceCoolingEfficiency = para_series[REG_Var().SpaceCoolingEfficiency]
        # self.SpaceCoolingEfficiency_unit = para_series["SpaceCoolingEfficiency_unit"]
        self.SpaceCoolingPower = para_series[REG_Var().SpaceCoolingPower]
        self.SpaceCoolingPower_unit = para_series["SpaceCoolingPower_unit"]
        # self.AdoptionStatus = para_series["AdoptionStatus"]