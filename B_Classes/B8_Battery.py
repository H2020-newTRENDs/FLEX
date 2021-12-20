from C_Model_Operation.C1_REG import REG_Var


class Battery:

    """
    Data type:
    (1)
    """

    def __init__(self, para_series):
        self.ID_BatteryType = para_series[REG_Var().ID_Battery]
        self.Capacity = para_series[REG_Var().Capacity]
        self.ChargeEfficiency = para_series[REG_Var().ChargeEfficiency]
        self.MaxChargePower = para_series[REG_Var().MaxChargePower]
        self.DischargeEfficiency = para_series[REG_Var().DischargeEfficiency]
        self.MaxDischargePower = para_series[REG_Var().MaxDischargePower]
        # self.Grid2Battery = para_series['Grid2Battery']



