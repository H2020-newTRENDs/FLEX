from C_Model_Operation.C1_REG import REG_Var


class Battery:

    """
    Data type:
    (1)
    """

    def __init__(self, para_series):
        self.ID_BatteryType = para_series[REG_Var().ID_Battery]
        self.Capacity = para_series["Capacity"]
        self.ChargeEfficiency = para_series["ChargeEfficiency"]
        self.MaxChargePower = para_series["MaxChargePower"]
        self.DischargeEfficiency = para_series["DischargeEfficiency"]
        self.MaxDischargePower = para_series["MaxDischargePower"]
        # self.Grid2Battery = para_series['Grid2Battery']



