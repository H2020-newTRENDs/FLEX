
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class Battery:

    """
    Data type:
    (1)
    """

    def __init__(self, household, para_series, conn):
        self.Household = household
        self.Conn = conn
