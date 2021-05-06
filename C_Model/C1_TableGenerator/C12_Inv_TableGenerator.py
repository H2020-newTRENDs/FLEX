
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class Inv_TableGenerator:

    def __init__(self, conn):
        self.Conn = conn
