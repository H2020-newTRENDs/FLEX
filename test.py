import numpy as np
import pandas as pd
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_Database import DB
from A_Infrastructure.A2_ToolKits.b_Table import Table





if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    Country = DB().read_DataFrame(REG().ID_Country, CONN)



