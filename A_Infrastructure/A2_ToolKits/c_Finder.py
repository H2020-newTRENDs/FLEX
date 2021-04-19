
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_Database import DB

class Find:

    def __init__(self):
        self.Conn = DB().create_Connection(CONS().RootDB)

    def country_ID2Code(self, id):

        CountryTable = DB().read_DataFrame(REG().ID_Country, self.Conn)
        CountryCode = CountryTable.iloc[int(id - 1)]["Name"]

        return CountryCode