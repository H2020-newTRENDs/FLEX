
import pandas as pd
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_Database import DB

class Table:

    def return_ValueTableIDs(self, table):
        column_name = list(table.columns)[:-2]
        return column_name

    def return_FuncType(self, func):

        def func_Multiply(x, y):
            result = x * y
            return result

        if type(func) != str:
            return func
        elif func == "Multiply":
            return func_Multiply
        else:
            pass

    def calc_VT12toVT3(self, VT1, VT2, func, VT3_Unit, VT3_Name, conn):

        VT1_IDs = self.return_ValueTableIDs(VT1)
        VT2_IDs = self.return_ValueTableIDs(VT2)
        Intersection_IDs_set = set(VT1_IDs).intersection(set(VT2_IDs))
        Intersection_IDs = list(Intersection_IDs_set)
        VT1_ExclusiveIDs = list(set(VT1_IDs) - Intersection_IDs_set)

        VT3_column = VT2_IDs + VT1_ExclusiveIDs + ["Unit", "Value"]
        VT3 = []
        for row_1 in range(0, len(VT1)):
            row_VT1IntersectionIDKey = VT1.iloc[row_1][Intersection_IDs].tolist()
            VT2_temp = VT2
            for num in range(0, len(Intersection_IDs)):
                ID = Intersection_IDs[num]
                IDKey = row_VT1IntersectionIDKey[num]
                VT2_temp = VT2_temp.loc[VT2_temp[ID] == IDKey]
            VT2_reduce = VT2_temp.reset_index()
            if len(VT2_reduce) == 0:
                pass
            else:
                VT1_value = VT1.iloc[row_1]["Value"]
                for row_2 in range(0, len(VT2_reduce)):
                    VT2_value = VT2_reduce.iloc[row_2]["Value"]
                    VT3_value = self.return_FuncType(func)(VT1_value, VT2_value)
                    VT3.append(VT2_reduce.iloc[row_2][VT2_IDs].tolist() + VT1_ExclusiveIDs + [VT3_Unit, VT3_value])

        DB().write_DataFrame(VT3, VT3_Name, VT3_column, conn)




