
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from C_Model.C1_TableGenerator.C11_Ope_TableGenerator import Ope_TableGenerator
from C_Model.C2_OperationOptimization.C21_OperationOptimization import OperationOptimization
from B_Classes.B2_Building import Building, HeatingCooling_noDR

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    Ope_TableGenerator(CONN).run()
    # OperationOptimization(CONN).run()






    #table_name = REG().Sce_ID_TimeStructure
    #table = DB().read_DataFrame(table_name, CONN, ID_Day=5)
    #table = DB().read_DataFrame(table_name, CONN)
    #table_select = table.iloc[10]["ID_Day"]
    #table_select = table.loc[table["ID_Day"] == 5 ]
    #print(table_select)
    #print(type(table_select))
    #print(table)

    #PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_BasePhotovoltaicProfile, CONN)
    #print(PhotovoltaicProfile.BasePhotovoltaicProfile[2000])        #value of hour 2001
    #x = PhotovoltaicProfile.BasePhotovoltaicProfile
    #print(sum(x))

    #Household = DB().read_DataFrame(REG().Gen_OBJ_ID_Household, CONN)
    #HouseholdID = Household.ID[0]
    #HouseholdID.

    #print(HouseholdID)

    # Run calculation on heat and cooling loads for all households without optimization:
    import time

    start = time.process_time()
    B = HeatingCooling_noDR(DB().read_DataFrame(REG().ID_BuildingOption, CONN))
    Temperature_outside = DB().read_DataFrame(REG().Sce_Weather_Temperature_test, CONN)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR = B.ref_HeatingCooling(Temperature_outside)
    print(time.process_time() - start)








