import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints


from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from B_Classes.B1_Household import Household

class OperationOptimization:

    """
    Intro

    Optimize the prosumaging behavior of all representative "household - environment" combinations.
    Read all big fundamental tables at the beginning and keep them in the memory.
    (1) Appliances parameter table
    (2) Energy price
    Go over all the aspects and summarize the flexibilities to be optimized.
    Formulate the final optimization problem.
    """

    def __init__(self, conn):
        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG().Gen_Sce_ID_Environment, self.Conn)
        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_BasePhotovoltaicProfile, self.Conn)

        # self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        # self.Temperature = DB().read_DataFrame(REG().Sce_Weather_Temperature, self.Conn)
        # self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)
        # you can import all the necessary tables into the memory here.
        # Then, it can be faster when we run the optimization problem for many "household - enviornment" combinations.

    def gen_Household(self, row_id):
        ParaSeries = self.ID_Household.iloc[row_id]
        HouseholdAgent = Household(ParaSeries, self.Conn)
        return HouseholdAgent

    def gen_Environment(self, row_id):
        EnvironmentSeries = self.ID_Environment.iloc[row_id]
        return EnvironmentSeries

    def stateTransition_SpaceHeatingTankTemperature(self, Household,
                                                    energy_from_boiler_to_tank,
                                                    energy_from_tank_to_room):
        TankTemperature = Household.SpaceHeating.calc_TankTemperature(energy_from_boiler_to_tank,
                                                                      energy_from_tank_to_room)

        return TankTemperature

    def run_Optimization(self, household_id, environment_id):
        Household = self.gen_Household(household_id)
        print(Household.SpaceHeating.TankSize)
        # Environment = self.gen_Environment(environment_id)
        Householdtype = Household.ID_HouseholdType
        print(Householdtype)

        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary



        # toDo
        # change values to dynamic from db
        # add battery losses
        # try to change function to classes



        ID_Hour = create_dict(self.TimeStructure.ID_Hour)
        LoadProfile = create_dict(self.LoadProfile.BaseElectricityProfile)
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.BasePhotovoltaicProfile
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = create_dict(PhotovoltaicProfileHH)

        SumOfPV = round(sum(PhotovoltaicProfileHH), 2)
        print(SumOfPV)
        SumOfLoad = round(sum(self.LoadProfile.BaseElectricityProfile),2)
        print(SumOfLoad)


        #print(Household.PV.PVPower)
        #print(Household.Battery.Capacity)
        #print(PhotovoltaicProfile[10])

        HoursOfSimulation = 8760
        FiT = create_dict([0.0792] * HoursOfSimulation)

        #print(FiT)
        GridPrice = create_dict([0.30] * HoursOfSimulation)

        m = pyo.AbstractModel()

        # Parameters
        m.t = pyo.RangeSet(1, HoursOfSimulation)
        m.LoadProfile = pyo.Param(m.t, initialize=LoadProfile)
        m.PhotovoltaicProfile = pyo.Param(m.t, initialize=PhotovoltaicProfile)
        m.FiT = pyo.Param(m.t, initialize=FiT)
        m.GridPrice = pyo.Param(m.t, initialize=GridPrice)


        # Variables
        m.StorageFillLevel = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.Capacity))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
        m.GridCover = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))
        m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0,16))
        m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0,16))


        # objective
        def minimize_cost(m):

            rule = sum(m.GridCover[t] * m.GridPrice[t] - m.PhotovoltaicFeedin[t] * m.FiT[t] for t in m.t)
            return rule
        m.OBJ = pyo.Objective(rule=minimize_cost)


        def calc_StorageFillLevel(m, t):
            if t == 1:
                return m.StorageFillLevel[t] == 1
            else:
                return m.StorageFillLevel[t] == m.StorageFillLevel[t-1] + m.BatCharge[t] - m.BatDischarge[t]
        m.calc_StorageFillLevel = pyo.Constraint(m.t, rule=calc_StorageFillLevel)

        def calc_BatteryCharge(m, t):
#            m.BatCharge[t] == m.PhotovoltaicProfile[t] - m.PhotovoltaicDirect[t] - m.PhotovoltaicFeedin[t]
            return m.BatCharge[t] == m.PhotovoltaicProfile[t] - m.PhotovoltaicDirect[t] - m.PhotovoltaicFeedin[t]
        m.calc_BatteryCharge = pyo.Constraint(m.t, rule=calc_BatteryCharge)

        def calc_BatteryDischarge(m,t):
#            m.BatDischarge[t] == m.LoadProfile[t] - m.GridCover[t] - m.PhotovoltaicDirect[t]
            return m.BatDischarge[t] == m.LoadProfile[t] - m.GridCover[t] - m.PhotovoltaicDirect[t]
        m.calc_BatterDischarge = pyo.Constraint(m.t, rule=calc_BatteryDischarge)



        instance = m.create_instance(report_timing=True)
        opt = pyo.SolverFactory("glpk")
        results = opt.solve(instance, tee=True)
        print(results)

        yearlyCost = round(instance.OBJ(), 2)
        print(yearlyCost)





        # Visulaization

        PhotovoltaicProfile = [PhotovoltaicProfile[i] for i in range(1, 8760)]
        LoadProfile = [LoadProfile[i] for i in range(1, 8760)]
        StorageFillLevel = [instance.StorageFillLevel[t]() for t in range(1, 8760)]

        plt.figure()
        plt.subplot(211)
        plt.plot(PhotovoltaicProfile, label='PV', linewidth = 0.5, color = 'orange')
        plt.plot(LoadProfile, label = 'HH Load', linewidth = 0.5, color = 'blue')
        plt.title('Household ID: ' +str(household_id) + ' with Exo_Profiles with HH-Demand ' +str(SumOfLoad) + ' kWh / year and PV generation ' + \
                  str(SumOfPV) + ' kWh / year')
        plt.ylabel('Power in kW')
        plt.xlabel('Time in h')
        plt.legend()

        plt.subplot(212)
        plt.plot(StorageFillLevel, linewidth = 0.5)
        plt.title('StorageFillLevel - Results: Yearly electricity cost: ' +str(yearlyCost) + ' € ' + 'with PV ' + \
                  str(Household.PV.PVPower) + ' kWp and Battery ' + str(Household.Battery.Capacity) + 'kWh')
        plt.ylabel('Storage fill level in kWh')
        plt.xlabel('Time in h')
        plt.show()

        pass








    def run(self):
        for household_id in range(42, 43):
            for environment_id in range(0, 1):
                self.run_Optimization(household_id, environment_id)
        pass



