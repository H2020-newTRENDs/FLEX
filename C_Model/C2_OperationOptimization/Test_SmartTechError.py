import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.util.infeasible import log_infeasible_constraints
import random

from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from B_Classes.B1_Household import Household
from A_Infrastructure.A1_Config.A11_Constants import CONS


# toDo
# (1) Create Scenario Cases
# (2) integrate solar gains
# (3) include cooling
# (4) Function into classes
# (5) analyse model
# (6) PV, battery
# (7) ... rest of optimization


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

        # later stage: this is included in the scenario
        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature_test, self.Conn)
        self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)

        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_PhotovoltaicProfile, self.Conn)
        self.DishWasherDays = DB().read_DataFrame(REG().Gen_Sce_DishWasherUseDays, self.Conn)
        # you can import all the necessary tables into the memory here.
        # Then, it can be faster when we run the optimization problem for many "household - environment" combinations.

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
        ID_Household = Household.ID
        print(ID_Household)

        Environment = self.gen_Environment(environment_id)
        ID_Environment = Environment.ID
        ID_ElectricityPriceType = Environment.ID_ElectricityPriceType

        # the data of the simulation are indexed for 1 year, start and stoptime in visualization
        HoursOfSimulation = 8760

        # used for the data import in pyomo
        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary

        CarAtHome = create_dict([1] * HoursOfSimulation)
        V2G = create_dict([1] * HoursOfSimulation)

        #print(CarAtHome)

        # PV, battery and electricity load profile
        LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH.to_numpy()

        ### From theoretical ON days to ON hours

        DishWasherWorkingDays = self.DishWasherDays.DishwasherWorkingDays.to_numpy()

        DishWasherTheorecitalHours = []
        for day in DishWasherWorkingDays:
            if day == 1:
                DishWasherTheorecitalHours.extend(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

            elif day == 0:
                DishWasherTheorecitalHours.extend(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

        SumOfPV = round(sum(PhotovoltaicProfileHH), 2)
        print(SumOfPV)
        SumOfLoad = round(sum(self.LoadProfile.BaseElectricityProfile), 2)
        print(SumOfLoad)

        # fixed starting values:
        T_TankStart = 45  # °C
        # min,max tank temperature for boundary of energy
        T_TankMax = 50  # °C
        T_TankMin = 25  # °C
        # sourounding temp of tank
        T_TankSourounding = 20  # °C
        # starting temperature of thermal mass 20°C
        CWater = 4200 / 3600

        # Parameters of SpaceHeatingTank
        # Mass of water in tank
        M_WaterTank = Household.SpaceHeating.TankSize
        # Surface of Tank in m2
        A_SurfaceTank = Household.SpaceHeating.TankSurfaceArea
        # insulation of tank, for calc of losses
        U_ValueTank = Household.SpaceHeating.TankLoss

        # Simulation of Building: konditionierte Nutzfläche
        Af = Household.Building.Af
        # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
        Atot = 4.5 * Af  # 7.2.2.2
        # Airtransfercoefficient
        Hve = Household.Building.Hve
        # Transmissioncoefficient wall
        Htr_w = Household.Building.Htr_w
        # Transmissioncoefficient opake Bauteile
        Hop = Household.Building.Hop
        # Speicherkapazität J/K
        Cm = Household.Building.CM_factor * Af
        # wirksame Massenbezogene Fläche [m^2]
        Am = float(Household.Building.Am_factor) * Af
        # internal gains
        Qi = Household.Building.spec_int_gains_cool_watt * Af

        # Kopplung Temp Luft mit Temp Surface Knoten s
        his = np.float_(3.45)  # 7.2.2.2
        # kopplung zwischen Masse und  zentralen Knoten s (surface)
        hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte W/K
        Htr_is = his * Atot
        # Equ. C.1
        PHI_ia = 0.5 * Qi
        # Equ. C.6
        Htr_1 = 1 / (1 / Hve + 1 / Htr_is)
        # Equ. C.7
        Htr_2 = Htr_1 + Htr_w
        # Equ.C.8
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)

        # Solar Gains, but to small calculated: in W/m²
        Q_sol = self.Radiation.loc[self.Radiation["ID_Country"] == 5].loc[:, "Radiation"].to_numpy()

        # (1) Scenario Case: select ElectricityPriceType
        if ID_ElectricityPriceType == 1:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 1].loc[:,
                               'HourlyElectricityPrice'].to_numpy()
            ElectricityPrice = list(ElectricityPrice)
            ElectricityPrice = ElectricityPrice * HoursOfSimulation

        elif ID_ElectricityPriceType == 2:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 2].loc[:,
                               'HourlyElectricityPrice'].to_numpy()

        # Select COP Air(dynamic) or COP Water
        if Household.SpaceHeating.ID_SpaceHeatingBoilerType == 1:
            Tout = self.Weather.Temperature
            COP = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 1].loc[:,
                  'COP_ThermalStorage'].to_numpy()
            COPtemp = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 1].loc[:,
                      'TemperatureEnvironment'].to_numpy()
            ListOfDynamicCOP = []
            j = 0

            for i in range(0, len(list(Tout))):
                for j in range(0, len(list(COPtemp))):
                    if COPtemp[j] < Tout[i]:
                        continue
                    else:
                        ListOfDynamicCOP.append(COP[j])
                    break
        elif Household.SpaceHeating.ID_SpaceHeatingBoilerType == 2:
            COP = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 2].loc[:,
                  'COP_ThermalStorage'].to_numpy()
            COP = list(COP)
            COP = COP * HoursOfSimulation
            ListOfDynamicCOP = COP

        # model for optimisation
        m = pyo.AbstractModel()

        # parameters
        m.t = pyo.RangeSet(1, HoursOfSimulation)
        # price
        m.ElectricityPrice = pyo.Param(m.t, initialize=create_dict(ElectricityPrice))
        # solar gains:
        m.Q_solar = pyo.Param(m.t, initialize=create_dict(Q_sol))
        # outside temperature
        m.T_outside = pyo.Param(m.t, initialize=create_dict(self.Weather.Temperature.to_numpy()))
        # m.T_outside = pyo.Param(m.t, initialize=create_dict(T_Out))
        m.COP_dynamic = pyo.Param(m.t, initialize=create_dict(ListOfDynamicCOP))
        # electricity load profile
        m.LoadProfile = pyo.Param(m.t, initialize=create_dict(LoadProfile))
        # PV profile
        m.PhotovoltaicProfile = pyo.Param(m.t, initialize=create_dict(PhotovoltaicProfile))

        m.CarStatus = pyo.Param(m.t, initialize=CarAtHome)
        m.V2B = pyo.Param(m.t, initialize=V2G)
        m.TheoreticalHours = pyo.Param(m.t, within=pyo.Binary, initialize=create_dict(DishWasherTheorecitalHours))

        # Variables PV and battery
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 6))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        # m.Bat2Grid

        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 35))

        m.DishWasher = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasherEnergy = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 3))

        # objective

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * 0.08 for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        # (1) Energy balance of supply == demand
        def calc_ElectricalEnergyBalance(m, t):
            return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] + m.EVDischarge[t] == m.Feedin[t] + \
                   m.Load[t] + m.BatCharge[t] + m.EVCharge[t]

        m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

        # (2)
        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2EV[t] * m.CarStatus[t]

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (3)
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] + m.PV2EV[t] * m.CarStatus[t] == m.PhotovoltaicProfile[t]

        m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (4)
        def calc_SumOfFeedin(m, t):
            return m.Feedin[t] == m.PV2Grid[t]

        m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (5)
        def calc_SupplyOfLoads(m, t):
            return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.V2B[t] * m.CarStatus[t] == m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)


        ### new ####################################

        # (6b) Fill DishWasherEnergy
        def calc_DishWasherEnergy(m, t):
            if m.t[t] == 1:
                return m.DishWasher[t] == 0
            elif m.TheoreticalHours[t] == 0 and m.TheoreticalHours[t - 1] == 1:
                return m.DishWasherEnergy[t] == m.DishWasherEnergy[t - 1] -3
            elif m.TheoreticalHours[t] == 1:
                return m.DishWasherEnergy[t] == m.DishWasherEnergy[t - 1] + m.DishWasher[t] * m.TheoreticalHours[t]
            return m.DishWasherEnergy[t] == 0


        m.calc_DishWasherEnergy = pyo.Constraint(m.t, rule=calc_DishWasherEnergy)

        # (6)
        def calc_SumOfLoads(m, t):
            if m.t[t] == 1:
                return pyo.Constraint.Skip
            else:
                return m.Load[t] == m.LoadProfile[t] + m.DishWasher[t]*0.36
        m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)

        ### new #########################################

        # (7)
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.BatDischarge[t] == 0
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarStatus[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8)
        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.EV2Bat[t] * m.V2B[t]

        m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (9)
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * 0.95 - m.BatDischarge[t] * 1.05

        m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)

        # (10)
        def calc_EVCharge(m, t):
            if m.CarStatus[t] == 0:
                return m.EVCharge[t] == 0
            else:
                return m.EVCharge[t] * m.CarStatus[t] == m.Grid2EV[t] * m.CarStatus[t] + m.PV2EV[t] * m.CarStatus[t] + \
                       m.Bat2EV[t] * m.CarStatus[t]

        m.calc_EVCharge = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (11)
        def calc_EVDischarge(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0
            elif m.CarStatus[t] == 0 and m.V2B[t] == 1:
                return m.EVDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.EVDischarge[t] == 0
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.V2B[t] * m.CarStatus[t] + m.EV2Bat[t] * m.V2B[t] * \
                       m.CarStatus[t]

        m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (12)
        def calc_EVSoC(m, t):
            if m.t[t] == 1:
                return m.EVSoC[t] == 0
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + (m.EVCharge[t] * m.CarStatus[t] * 0.95) - (
                        m.EVDischarge[t] * m.CarStatus[t] * 1.05)

        m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)

        instance = m.create_instance(report_timing=True)
        # opt = pyo.SolverFactory("glpk")
        opt = pyo.SolverFactory("gurobi")
        results = opt.solve(instance, tee=True)
        instance.display("./log.txt")
        print(results)
        # return relevant data
        return instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile

    def run(self):
        for household_id in range(37000, 37001):
            for environment_id in range(3, 4):
                instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile = self.run_Optimization(
                    household_id,
                    environment_id)
                return instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile


# create plots to visualize results price
def show_results(instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater,
                 PhotovoltaicProfile):
    # Visualization
    starttime = 500
    endtime = 600

    # battery, grid and PV

    StorageFillLevel = np.array(list(instance.BatSoC.extract_values().values())[starttime: endtime])
    BatDischarge = np.array(list(instance.BatDischarge.extract_values().values())[starttime: endtime])

    BatCharge = np.array(list(instance.BatCharge.extract_values().values())[starttime: endtime])
    Load = np.array(list(instance.Load.extract_values().values())[starttime: endtime])

    TheoreticalHours = np.array(list(instance.TheoreticalHours.extract_values().values())[starttime: endtime])
    DishWasher = np.array(list(instance.DishWasher.extract_values().values())[starttime: endtime])


    total_cost = instance.OBJ()

    # Plot
    x_achse = np.arange(starttime, endtime)

    plt.figure()
    #plt.plot(x_achse, StorageFillLevel, linewidth=1, label='StorageFillLevel', color='gray')
    #plt.bar(x_achse, BatDischarge, label='GridCover', color='red')
    #plt.bar(x_achse, BatCharge, label='PhotovoltaicFeedin', color='green')
    plt.plot(x_achse, Load, label='Load', color='red')

    plt.bar(x_achse, TheoreticalHours, label='TheoreticalHours', color='grey', alpha=0.10)
    plt.bar(x_achse, DishWasher, label='DishWasher', color='orange', alpha=0.8)



    plt.legend()
    plt.xlabel('Hour of year')
    plt.ylabel('Power in kW')
    plt.title('Costs: ' + str(round(total_cost, 2)) + ' €')
    plt.show()


if __name__ == "__main__":
    A = OperationOptimization(DB().create_Connection(CONS().RootDB))
    instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile = A.run()
    show_results(instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater,
                 PhotovoltaicProfile)
