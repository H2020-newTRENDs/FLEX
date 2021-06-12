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

        # new
        self.DishWasherHours = DB().read_DataFrame(REG().Gen_Sce_DishWasherHours, self.Conn)
        self.Sce_Demand_DishWasher = DB().read_DataFrame(REG().Sce_Demand_DishWasher, self.Conn)

        self.WashingMachineHours = DB().read_DataFrame(REG().Gen_Sce_WashingMachineHours, self.Conn)
        self.Sce_Demand_WashingMachine = DB().read_DataFrame(REG().Sce_Demand_WashingMachine, self.Conn)

        self.DryerHours = DB().read_DataFrame(REG().Gen_Sce_DryerHours, self.Conn)
        self.Sce_Demand_Dryer = DB().read_DataFrame(REG().Sce_Demand_Dryer, self.Conn)


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

        # PV, battery and electricity load profile
        LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH.to_numpy()

        ########### new for Smart Technologies #####################################################

        # This is 1, only if there is no Dishwasher in the household, then 0
        DishWasherAdoption = 1

        DishWasherTheoreticalHours = (self.DishWasherHours.DishWasherHours.to_numpy()) * DishWasherAdoption
        HourOfDay = self.TimeStructure.ID_DayHour.to_numpy()
        DishWasherCycle = int(self.Sce_Demand_DishWasher.DishWasherCycle)
        DishWasherDuration = int(self.Sce_Demand_DishWasher.DishWasherDuration)

        # This is for testing a lower duration, max = 3
        # DishWasherDuration= 1

        DishWasherPower = Household.ApplianceGroup.DishWasherPower
        DishWasherSmartStatus = Household.ApplianceGroup.DishWasherShifting

        # This is for setting a smart status manually, 0 or 1
        DishWasherSmartStatus = 1


        # -----------------------------------------------------------------------------

        # This is 1, only if there is no Dishwasher in the household, then 0
        WashingMachineAdoption = 1

        WashingMachineTheoreticalHours = (self.WashingMachineHours.WashingMachineHours.to_numpy()) *WashingMachineAdoption
        WashingMachineCycle = int(self.Sce_Demand_WashingMachine.WashingMachineCycle)
        WashingMachineDuration = int(self.Sce_Demand_WashingMachine.WashingMachineDuration)

        # This is for testing a lower duration, max = 3
        # WashingMachineDuration= 3

        WashingMachinePower = Household.ApplianceGroup.WashingMachinePower
        WashingMachineSmartStatus = Household.ApplianceGroup.WashingMachineShifting

        # This is for setting a smart status manually, 0 or 1
        WashingMachineSmartStatus = 1

        # -----------------------------------------------------------------------------
        # This is 1, only if there is no Dishwasher in the household, then 0
        DryerAdoption = 1

        # toDo:
        # () Done: Last day of year? With or 8759 or 8760 ...
        # () Done: changing Demand Profiles with different duration hours: Between 1-3 Hours is in code
        # () Done: if valid: change UseDays to UseHours in databank
        # () Done: Smart ON / OFF: use fixed use time, all functions included?
        # () Done: Adoption to 1 and 0
        # () Extend to WaschingMachine
        # (-) How to use Dryer? Profile?
        # (-) Extend to Dryer
        # (-) Minimize LoadProfile
        # (-) create Gen_Sce_Appliances... + extent Environment with DemandProfiles of SmartApp

        #################################################################################################

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

        ########## new Parameter for Smart Technologies #####################################

        m.DishWasherTheoreticalHours = pyo.Param(m.t, within=pyo.Binary,
                                                 initialize=create_dict(DishWasherTheoreticalHours))
        m.HourOfDay = pyo.Param(m.t, initialize=create_dict(HourOfDay))

        m.WashingMachineTheoreticalHours = pyo.Param(m.t, within=pyo.Binary,
                                                 initialize=create_dict(WashingMachineTheoreticalHours))

        ######################################################################################

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

        ###### new Variables for Smart Technologies ###################################################

        m.DishWasher1 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher2 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher3 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasherStart = pyo.Var(m.t, within=pyo.Binary)
        # ------------------------------------------------------

        m.WashingMachine1 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine2 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine3 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachineStart = pyo.Var(m.t, within=pyo.Binary)

        m.Dryer1 = pyo.Var(m.t, within=pyo.Binary)
        m.Dryer2 = pyo.Var(m.t, within=pyo.Binary)

        ###############################################################################################

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
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2EV[t] * m.CarAtHomeStatus[t]

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (3)
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] == m.PhotovoltaicProfile[t]

        m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (4)
        def calc_SumOfFeedin(m, t):
            return m.Feedin[t] == m.PV2Grid[t]

        m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (5)
        def calc_SupplyOfLoads(m, t):
            return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.V2B[t] * m.CarAtHomeStatus[t] == m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        ########## new Constraints Smart Technologies #############################################################
        ############################################################################################################

        def calc_DishWasherHours2(m, t):
            if t == 8759 or t == 8760:
                return m.DishWasher2[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and (DishWasherDuration == 2 or DishWasherDuration == 3):
                return m.DishWasher1[t] == m.DishWasher2[t + 1]
            return m.DishWasher2[t] == 0

        m.calc_DishWasherHours2 = pyo.Constraint(m.t, rule=calc_DishWasherHours2)

        def calc_DishWasherHours3(m, t):
            if t == 8758 or t == 8759 or t == 8760:
                return m.DishWasher2[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and DishWasherDuration == 3:
                return m.DishWasher1[t] == m.DishWasher3[t + 2]
            return m.DishWasher3[t] == 0

        m.calc_DishWasherHours3 = pyo.Constraint(m.t, rule=calc_DishWasherHours3)

        def calc_DishWasherStartTime(m, t):
            if m.t[t] == 1:
                return m.DishWasherStart[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and m.HourOfDay[t] == 7 and DishWasherSmartStatus == 0:
                return m.DishWasher1[t] == 1
            elif m.DishWasherTheoreticalHours[t] == 1 and m.HourOfDay[t] == 24:
                return m.DishWasherStart[t] == m.DishWasherStart[t - 1] - 1 * DishWasherSmartStatus
            return m.DishWasherStart[t] == m.DishWasherStart[t - 1] \
                   + m.DishWasher1[t] * m.DishWasherTheoreticalHours[t] * DishWasherSmartStatus

        m.calc_DishWasherStartTime = pyo.Constraint(m.t, rule=calc_DishWasherStartTime)

        # ------------------------------------------------------------------------------------------------

        def calc_WashingMachineHours2(m, t):
            if t >= 8759:
                return m.WashingMachine2[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1 and (
                    WashingMachineDuration == 2 or WashingMachineDuration == 3):
                return m.WashingMachine1[t] == m.WashingMachine2[t + 1]
            return m.WashingMachine2[t] == 0

        m.calc_WashingMachineHours2 = pyo.Constraint(m.t, rule=calc_WashingMachineHours2)

        def calc_WashingMachineHours3(m, t):
            if t >= 8758:
                return m.WashingMachine2[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1 and WashingMachineDuration == 3:
                return m.WashingMachine1[t] == m.WashingMachine3[t + 2]
            return m.WashingMachine3[t] == 0

        m.calc_WashingMachineHours3 = pyo.Constraint(m.t, rule=calc_WashingMachineHours3)

        def calc_WashingMachineStartTime(m, t):
            if m.t[t] == 1:
                return m.WashingMachineStart[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.HourOfDay[t] == 20 and WashingMachineSmartStatus == 0:
                return m.WashingMachine1[t] == 1
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.HourOfDay[t] == 24:
                return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] - 1 * WashingMachineSmartStatus
            return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] \
                   + m.WashingMachine1[t] * m.WashingMachineTheoreticalHours[t] * WashingMachineSmartStatus

        m.calc_WashingMachineStartTime = pyo.Constraint(m.t, rule=calc_WashingMachineStartTime)

        # -------------------------------------------------------------------------------
        def calc_Dryer1(m, t):
            if t >= 8757:
                return m.Dryer1[t] == 0
            if DryerAdoption == 0:
                return m.Dryer1[t] == 0
            return m.WashingMachine1[t] == m.Dryer1[t+3]

        m.calc_Dryer1 = pyo.Constraint(m.t, rule=calc_Dryer1)

        def calc_Dryer2(m, t):
            if t >= 8756:
                return m.Dryer2[t] == 0
            if DryerAdoption == 0:
                return m.Dryer2[t] == 0
            return m.WashingMachine1[t] == m.Dryer2[t+4]

        m.calc_Dryer2 = pyo.Constraint(m.t, rule=calc_Dryer2)

        #------------------------------------------------------------------

        # (6)
        def calc_SumOfLoads(m, t):
            if m.t[t] == 1:
                return m.Load[t] == m.LoadProfile[t]
            else:
                return m.Load[t] == m.LoadProfile[t] \
                       + (m.DishWasher1[t] + m.DishWasher2[t] + m.DishWasher3[t]) * DishWasherPower \
                       + (m.WashingMachine1[t] + m.WashingMachine2[t] + m.WashingMachine3[t]) * WashingMachinePower \
                       + (m.Dryer1[t] + m.Dryer2[t])*1.5

        m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)

        ############################################################################################################
        ############################################################################################################

        # (7)
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.BatDischarge[t] == 0
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarAtHomeStatus[t]

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
            if m.CarAtHomeStatus[t] == 0:
                return m.EVCharge[t] == 0
            else:
                return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] + \
                       m.Bat2EV[t] * m.CarAtHomeStatus[t]

        m.calc_EVCharge = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (11)
        def calc_EVDischarge(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0
            elif m.CarAtHomeStatus[t] == 0 and m.V2B[t] == 1:
                return m.EVDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.EVDischarge[t] == 0
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.V2B[t] * m.CarAtHomeStatus[t] + m.EV2Bat[t] * m.V2B[t] * \
                       m.CarAtHomeStatus[t]

        m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (12)
        def calc_EVSoC(m, t):
            if m.t[t] == 1:
                return m.EVSoC[t] == 0
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + (m.EVCharge[t] * m.CarAtHomeStatus[t] * 0.95) - (
                        m.EVDischarge[t] * m.CarAtHomeStatus[t] * 1.05)

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
    starttime = 3000
    endtime = 3050

    # battery, grid and PV

    StorageFillLevel = np.array(list(instance.BatSoC.extract_values().values())[starttime: endtime])
    BatDischarge = np.array(list(instance.BatDischarge.extract_values().values())[starttime: endtime])

    BatCharge = np.array(list(instance.BatCharge.extract_values().values())[starttime: endtime])
    Load = np.array(list(instance.Load.extract_values().values())[starttime: endtime])

    DishWasherTheoreticalHours = np.array(list(instance.DishWasherTheoreticalHours.extract_values().values())[starttime: endtime])
    DishWasher1 = np.array(list(instance.DishWasher1.extract_values().values())[starttime: endtime])
    DishWasher2 = np.array(list(instance.DishWasher2.extract_values().values())[starttime: endtime])
    DishWasher3 = np.array(list(instance.DishWasher3.extract_values().values())[starttime: endtime])
    DishWasher = DishWasher1 + DishWasher2 + DishWasher3

    WashingMachineTheoreticalHours = np.array(
        list(instance.WashingMachineTheoreticalHours.extract_values().values())[starttime: endtime])
    WashingMachine1 = np.array(list(instance.WashingMachine1.extract_values().values())[starttime: endtime])
    WashingMachine2 = np.array(list(instance.WashingMachine2.extract_values().values())[starttime: endtime])
    WashingMachine3 = np.array(list(instance.WashingMachine3.extract_values().values())[starttime: endtime])
    WashingMachine = WashingMachine1 + WashingMachine2 + WashingMachine3

    Dryer1 = np.array(list(instance.Dryer1.extract_values().values())[starttime: endtime])
    Dryer2 = np.array(list(instance.Dryer2.extract_values().values())[starttime: endtime])
    Dryer = Dryer1 + Dryer2

    total_cost = instance.OBJ()

    # Plot
    x_achse = np.arange(starttime, endtime)

    plt.figure()
    # plt.plot(x_achse, StorageFillLevel, linewidth=1, label='StorageFillLevel', color='gray')
    # plt.bar(x_achse, BatDischarge, label='GridCover', color='red')
    # plt.bar(x_achse, BatCharge, label='PhotovoltaicFeedin', color='green')
    plt.plot(x_achse, Load, label='Load', color='red')

    plt.bar(x_achse, DishWasherTheoreticalHours, label='DishWasherTheoreticalHours', color='blue', alpha=0.05)
    plt.bar(x_achse, WashingMachineTheoreticalHours, bottom = DishWasherTheoreticalHours, label='WashingMachineTheoreticalHours', color='orange', alpha=0.10)


    plt.bar(x_achse, DishWasher, label='DishWasher', color='blue', alpha=0.4)
    plt.bar(x_achse, WashingMachine, bottom = DishWasher, label='WashingMachine', color='orange', alpha=0.6)
    plt.bar(x_achse, Dryer, bottom = DishWasher + WashingMachine, label='Dryer', color='green', alpha=0.6)



    plt.legend()
    plt.xlabel('Hour of year')
    plt.ylabel('Power in kW')
    plt.title('Costs: ' + str(round(total_cost, 2)) + ' €')
    plt.savefig('DishWasher.png', dpi=350)
    plt.show()


if __name__ == "__main__":
    A = OperationOptimization(DB().create_Connection(CONS().RootDB))
    instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile = A.run()
    show_results(instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater,
                 PhotovoltaicProfile)
