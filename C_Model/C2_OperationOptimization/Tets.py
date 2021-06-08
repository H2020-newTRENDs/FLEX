import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.util.infeasible import log_infeasible_constraints

import matplotlib.dates as mdates
import datetime


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

        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature_test, self.Conn)
        self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FiT = DB().read_DataFrame(REG().Sce_Price_HourlyFeedinTariff, self.Conn)
        self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG().Sce_Demand_HotWaterProfile, self.Conn)
        self.CarAtHome = DB().read_DataFrame(REG().Gen_Sce_CarAtHomeHours, self.Conn)
        self.Demand_EV = DB().read_DataFrame(REG().Sce_Demand_ElectricVehicleBehavior, self.Conn)

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
        print('TankSize: ')
        print(Household.SpaceHeating.TankSize)
        ID_Household = Household.ID
        print('Household ID: ')
        print(ID_Household)

        Environment = self.gen_Environment(environment_id)
        ID_Environment = Environment.ID
        ID_ElectricityPriceType = Environment.ID_ElectricityPriceType
        ID_FiT = Environment.ID_FeedinTariffType

        # used for the data import in pyomo
        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary

        # the data of the simulation are indexed for 1 year, start and stoptime in visualization
        HoursOfSimulation = 8760

        #########################################################################################
        # EV

        # BatteryCapacity = 0
        if Household.ElectricVehicle.BatterySize == 0:
            CarAtHome = create_dict([0] * HoursOfSimulation)
            V2B = create_dict([0] * HoursOfSimulation)      #V2B = Vehicle to building

        # BatteryCapacity =/ 0
        else:
            CarAtHome = create_dict(self.CarAtHome.CarAtHomeHours.to_numpy())

            # Swichting the Vehicle to Grid 1 or 0
            V2B = create_dict([1] * HoursOfSimulation)

        # Set car demand to zero, for testing saving
        CarDemand = 1

        print('The EV Batterysize is: ' + str(Household.ElectricVehicle.BatterySize))
        print('The Batterysize is : ' + str(Household.Battery.Capacity))
        print('The PV Size is: ' + str(Household.PV.PVPower))

        # Calculation of Hourly EV Demand
        EV_DailyDemand = int(self.Demand_EV.EVDailyElectricityConsumption)
        EV_LeaveHour = int(self.Demand_EV.EVLeaveHomeClock)
        EV_ArriveHour = int(self.Demand_EV.EVArriveHomeClock)
        EV_AwayHours = EV_ArriveHour - EV_LeaveHour
        EV_HourlyDemand = EV_DailyDemand / EV_AwayHours

        #############################################################################################

        # PV, battery and electricity load profile
        LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile.to_numpy()
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH

        SumOfPV = round(sum(PhotovoltaicProfileHH), 2)
        SumOfLoad = round(sum(self.LoadProfile.BaseElectricityProfile), 2)

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

        # Calculation of solar gains
        path2excel = CONS().DatabasePath
        # window areas in celestial directions
        Awindows_rad_east_west = Household.Building.average_effective_area_wind_west_east_red_cool
        Awindows_rad_south = Household.Building.average_effective_area_wind_south_red_cool
        Awindows_rad_north = Household.Building.average_effective_area_wind_north_red_cool
        # solar gains from different celestial directions
        Q_sol_all = pd.read_csv(CONS().DatabasePath + "directRadiation_himmelsrichtung_GER.csv", sep=";")
        Q_sol_north = np.outer(Q_sol_all.loc[:, "RadiationNorth"].to_numpy(), Awindows_rad_north)
        Q_sol_south = np.outer(Q_sol_all.loc[:, "RadiationSouth"].to_numpy(), Awindows_rad_south)
        Q_sol_east_west = np.outer((Q_sol_all.loc[:, "RadiationEast"].to_numpy() +
                                    Q_sol_all.loc[:, "RadiationWest"].to_numpy()), Awindows_rad_east_west)
        Q_sol = (Q_sol_north + Q_sol_south + Q_sol_east_west).squeeze()
        # Solar Gains, but to small calculated: in W/m²
        # Q_sol = self.Radiation.loc[self.Radiation["ID_Country"] == 5].loc[:, "Radiation"].to_numpy()

        # (1) Scenario Case: select ElectricityPriceType
        if ID_ElectricityPriceType == 1:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 1].loc[:,
                               'HourlyElectricityPrice'].to_numpy()
            ElectricityPrice = list(ElectricityPrice)
            ElectricityPrice = ElectricityPrice * HoursOfSimulation

        elif ID_ElectricityPriceType == 2:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 2].loc[:,
                               'HourlyElectricityPrice'].to_numpy()

        # (2) Scenario Case: select FiT
        # until now only 1 FiT Type in DB
        if ID_FiT == 1:
            FiT = self.FiT.loc[self.FiT['ID_FeedinTariffType'] == 1].loc[:,
                  'HourlyFeedinTariff'].to_numpy()
            FiT = list(FiT)
            FiT = FiT * HoursOfSimulation
        else:
            FiT = self.FiT.loc[self.FiT['ID_FeedinTariffType'] == 1].loc[:,
                  'HourlyFeedinTariff'].to_numpy()
            FiT = list(FiT)
            FiT = FiT * HoursOfSimulation

        # (3) Select COP Air or COP Water for Space Heating
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

        # (4) Select COP Air or COP Water for HotWater

        if Household.SpaceHeating.ID_SpaceHeatingBoilerType == 1:
            Tout = self.Weather.Temperature
            COP = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 1].loc[:,
                  'COP_HotWater'].to_numpy()
            COPtemp = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 1].loc[:,
                      'TemperatureEnvironment'].to_numpy()
            COP_HotWater = []
            j = 0

            for i in range(0, len(list(Tout))):
                for j in range(0, len(list(COPtemp))):
                    if COPtemp[j] < Tout[i]:
                        continue
                    else:
                        COP_HotWater.append(COP[j])
                    break
        elif Household.SpaceHeating.ID_SpaceHeatingBoilerType == 2:
            COP = self.HeatPumpCOP.loc[self.HeatPumpCOP['ID_SpaceHeatingBoilerType'] == 2].loc[:,
                  'COP_HotWater'].to_numpy()
            COP = list(COP)
            COP = COP * HoursOfSimulation
            COP_HotWater = COP

        # Hot Water Demand with Part 1 (COP SpaceHeating) and Part 2 (COP HotWater, lower)
        HotWaterProfile1 = self.HotWaterProfile.loc[self.HotWaterProfile['ID_HotWaterProfileType'] == 1].loc[:,
                           'HotWaterPart1'].to_numpy()
        HotWaterProfile2 = self.HotWaterProfile.loc[self.HotWaterProfile['ID_HotWaterProfileType'] == 1].loc[:,
                           'HotWaterPart2'].to_numpy()

        # model for optimisation
        m = pyo.AbstractModel()

        # parameters
        m.t = pyo.RangeSet(1, HoursOfSimulation)
        # price
        m.ElectricityPrice = pyo.Param(m.t, initialize=create_dict(ElectricityPrice))
        # Feed in Tariff of Photovoltaic
        m.FiT = pyo.Param(m.t, initialize=create_dict(FiT))

        # solar gains:
        m.Q_Solar = pyo.Param(m.t, initialize=create_dict(Q_sol))
        # outside temperature
        m.T_outside = pyo.Param(m.t, initialize=create_dict(self.Weather.Temperature.to_numpy()))
        # COP of heatpump
        m.COP_dynamic = pyo.Param(m.t, initialize=create_dict(ListOfDynamicCOP))
        # electricity load profile
        m.LoadProfile = pyo.Param(m.t, initialize=create_dict(LoadProfile))
        # PV profile
        m.PhotovoltaicProfile = pyo.Param(m.t, initialize=create_dict(PhotovoltaicProfile))
        # HotWater
        m.HWPart1 = pyo.Param(m.t, initialize=create_dict(HotWaterProfile1))
        m.HWPart2 = pyo.Param(m.t, initialize=create_dict(HotWaterProfile2))
        m.COP_HotWater = pyo.Param(m.t, initialize=create_dict(COP_HotWater))

        # Vehicle to Grid Status
        m.V2B = pyo.Param(m.t, initialize=V2B)
        m.CarStatus = pyo.Param(m.t, initialize=CarAtHome)

        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))  # max Power of Boiler; DB?
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * M_WaterTank * (273.15 + T_TankMin),
                                                                     CWater * M_WaterTank * (273.15 + T_TankMax)))
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))

        # energy used for cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 6_000))  # 6kW thermal, 2 kW electrical
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 24))  # Change to TargetTemp
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 60))

        # EV
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21)) #380 * 32 * 1,72
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.Capacity))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxChargePower))
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        # m.Bat2Grid...

        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals,
                                bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals,
                             bounds=(0, Household.ElectricVehicle.BatteryMaxChargePower))
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals,
                            bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals,
                           bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatterySize))

        # m. EVtoGrid...

        # objective

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        # (1) Energy balance of supply == demand
        def calc_ElectricalEnergyBalance(m, t):
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] == m.Feedin[t] + \
                       m.Load[t] + m.BatCharge[t]
            elif Household.Battery.Capacity == 0:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.EVDischarge[t] == m.Feedin[t] + \
                       m.Load[t] + m.EVCharge[t]
            else:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] + m.EVDischarge[t] == m.Feedin[t] + \
                       m.Load[t] + m.BatCharge[t] + m.EVCharge[t]

        m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

        # (2)
        def calc_UseOfGrid(m, t):
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid[t] == m.Grid2Load[t]
            else:
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
            if Household.ElectricVehicle.BatterySize == 0 and Household.ElectricVehicle.BatterySize == 0:
                return m.Grid2Load[t] + m.PV2Load[t] == m.Load[t]
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]
            elif Household.Battery.Capacity == 0:
                return m.Grid2Load[t] + m.PV2Load[t] + m.EV2Load[t] * m.V2B[t] * m.CarStatus[t] == m.Load[t]
            else:
                return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.V2B[t] * m.CarStatus[t] == \
                       m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # (6)
        def calc_SumOfLoads(m, t):
            return m.Load[t] == m.LoadProfile[t] \
                   + ((m.Q_TankHeating[t] / m.COP_dynamic[t]) / 1_000) \
                   + (m.Q_RoomCooling[t] / 3 / 1_000) \
                   + (m.HWPart1[t] / m.COP_dynamic[t]) \
                   + (m.HWPart2[t] / m.COP_HotWater[t])

        m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)

        # (7)
        def calc_BatDischarge(m, t):
            if Household.Battery.Capacity == 0:
                return m.BatDischarge[t] == 0
            elif m.t[t] == 1:
                return m.BatDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.BatDischarge[t] == 0
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarStatus[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8)
        def calc_BatCharge(m, t):
            if Household.Battery.Capacity == 0:
                return m.BatCharge[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.BatCharge[t] == m.PV2Bat[t]
            else:
                return m.BatCharge[t] == m.PV2Bat[t] + m.EV2Bat[t] * m.V2B[t]

        m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (9)
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.BatSoC[t] == 0
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * Household.Battery.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - Household.Battery.DischargeEfficiency))

        m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)

        # (10)
        def calc_EVCharge(m, t):
            if Household.ElectricVehicle.BatterySize == 0:
                return m.EVCharge[t] == 0
            elif m.CarStatus[t] == 0:
                return m.EVCharge[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.EVCharge[t] * m.CarStatus[t] == m.Grid2EV[t] * m.CarStatus[t] + m.PV2EV[t] * m.CarStatus[t]
            else:
                return m.EVCharge[t] * m.CarStatus[t] == m.Grid2EV[t] * m.CarStatus[t] + m.PV2EV[t] * m.CarStatus[t] + \
                       m.Bat2EV[t] * m.CarStatus[t]

        m.calc_EVCharge = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (11)
        def calc_EVDischarge(m, t):
            if t == 1:
                return m.EVDischarge[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.EVDischarge[t] == 0
            elif m.CarStatus[t] == 0 and m.V2B[t] == 1:
                return m.EVDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.EVDischarge[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.EVDischarge[t] == m.EV2Load[t] * m.V2B[t] * m.CarStatus[t]
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.V2B[t] * m.CarStatus[t] + m.EV2Bat[t] * m.V2B[t] * \
                       m.CarStatus[t]

        m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (12)
        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.EVSoC[t] == 0
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + (
                            m.EVCharge[t] * m.CarStatus[t] * Household.ElectricVehicle.BatteryChargeEfficiency) \
                       - (m.EVDischarge[t] * m.CarStatus[t] * (
                            1 + (1 - Household.ElectricVehicle.BatteryDischargeEfficiency))) \
                       - (EV_HourlyDemand * (1 - m.CarStatus[t]))

        m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)

        ################################  Tank and Building ##########################################

        # energy input and output of tank energy
        def tank_energy(m, t):
            if t == 1:
                return m.E_tank[t] == CWater * M_WaterTank * (273.15 + T_TankStart)
            else:
                return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] \
                       - U_ValueTank * A_SurfaceTank * ((m.E_tank[t] / (M_WaterTank * CWater)) - T_TankSourounding)

        m.tank_energy_rule = pyo.Constraint(m.t, rule=tank_energy)

        # 5R 1C model:
        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                return m.Tm_t[t] == 15

            else:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_Solar[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) /
                                                                    Hve) + T_sup)) / Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                T_air = 20
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                        T_sup + (PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / Hve)) / \
                      (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (Htr_is + Hve)
                # T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t]) / (Htr_is + Hve)
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        instance = m.create_instance(report_timing=True)
        # opt = pyo.SolverFactory("glpk")
        opt = pyo.SolverFactory("gurobi")
        results = opt.solve(instance, tee=True)
        instance.display("./log.txt")
        print(results)
        # return relevant data
        return instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater

    def run(self):
        for household_id in range(37000, 37001):
            for environment_id in range(3, 4):
                instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater = self.run_Optimization(
                    household_id,
                    environment_id)
                return instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater


# create plots to visualize results price
def show_results(instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, colors):
    starttime = 1000
    endtime = 1100

    # exogenous profiles
    ElectricityPrice = np.array(list(instance.ElectricityPrice.extract_values().values())[starttime: endtime])
    LoadProfile = np.array(list(instance.LoadProfile.extract_values().values())[starttime: endtime])

    # tank and building
    Q_TankHeating = np.array(list(instance.Q_TankHeating.extract_values().values())[starttime: endtime]) / 1000
    Q_Solar = np.array(list(instance.Q_Solar.extract_values().values())[starttime: endtime]) / 1_000  # from W to kW
    Q_RoomHeating = np.array(
        list(instance.Q_RoomHeating.extract_values().values())[starttime: endtime]) / 1000  # from W to kW
    Q_RoomCooling = np.array(
        list(instance.Q_RoomCooling.extract_values().values())[starttime: endtime]) / 1000  # from W to kW
    T_room = np.array(list(instance.T_room.extract_values().values())[starttime: endtime])
    E_tank = np.array(list(instance.E_tank.extract_values().values())[starttime: endtime])
    T_tank = ((pd.Series(E_tank) / (M_WaterTank * CWater)) - 273.15)

    # grid
    Grid = np.array(list(instance.Grid.extract_values().values())[starttime: endtime])
    Grid2EV = np.nan_to_num(
        np.array(np.array(list(instance.Grid2EV.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    Grid2Load = np.array(list(instance.Grid2Load.extract_values().values())[starttime: endtime])

    # battery
    BatSoC = np.nan_to_num(
        np.array(np.array(list(instance.BatSoC.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    BatDischarge = np.nan_to_num(
        np.array(np.array(list(instance.BatDischarge.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)
    BatCharge = np.nan_to_num(
        np.array(np.array(list(instance.BatCharge.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)
    Bat2EV = np.nan_to_num(
        np.array(np.array(list(instance.Bat2EV.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    Bat2Load = np.nan_to_num(
        np.array(np.array(list(instance.Bat2Load.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)

    # Feed in
    Feedin = np.array(list(instance.Feedin.extract_values().values())[starttime: endtime])

    # Loads
    Load = np.array(list(instance.Load.extract_values().values())[starttime: endtime])
    HotWater1 = np.array(list(instance.HWPart1.extract_values().values())[starttime: endtime]) / \
                np.array(list(instance.COP_dynamic.extract_values().values())[starttime: endtime])
    ElectricityDemandHeatPump = np.array(
        list(instance.Q_TankHeating.extract_values().values())[starttime: endtime]) / 1000 / \
                                np.array(list(instance.COP_dynamic.extract_values().values())[starttime: endtime])
    ElectricityCooling = np.array(
        list(instance.Q_RoomCooling.extract_values().values())[starttime: endtime]) / 1000 / 3

    HotWater2 = np.array(list(instance.HWPart2.extract_values().values())[starttime: endtime]) / \
                np.array(list(instance.COP_HotWater.extract_values().values())[starttime: endtime])
    HotWater = HotWater1 + HotWater2

    # EV
    EVSoC = np.nan_to_num(
        np.array(np.array(list(instance.EVSoC.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)
    EVCharge = np.nan_to_num(
        np.array(np.array(list(instance.EVCharge.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)
    EVDischarge = np.nan_to_num(
        np.array(np.array(list(instance.EVDischarge.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)

    EV2Load = np.nan_to_num(
        np.array(np.array(list(instance.EV2Load.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    EV2Bat = np.nan_to_num(
        np.array(np.array(list(instance.EV2Bat.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    CarStatus = np.nan_to_num(
        np.array(np.array(list(instance.CarStatus.extract_values().values())[starttime: endtime]), dtype=np.float),
        nan=0)

    # PV
    PhotovoltaicProfile = np.array(list(instance.PhotovoltaicProfile.extract_values().values())[starttime: endtime])
    PV2Load = np.array(list(instance.PV2Load.extract_values().values())[starttime: endtime])
    PV2Bat = np.nan_to_num(
        np.array(np.array(list(instance.PV2Bat.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)
    PV2Grid = np.array(list(instance.PV2Grid.extract_values().values())[starttime: endtime])
    PV2EV = np.nan_to_num(
        np.array(np.array(list(instance.PV2EV.extract_values().values())[starttime: endtime]), dtype=np.float), nan=0)


    x_achse = np.arange(starttime, endtime)

    timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
                              tz=datetime.timezone.utc)





    #  (1) EV ####################################################################################################
    fig, ax1 = plt.subplots()
    ax1.bar(x_achse, CarStatus, label='CarStatus', color='grey', alpha=0.3)
    ax1.bar(x_achse, EVCharge, label='EVCharge', color='green', alpha=0.3)
    ax1.bar(x_achse, EVDischarge, label='EVDischarge', color='red', alpha=0.3)
    ax1.set_ylabel("Power kW")
    ax2 = ax1.twinx()

    ax2.plot(x_achse, EVSoC, linewidth=2, label='EVSoC', color='blue', alpha=0.2)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_ylabel("SoC in kWh")

    plt.title('(1) EV charge and discharge')
    plt.tight_layout()
    fig.savefig('(1) charge and discharge', dpi=200)
    plt.show()

    #  (2) EV and PV #############################################################################################
    fig, ax1 = plt.subplots()

    ax1.plot(x_achse, PhotovoltaicProfile, linewidth=1.5, label='PhotovoltaicProfile',
             color=colors["PhotovoltaicProfile"])
    ax1.set_ylabel("Power kW")

    ax2 = ax1.twinx()
    ax2.bar(x_achse, CarStatus, linewidth=2.5, label='CarStatus', color='grey', alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_ylabel("Car at home = 1, away = 0")

    plt.title('(2) EV and PV')
    plt.tight_layout()
    fig.savefig('(2) EV an PV', dpi=200)
    plt.show()

    #  (3) Discharge of EV #####################################################################################
    fig, ax1 = plt.subplots()

    ax1.plot(x_achse, EVDischarge, linewidth=0.5, label='EVDischarge', color='red')
    ax1.bar(x_achse, EV2Load, label='EV2Load', color='orange', alpha=0.5)
    ax1.bar(x_achse, EV2Bat, bottom=EV2Load, label='EV2Bat', color='green', alpha=0.5)

    ax1.plot(x_achse, EVCharge, linewidth=0.5, label='EVCharge', color='green')
    ax1.bar(x_achse, PV2EV, label='PV2EV', color='blue', alpha=0.5)
    ax1.bar(x_achse, Bat2EV, bottom=PV2EV, label='Bat2EV', color='pink', alpha=0.5)
    ax1.bar(x_achse, Grid2EV, bottom=PV2EV + Bat2EV, label='Grid2EV', color='grey', alpha=0.5)

    ax1.set_ylabel("Power in kW")

    ax2 = ax1.twinx()
    ax2.bar(x_achse, CarStatus, linewidth=2.5, label='CarStatus', color='grey', alpha=0.1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_ylabel("Car at home = 1, away = 0")

    plt.title('(3) Parts of Charge and Discharge of EV')
    plt.tight_layout()
    fig.savefig('(3) Parts of Charge and Discharge of EV', dpi=200)
    plt.show()

    #  (4) Grid and Price ########################################################################################
    fig, ax1 = plt.subplots()

    ax1.bar(x_achse, Grid, label='Grid', color='red', alpha=0.3)
    ax1.plot(x_achse, PhotovoltaicProfile, linewidth=1, label='PV', color='orange', alpha=0.6)

    ax1.set_ylabel("Power kW")

    ax2 = ax1.twinx()
    ax2.plot(x_achse, ElectricityPrice, linewidth=0.5, label='ElectricityPrice', color='black', linestyle='dotted')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_ylabel("Price in Ct/€")

    plt.title('(4) Grid and Price')
    plt.tight_layout()
    fig.savefig('(4) Grid and Price', dpi=200)
    plt.show()

    #  (5) Stationary Battery #######################################################################################
    fig, ax1 = plt.subplots()
    ax1.bar(x_achse, BatCharge, label='BatCharge', color='green', alpha=0.6)
    ax1.bar(x_achse, BatDischarge, label='BatDischarge', color='red', alpha=0.6)
    ax1.bar(x_achse, PhotovoltaicProfile, label='PV', color='orange', alpha=0.2)

    ax1.set_ylabel("Power in kW")

    ax2 = ax1.twinx()
    ax2.plot(x_achse, BatSoC, linewidth=1, label='BatSoC', color='black', linestyle='dotted')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.set_ylabel("SoC in kWh")

    plt.title('(5) Stationary Battery')
    plt.tight_layout()
    fig.savefig('(5) Stationary Battery', dpi=200)
    plt.show()

    #  (6) Use of PV Power ###########################################################################################
    fig, ax1 = plt.subplots()
    ax1.plot(x_achse, PhotovoltaicProfile, linewidth=1, label='PV', color='orange', alpha=0.8)
    ax1.plot(x_achse, Load, linewidth=1, label='Load', color='black', alpha=0.8)

    ax1.bar(x_achse, PV2Load, label='PV2Load', color='green', alpha=0.5)
    ax1.bar(x_achse, PV2EV, bottom=PV2Load, label='PV2EV', color='grey', alpha=0.5)
    ax1.bar(x_achse, PV2Bat, bottom=PV2Load + PV2EV, label='PV2Bat', color='blue', alpha=0.5)
    ax1.bar(x_achse, PV2Grid, bottom=PV2Load + PV2EV + PV2Bat, label='PV2Grid', color='red', alpha=0.5)

    ax1.bar(x_achse, Bat2Load, bottom=PV2Load, label='Bat2Load', color='orange', alpha=0.5)
    ax1.bar(x_achse, Grid2Load, bottom=PV2Load + Bat2Load, label='Grid2Load', color='pink', alpha=0.5)
    ax1.bar(x_achse, EV2Load, bottom=PV2Load + Bat2Load + Grid2Load, label='EV2Load', color='brown', alpha=0.5)

    # timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
    #                           tz=datetime.timezone.utc)
    #
    # y = np.arange(100)
    # plt.plot(timearray[1000:1100], y)
    # ax1 = plt.gca()
    # ax1.xaxis.set(
    #     major_locator=mdates.DayLocator(),
    #     major_formatter=mdates.DateFormatter("\n\n%b-%d"),
    #     minor_locator=mdates.HourLocator((0, 12)),
    #     minor_formatter=mdates.DateFormatter("%H"),)
    #
    # plt.tight_layout()
    # plt.show()

    ax1.set_ylabel("Power in kW")
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    plt.title('(6) Use of PV Power')
    plt.tight_layout()
    fig.savefig('(6) Use of PV Power', dpi=200)
    plt.show()

    #  (7) Supply and demand of loads #############################################################################
    fig, ax1 = plt.subplots()
    ax1.plot(x_achse, Load, linewidth=0.5, label='Load', color='black', alpha=0.5)

    ax1.bar(x_achse, LoadProfile, label='LoadProfile', color='grey', alpha=0.5)
    ax1.bar(x_achse, HotWater, bottom=LoadProfile, label='HotWater', color='blue', alpha=0.5)
    ax1.bar(x_achse, ElectricityDemandHeatPump, bottom=LoadProfile + HotWater, label='HP', color='red', alpha=0.5)
    ax1.bar(x_achse, ElectricityCooling, bottom=LoadProfile + HotWater + ElectricityDemandHeatPump, label='Climate',
            color='turquoise', alpha=0.5)

    ax1.bar(x_achse, -PV2Load, label='PV2Load', color='green', alpha=0.5)
    ax1.bar(x_achse, -Bat2Load, bottom=-PV2Load, label='Bat2Load', color='orange', alpha=0.5)
    ax1.bar(x_achse, -Grid2Load, bottom=-PV2Load + -Bat2Load, label='Grid2Load', color='black', alpha=0.5)
    ax1.bar(x_achse, -EV2Load, bottom=-PV2Load + -Bat2Load + -Grid2Load, label='EV2Load', color='grey', alpha=0.5)

    ax1.set_ylabel("Power in kW")
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    ax1.grid(True, which='both')
    plt.title('(7) Supply and demand of loads, no EV')
    plt.tight_layout()
    fig.savefig('(7) Supply and demand of loads', dpi=200)
    plt.show()

    #  (8) Supply and demand of loads and EV ########################################################################
    fig, ax1 = plt.subplots()

    ax1.bar(x_achse, Load, label='Load', color='red', alpha=0.5)
    ax1.bar(x_achse, EVCharge, bottom=Load, label='EVCharge', color='green', alpha=0.5)

    ax1.bar(x_achse, -Grid2Load, label='Grid', color='grey', alpha=0.5)
    ax1.bar(x_achse, -Grid2EV, bottom=+ -Grid2Load, color='grey', alpha=0.5)

    ax1.bar(x_achse, -PV2Load, bottom=+-Grid2Load + -Grid2EV, label='PV', color='orange', alpha=0.5)
    ax1.bar(x_achse, -PV2EV, bottom=+-Grid2Load + -Grid2EV + -PV2Load, color='orange', alpha=0.5)

    ax1.bar(x_achse, -Bat2Load, bottom=+-Grid2Load + -Grid2EV + -PV2Load, label='Bat', color='blue', alpha=0.5)
    ax1.bar(x_achse, -Bat2EV, bottom=+-Grid2Load + -Grid2EV + -PV2Load + -Bat2Load, color='blue', alpha=0.5)

    ax1.set_ylabel("Power in kW")
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    ax1.grid(True, which='both')
    plt.title('(8) Supply and demand of loads and EV')
    plt.tight_layout()
    fig.savefig('(8) Supply and demand of loads and EV', dpi=200)
    plt.show()

    print('For this the starttime = 1 and stoptime = 8760!!! - otherwise partly wrong results')
    # yearly values
    total_cost = instance.OBJ()
    print('Yearly electricity cost of all technologies: ' + str(total_cost) + '  €')

    YearlyHeatGeneration = np.sum(Q_TankHeating)
    print('Yearly Output of Heat pump (y): ' + str(YearlyHeatGeneration) + ' kWh')

    JAZ = round(np.sum(Q_TankHeating) / np.sum(ElectricityDemandHeatPump), 2)
    print('Annual performance factor of heatpump: ' + str(JAZ))

    YearlyHeatDemand = np.sum(Q_RoomHeating)
    print('Yearly demand of Space Heating (y): ' + str(YearlyHeatDemand) + ' kWh')

    YearlyCoolingDemand = np.sum(Q_RoomCooling)
    print('Yearly demand of Space Cooling (y): ' + str(YearlyCoolingDemand) + ' kWh')

    YearlySolarGain = np.sum(Q_Solar)
    print('Yearly demand of Solar gains (y): ' + str(YearlySolarGain) + ' kWh')

    ### Grid PV ###
    YearlyGrid = np.sum(Grid)
    print('Yearly cover from electricity grid: ' + str(YearlyGrid) + ' kWh')

    YearlyPVGeneration = np.sum(PhotovoltaicProfile)
    print('Yearly generation of Photvoltaic power: (y) ' + str(YearlyPVGeneration) + ' kWh')

    YearlyFeedin = np.sum(Feedin)
    print('Yearly photovoltaic feed in  : ' + str(YearlyFeedin) + ' kWh')

    ### Loads ###
    YearlyElectricityDemand = np.sum(Load)
    print('Yearly demand of electricity (all) (y) : ' + str(YearlyElectricityDemand) + ' kWh')

    YearlyElectricityDemandHeatPump = np.sum(ElectricityDemandHeatPump)
    print('Yearly electricity demand of heatpump (y): ' + str(YearlyElectricityDemandHeatPump) + ' kWh')

    YearlyElectricityCooling = np.sum(ElectricityCooling)
    print('Yearly electricity demand cooling (y): ' + str(YearlyElectricityCooling) + ' kWh')

    YearlyBaseLoad = np.sum(LoadProfile)
    print('Yearly base electricity demand (y) : ' + str(YearlyBaseLoad) + ' kWh')

    YearlyHW1 = np.sum(HotWater1)
    print('Yearly hot water part 1 electricity demand (y) : ' + str(YearlyHW1) + ' kWh')

    YearlyHW2 = np.sum(HotWater2)
    print('Yearly hot water part 2 electricity demand (y) : ' + str(YearlyHW2) + ' kWh')

    YearlyHW = np.sum(HotWater)
    print('Yearly hot water 1+2 electricity demand (y) : ' + str(YearlyHW) + ' kWh')

    ### EV and battery
    YearlyStoredEnergy = np.sum(BatCharge)
    print('Yearly charged power in battery : ' + str(YearlyStoredEnergy) + ' kWh')

    YearlyDischargedEnergy = np.sum(BatDischarge)
    print('Yearly discharged power in battery : ' + str(YearlyDischargedEnergy) + ' kWh')

    YearlyLossesOfBattery = YearlyStoredEnergy - YearlyDischargedEnergy
    print('Yearly losses of Battery : ' + str(YearlyLossesOfBattery) + ' kWh')

    YearlyChargeOfEV = np.sum(EVCharge)
    print('Yearly charge of EV : ' + str(YearlyChargeOfEV) + ' kWh')

    YearlyDishargeOfEV = np.sum(EVDischarge)
    print('Yearly discharge of EV : ' + str(YearlyDishargeOfEV) + ' kWh')

    YearlyEVDemand = 10 * 365
    print('Yearly driving demand of EV : ' + str(YearlyEVDemand) + ' kWh')

    YearlyLossesOfEV = YearlyChargeOfEV - YearlyDishargeOfEV - YearlyEVDemand
    print('Yearly losses and drive demand of EV: ' + str(YearlyLossesOfEV) + ' kWh')

    UseOfPV = YearlyPVGeneration - YearlyFeedin

    SelfConsumption = round(UseOfPV / YearlyPVGeneration, 2)
    SelfSufficiency = round(UseOfPV / (YearlyElectricityDemand + YearlyEVDemand), 2)

    print('Self-consumption rate: ' + str(SelfConsumption))
    print('Self-sufficiency rate: ' + str(SelfSufficiency))

    ### old plots ########################################################################################

    x_achse = np.arange(starttime, endtime)

    # # Plot (1) Room and building
    # fig, (ax1, ax3) = plt.subplots(2, 1)
    # ax2 = ax1.twinx()
    # ax4 = ax3.twinx()
    # ax1.plot(x_achse, Q_TankHeating, label="Q_TankHeating", color=colors["Q_TankHeating"], linewidth=0.25, alpha=0.6)
    # ax1.plot(x_achse, Q_RoomHeating, label="Q_RoomHeating", color=colors["Q_RoomHeating"], linewidth=0.25, alpha=0.6)
    # ax1.plot(x_achse, Q_RoomCooling, label="Q_RoomCooling", color=colors["Q_RoomCooling"], linewidth=0.25, alpha=0.6)
    # ax1.plot(x_achse, Q_Solar, label="Q_Solar", color=colors["Q_Solar"], linewidth=0.5, alpha=0.6)
    # ax2.plot(x_achse, ElectricityPrice, color=colors["ElectricityPrice"], label="Price", linewidth=0.50, linestyle=':',
    #          alpha=0.6)
    #
    # ax1.set_ylabel("Energy kW")
    # ax2.set_ylabel("Price per kWh in Ct/€")
    #
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2, loc=0)
    #
    # ax3.plot(x_achse, T_room, label="TempRoom", color=colors["T_room"], linewidth=0.15, alpha=0.8)
    # ax4.plot(x_achse, T_tank, label="TempTank", color=colors["T_tank"], linewidth=0.15, alpha=0.8)
    #
    # ax3.set_ylabel("Room temperature in °C", color='black')
    # ax4.set_ylabel("Tank Temperature in °C", color='black')
    #
    # lines, labels = ax3.get_legend_handles_labels()
    # lines2, labels2 = ax4.get_legend_handles_labels()
    # ax3.legend(lines + lines2, labels + labels2, loc=0)
    #
    # plt.grid()
    # ax1.set_title(
    #     '(1) + (2) Yearly heat generation: ' + str(round(YearlyHeatGeneration / 1000, 2)) + ' kWh ' + 'APF: ' + \
    #     str(JAZ))
    # plt.tight_layout()
    # fig.savefig('Room and building', dpi=300)
    # plt.show()


if __name__ == "__main__":
    # colorcode
    red = '#F47070'
    blue = '#8EA9DB'

    green = '#088A29'
    orange = '#F4B084'
    yellow = '#FFBF00'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#0404B4'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    dark_red = '#c70d0d'
    dark_grey = '#2c2e2e'
    light_brown = '#db8b55'
    black = "#000000"
    red_pink = "#f75d82"
    brown = '#A52A2A'
    colors = {"Q_RoomHeating": dark_red,
              "Q_TankHeating": red,
              "Q_RoomCooling": dark_blue,
              "T_room": orange,
              "T_room_noDR": pink,
              "T_tank": brown,
              "ElectricityPrice": dark_grey,
              "Q_Solar": orange,
              "PhotovoltaicProfile": yellow,
              "PhotovoltaicFeedin": green,
              "PhotovoltaicDirect": dark_grey,
              "Q_RoomHeating_noDR": red_pink,
              "Q_RoomCooling_noDR": turquoise,
              "GridCover": light_brown,
              'BatCharge': black,
              'BatDischarge': red,
              'SumOfLoads': black,
              "StateOfCharge": black}
    A = OperationOptimization(DB().create_Connection(CONS().RootDB))
    instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater = A.run()
    show_results(instance, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, colors)
