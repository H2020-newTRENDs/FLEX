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

        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature_test, self.Conn)
        self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG().Sce_Demand_HotWaterProfile, self.Conn)
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

        # the data of the simulation are indexed for 1 year, start and stoptime in visualization
        HoursOfSimulation = 8760

        # used for the data import in pyomo
        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary

        # PV, battery and electricity load profile
        LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile.to_numpy()
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH

        SumOfPV = round(sum(PhotovoltaicProfileHH), 2)
        print('Sum of yearly PV; ')
        print(SumOfPV)
        SumOfLoad = round(sum(self.LoadProfile.BaseElectricityProfile), 2)
        print('Sum of yearly ElectricityBaseLoad: ')
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

        # (2) Select COP Air or COP Water for Space Heating
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

        # (2) Select COP Air or COP Water for HotWater

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

        print(COP_HotWater)

        # Hot Water Part 1
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
        # m.FeedinTariff = pyo.Param(m.t, initialize=create_dict(FeedinTariff)) # only 1 constant value

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

        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))  # max Power of Boiler; DB?
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * M_WaterTank * (273.15 + T_TankMin),
                                                                     CWater * M_WaterTank * (273.15 + T_TankMax)))
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))

        # energy used for cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 6_000))  # 6kW thermal, 2 kW electrical
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 24))  # Change to TargetTemp
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 60))

        # Variables PV and battery
        m.StateOfCharge = pyo.Var(m.t, initialize=1, within=pyo.NonNegativeReals,
                                  bounds=(0, Household.Battery.Capacity))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxChargePower))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.GridCover = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10.5))  # 380 V * 16 A * 1,73 = 10,5 kW
        m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10.5))
        m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10.5))
        m.SumOfLoads = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # objective

        # m.Q_TankHeating[t] / m.COP_dynamic[t] * m.p[t]
        def minimize_cost(m):
            rule = sum(m.GridCover[t] * m.ElectricityPrice[t] - m.PhotovoltaicFeedin[t] * 0.0792 for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        # Battery
        def calc_StateOfCharge(m, t):
            if t == 1:
                return m.StateOfCharge[t] == 1
            else:
                return m.StateOfCharge[t] == m.StateOfCharge[t - 1] \
                       + m.BatCharge[t - 1] * Household.Battery.ChargeEfficiency \
                       - m.BatDischarge[t - 1] * (1 + (1 - Household.Battery.DischargeEfficiency)) \
                       - (Household.Battery.Capacity * 0.02 / 30 / 24)  # self discharge
                # SelfDischarge into Database -> 2 % with Capacity

        m.calc_BatteryFillLevel = pyo.Constraint(m.t, rule=calc_StateOfCharge)

        # # Photovoltaic
        def calc_UseOfPhotovoltaic(m, t):
            return m.BatCharge[t] == m.PhotovoltaicProfile[t] - m.PhotovoltaicDirect[t] - m.PhotovoltaicFeedin[t]

        m.calc_UseOfPhotovoltaic_rule = pyo.Constraint(m.t, rule=calc_UseOfPhotovoltaic)

        # Sum of the Loads
        def calc_SumOfLoads(m, t):
            return m.SumOfLoads[t] == m.LoadProfile[t] \
                   + ((m.Q_TankHeating[t] / m.COP_dynamic[t]) / 1_000) \
                   + (m.Q_RoomCooling[t] / 3 / 1_000) \
                   + (m.HWPart1[t] / m.COP_dynamic[t]) \
                   + (m.HWPart2[t] / m.COP_HotWater[t])

        m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)

        # Electrical energy balance: supply and demand
        def calc_ElectricalEnergyBalance(m, t):

            return m.SumOfLoads[t] + m.PhotovoltaicFeedin[t] + m.BatCharge[t] == m.GridCover[t] + m.BatDischarge[t] + \
                   m.PhotovoltaicProfile[t]

        m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

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

    starttime = 6000
    endtime = 6150

    # exogenous profiles
    ElectricityPrice = np.array(list(instance.ElectricityPrice.extract_values().values())[starttime: endtime])
    PhotovoltaicProfile = np.array(list(instance.PhotovoltaicProfile.extract_values().values())[starttime: endtime])

    LoadProfile = np.array(list(instance.LoadProfile.extract_values().values())[starttime: endtime])

    # tank and building
    Q_TankHeating = np.array(list(instance.Q_TankHeating.extract_values().values())[starttime: endtime]) / 1000
    ElectricityDemandHeatPump = np.array(
        list(instance.Q_TankHeating.extract_values().values())[starttime: endtime]) / 1000 / \
                                np.array(list(instance.COP_dynamic.extract_values().values())[starttime: endtime])
    Q_Solar = np.array(list(instance.Q_Solar.extract_values().values())[starttime: endtime]) / 1_000  # from W to kW
    Q_RoomHeating = np.array(
        list(instance.Q_RoomHeating.extract_values().values())[starttime: endtime]) / 1000  # from W to kW
    Q_RoomCooling = np.array(
        list(instance.Q_RoomCooling.extract_values().values())[starttime: endtime]) / 1000  # from W to kW
    ElectricityCooling = np.array(
        list(instance.Q_RoomCooling.extract_values().values())[starttime: endtime]) / 1000 / 3
    T_room = np.array(list(instance.T_room.extract_values().values())[starttime: endtime])
    E_tank = np.array(list(instance.E_tank.extract_values().values())[starttime: endtime])
    T_tank = ((pd.Series(E_tank) / (M_WaterTank * CWater)) - 273.15)

    # battery, grid
    GridCover = np.array(list(instance.GridCover.extract_values().values())[starttime: endtime])
    StateOfCharge = np.array(list(instance.StateOfCharge.extract_values().values())[starttime: endtime])
    BatDischarge = np.array(list(instance.BatDischarge.extract_values().values())[starttime: endtime])
    BatCharge = np.array(list(instance.BatCharge.extract_values().values())[starttime: endtime])
    PhotovoltaicDirect = np.array(list(instance.PhotovoltaicDirect.extract_values().values())[starttime: endtime])
    PhotovoltaicFeedin = np.array(list(instance.PhotovoltaicFeedin.extract_values().values())[starttime: endtime])
    SumOfLoads = np.array(list(instance.SumOfLoads.extract_values().values())[starttime: endtime])

    # Hot water
    HotWater1 = np.array(list(instance.HWPart1.extract_values().values())[starttime: endtime]) / \
                np.array(list(instance.COP_dynamic.extract_values().values())[starttime: endtime])

    HotWater2 = np.array(list(instance.HWPart2.extract_values().values())[starttime: endtime]) / \
                np.array(list(instance.COP_HotWater.extract_values().values())[starttime: endtime])
    HotWater = HotWater1 + HotWater2

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

    YearlyGridCover = np.sum(GridCover)
    print('Yearly cover from electricity grid: ' + str(YearlyGridCover) + ' kWh')

    YearlyPVGeneration = np.sum(PhotovoltaicProfile)
    print('Yearly generation of Photvoltaic power: (y) ' + str(YearlyPVGeneration) + ' kWh')

    YearlyUseOfPhotovoltaic = np.sum(PhotovoltaicDirect)
    print('Yearly self consumption of photovoltaic : ' + str(YearlyUseOfPhotovoltaic) + ' kWh')

    YearlyStoredEnergy = np.sum(BatCharge)
    print('Yearly charged power in battery : ' + str(YearlyStoredEnergy) + ' kWh')

    YearlyDischargedEnergy = np.sum(BatDischarge)
    print('Yearly discharged power in battery : ' + str(YearlyDischargedEnergy) + ' kWh')

    YearlyFeedin = np.sum(PhotovoltaicFeedin)
    print('Yearly photovoltaic feed in  : ' + str(YearlyFeedin) + ' kWh')

    YearlyElectricityDemand = np.sum(SumOfLoads)
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

    SelfConsumption = round((YearlyUseOfPhotovoltaic + YearlyStoredEnergy) / YearlyPVGeneration, 2)
    SelfSufficiency = round((YearlyUseOfPhotovoltaic + YearlyStoredEnergy) / YearlyElectricityDemand, 2)

    x_achse = np.arange(starttime, endtime)

    # Plot (1) Room and building
    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax1.plot(x_achse, Q_TankHeating, label="Q_TankHeating", color=colors["Q_TankHeating"], linewidth=0.25, alpha=0.6)
    ax1.plot(x_achse, Q_RoomHeating, label="Q_RoomHeating", color=colors["Q_RoomHeating"], linewidth=0.25, alpha=0.6)
    ax1.plot(x_achse, Q_RoomCooling, label="Q_RoomCooling", color=colors["Q_RoomCooling"], linewidth=0.25, alpha=0.6)
    ax1.plot(x_achse, Q_Solar, label="Q_Solar", color=colors["Q_Solar"], linewidth=0.5, alpha=0.6)
    ax2.plot(x_achse, ElectricityPrice, color=colors["ElectricityPrice"], label="Price", linewidth=0.50, linestyle=':',
             alpha=0.6)

    ax1.set_ylabel("Energy kW")
    ax2.set_ylabel("Price per kWh in Ct/€")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="TempRoom", color=colors["T_room"], linewidth=0.15, alpha=0.8)
    ax4.plot(x_achse, T_tank, label="TempTank", color=colors["T_tank"], linewidth=0.15, alpha=0.8)

    ax3.set_ylabel("Room temperature in °C", color='black')
    ax4.set_ylabel("Tank Temperature in °C", color='black')

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()
    ax1.set_title(
        '(1) + (2) Yearly heat generation: ' + str(round(YearlyHeatGeneration / 1000, 2)) + ' kWh ' + 'APF: ' + \
        str(JAZ))
    plt.tight_layout()
    fig.savefig('Room and building', dpi=300)
    plt.show()

    # # Plot (2) SoC: Charge and Discharge
    # fig, ax1 = plt.subplots()
    #
    # ax1.plot(x_achse, BatCharge, linewidth=0.5, label='BatCharge', color=colors["BatCharge"], alpha=0.8)
    # ax1.plot(x_achse, BatDischarge, linewidth=0.5, label='BatDischarge', color=colors["BatDischarge"], alpha=0.8)
    # ax1.set_ylabel("Power kW")
    #
    # ax2 = ax1.twinx()
    # ax2.plot(x_achse, StateOfCharge, linewidth=0.75, label='SoC', color=colors["StateOfCharge"], alpha=0.2)
    #
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2, loc = 'upper right')
    # ax2.set_ylabel("SoC in kWh")
    #
    # plt.title('(2) SoC: Charge and Discharge')
    # plt.tight_layout()
    # fig.savefig('Battery Charge Discharge', dpi=300)
    # plt.show()
    #
    # # Plot (3) Use of the PV Power
    # fig, ax1 = plt.subplots()
    #
    # ax1.plot(x_achse, BatCharge, linewidth=0.5, label='BatCharge', color=colors["BatCharge"], alpha=0.6)
    # ax1.plot(x_achse, PhotovoltaicFeedin, linewidth=0.5, label='PhotovoltaicFeedin', color=colors["PhotovoltaicFeedin"],
    #          alpha=0.6)
    # ax1.plot(x_achse, PhotovoltaicDirect, linewidth=0.5, label='PhotovoltaicDirect', color=colors["PhotovoltaicDirect"],
    #          alpha=0.6)
    # ax1.plot(x_achse, PhotovoltaicProfile, linewidth=1.5, label='PhotovoltaicProfile',
    #          color=colors["PhotovoltaicProfile"], alpha=0.3)
    #
    # ax1.set_ylabel("Power kW")
    # plt.legend(loc = 'upper right')
    # plt.title('(3) Use of the PV Power')
    # plt.tight_layout()
    # fig.savefig('Use of the PV Power', dpi=300)
    # plt.show()
    #
    # # Plot (4) Supply of Loads
    # fig, ax1 = plt.subplots()
    #
    # ax1.plot(x_achse, SumOfLoads, linewidth=0.5, label='SumOfLoads', color=colors["SumOfLoads"], alpha=0.2)
    # ax1.plot(x_achse, GridCover, linewidth=0.5, label='GridCover', color=colors["GridCover"], alpha=0.6)
    # ax1.plot(x_achse, BatDischarge, linewidth=0.5, label='BatDischarge', color=colors["BatDischarge"], alpha=0.6)
    # ax1.plot(x_achse, PhotovoltaicDirect, linewidth=0.5, label='PhotovoltaicDirect', color=colors["PhotovoltaicDirect"],
    #          alpha=0.6)
    #
    # ax1.set_ylabel("Power kW")
    # plt.legend(loc = 'upper right')
    # plt.title('(4) Supply of Loads: ' + 'Self-consumption: ' + str(SelfConsumption) + ' Self-sufficiency: ' + str(
    #     SelfSufficiency))
    # plt.tight_layout()
    # fig.savefig('Supply of Loads', dpi=300)
    # plt.show()
    #
    # # Plot (5) Loads
    # fig, ax1 = plt.subplots()
    #
    # ax1.plot(x_achse, SumOfLoads, linewidth=1, label='SumOfLoads', color=colors["SumOfLoads"], alpha=0.5)
    # ax1.plot(x_achse, BatCharge, linewidth=0.5, label='BatCharge', color=colors["BatCharge"], alpha=0.5)
    # ax1.plot(x_achse, ElectricityDemandHeatPump, linewidth=0.3, label='ElectricityDemandHeatPump', color='orange',
    #          alpha=0.6)
    # ax1.plot(x_achse, ElectricityCooling, linewidth=0.5, label='ElectricityCooling', color='blue', alpha=0.6)
    # ax1.plot(x_achse, LoadProfile, linewidth=0.5, label='LoadProfile', color='green', alpha=0.6)
    #
    # ax1.set_ylabel("Power kW")
    # plt.legend(loc = 'upper right')
    # plt.title('(5) Electrical Loads')
    # plt.tight_layout()
    # fig.savefig('Electrical Loads', dpi=300)
    # plt.show()

    # Plt (6) Test with bar diagramm

    plt.figure(figsize=(16, 9))
    plt.bar(x_achse, LoadProfile, color="red", label="LoadProfile")

    plt.bar(x_achse, HotWater, color="blue", bottom=np.array(LoadProfile), label="HotWater")

    plt.bar(x_achse, ElectricityCooling, color="cyan", bottom=np.array(LoadProfile) + np.array(HotWater),
            label="ElectricityCooling")
    plt.bar(x_achse, ElectricityDemandHeatPump, color="grey",
            bottom=np.array(LoadProfile) + np.array(HotWater) + np.array(ElectricityCooling),
            label="ElectricityDemandHeatPump")
    plt.bar(x_achse, BatCharge, color="orange",
            bottom=np.array(LoadProfile) + np.array(HotWater) + np.array(ElectricityDemandHeatPump) + np.array(
                ElectricityCooling),
            label="BatCharge")
    plt.bar(x_achse, PhotovoltaicFeedin, color="green",
            bottom=np.array(LoadProfile) + np.array(HotWater) + np.array(ElectricityDemandHeatPump) + np.array(
                ElectricityCooling) + np.array(BatCharge),
            label="PhotovoltaicFeedin")

    plt.plot(x_achse, PhotovoltaicProfile, linewidth=2, label='PhotovoltaicProfile',
             color=colors["PhotovoltaicProfile"], alpha=0.50)
    plt.plot(x_achse, GridCover, linewidth=1, label='GridCover',
             color= 'black', alpha=0.50)
    plt.plot(x_achse, BatDischarge, linewidth=1, label='BatDischarge',
             color=colors["BatDischarge"], alpha=0.5)


    plt.legend(loc = 'upper right')
    plt.ylabel('Power in kWh')
    plt.xlabel('Hour of year')
    plt.title('(6) Sum of Loads')
    plt.grid()
    plt.savefig('Electrical Loads in bar diagramm', dpi=300)
    plt.show()

    # # hot water plot
    #
    # plt.figure(figsize=(16, 9))
    # plt.bar(x_achse, HotWater1, color="blue", label="HotWater1")
    # plt.bar(x_achse, HotWater2, color="green", bottom=np.array(HotWater1),
    #         label="HotWater2")
    #
    # plt.plot(x_achse, HotWater, linewidth=2, label='HotWater',
    #          color='red', alpha=0.30)
    # plt.legend()
    # plt.ylabel('Power in kWh')
    # plt.xlabel('Hour of year')
    # plt.title('(7) Hot Water Demand ')
    # plt.grid()
    # plt.savefig('HotWater', dpi=600)
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
