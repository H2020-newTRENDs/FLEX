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

        ############################################################################################
        # (1) Reading data from DB
        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG().Gen_Sce_ID_Environment, self.Conn)

        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature, self.Conn)
        self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        self.Radiation_SkyDirections = DB().read_DataFrame(REG().Sce_Weather_Radiation_SkyDirections, self.Conn)

        self.ElectricityPrice = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FiT = DB().read_DataFrame(REG().Sce_Price_HourlyFeedinTariff, self.Conn)
        self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG().Sce_Demand_HotWaterProfile, self.Conn)
        self.CarAtHomeStatus = DB().read_DataFrame(REG().Gen_Sce_CarAtHomeHours, self.Conn)
        self.Demand_EV = DB().read_DataFrame(REG().Sce_Demand_ElectricVehicleBehavior, self.Conn)

        self.DishWasherHours = DB().read_DataFrame(REG().Gen_Sce_DishWasherHours, self.Conn)
        self.Sce_Demand_DishWasher = DB().read_DataFrame(REG().Sce_Demand_DishWasher, self.Conn)

        self.WashingMachineHours = DB().read_DataFrame(REG().Gen_Sce_WashingMachineHours, self.Conn)
        self.Sce_Demand_WashingMachine = DB().read_DataFrame(REG().Sce_Demand_WashingMachine, self.Conn)

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

    def run_Optimization(self, household_id, environment_id):

        ############################################################################################
        # (2) Select HH and scenario

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

        # fucntion to convert a list into a dict (needed for pyomo data input)
        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary

        # the data of the simulation are indexed for 1 year, start and stoptime in visualization
        HoursOfSimulation = 8760

        ############################################################################################
        # (3) Define reading of data from DB

        # (3.1) Smart Technologies
        # DishWasher = 200 * 1,1 kWh = 216
        # WashingMachine = 150 * 0,85 kWh = 127,5 kWh
        # Dryer = 150 * 0,85 kWh = 375 kWh
        # yearly sum = 718,5 kWh -> The yearly BaseProfile is reduced with this part

        # (3.1.1) DishWasher

        # This is 1, only if there is no Dishwasher in the household, then 0
        DishWasherAdoption = Household.ApplianceGroup.DishWasherAdoption

        DishWasherTheoreticalHours = (self.DishWasherHours.DishWasherHours.to_numpy()) * DishWasherAdoption
        HourOfDay = self.TimeStructure.ID_DayHour.to_numpy()
        DishWasherDuration = int(self.Sce_Demand_DishWasher.DishWasherDuration)
        DishWasherStartTime = int(self.Sce_Demand_DishWasher.DishWasherStartTime)

        DishWasherPower = Household.ApplianceGroup.DishWasherPower
        DishWasherSmartStatus = Household.ApplianceGroup.DishWasherShifting

        # (3.1.2) WashingMachine

        # This is 1, only if there is no Dishwasher in the household, then 0
        WashingMachineAdoption = Household.ApplianceGroup.WashingMachineAdoption

        WashingMachineTheoreticalHours = (
                                             self.WashingMachineHours.WashingMachineHours.to_numpy()) * WashingMachineAdoption
        WashingMachineDuration = int(self.Sce_Demand_WashingMachine.WashingMachineDuration)
        WashingMachineStartTime = int(self.Sce_Demand_WashingMachine.WashingMachineStartTime)

        WashingMachinePower = Household.ApplianceGroup.WashingMachinePower
        WashingMachineSmartStatus = Household.ApplianceGroup.WashingMachineShifting

        # (3.1.3) Dryer

        # Dryer is depending on Dishwasher and runs after the last hour of WashingMachine
        # If there is no WashingMachine adopted, there is also no Dryer

        # This is 1, only if there is no Dishwasher in the household, then 0
        DryerPower = Household.ApplianceGroup.DryerPower
        DryerDuration = int(self.Sce_Demand_Dryer.DryerDuration)
        DryerAdoption = Household.ApplianceGroup.DryerAdoption

        # toDo:
        # () Done: Last day of year? With or 8759 or 8760 ...
        # () Done: changing Demand Profiles with different duration hours: Between 1-3 Hours is in code
        # () Done: if valid: change UseDays to UseHours in databank
        # () Done: Smart ON / OFF: use fixed use time, all functions included?
        # () Done: Adoption to 1 and 0
        # () Done: Extend to WaschingMachine
        # () Done: How to use Dryer? Profile? - after last hour of WashingMachine
        # () DOne: Extend to Dryer
        # () Done: Minimize LoadProfile: 2376 - 718 = 1658 kWh: 2 profiles now in the DB
        # (-) create Gen_Sce_Appliances... + extent Environment with DemandProfiles of SmartApp

        ############################################################################################
        # (3.2) EV

        # (3.2.1) Check if the EV adopted, by checking the capacity of the EV

        # Case: BatteryCapacity = 0: EV not adopted
        if Household.ElectricVehicle.BatterySize == 0:
            CarAtHomeStatus = create_dict([0] * HoursOfSimulation)
            V2B = 0  # Vif EV is not adopted, V2B have to be 0

        # Case: BatteryCapacity > 0: EV is adopted
        else:
            CarAtHomeStatus = create_dict(self.CarAtHomeStatus.CarAtHomeHours.to_numpy())
            V2B = Household.ElectricVehicle.V2B

        # (2.3) Calculation of the hourly EV demand for the discharge of the EV, if not at home
        EV_DailyDemand = int(self.Demand_EV.EVDailyElectricityConsumption)
        EV_LeaveHour = int(self.Demand_EV.EVLeaveHomeClock)
        EV_ArriveHour = int(self.Demand_EV.EVArriveHomeClock)
        EV_AwayHours = EV_ArriveHour - EV_LeaveHour
        EV_HourlyDemand = EV_DailyDemand / EV_AwayHours

        ############################################################################################
        # (3.3) Calculation of the installed PV-Power with the PVProfile and installed Power

        # (3.3.1) LoadProfile, BaseLoadProfile is 2376 kWh, SmartAppElectricityProfile is 1658 kWh
        # LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        LoadProfile = self.LoadProfile.SmartAppElectricityProfile.to_numpy()

        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile.to_numpy()
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH

        # (3.3.2) Load Profile

        SumOfPV = round(sum(PhotovoltaicProfileHH), 2)
        SumOfLoad = round(sum(self.LoadProfile.BaseElectricityProfile), 2)

        ############################################################################################

        # Cooling
        print('cooling')
        print(Household.SpaceCooling.SpaceCoolingEfficiency)

        # (3.4) Defintion of parameters for Heating and Cooling

        # (3.4.1) Tank

        # fixed starting values:
        T_TankStart = 45  # °C
        # min,max tank temperature for boundary of energy
        T_TankMax = 50  # °C
        T_TankMin = 25  # °C
        # sourounding temp of tank
        T_TankSourounding = 20  # °C
        # starting temperature of thermal mass 20°C
        CWater = 4200 / 3600

        # Set into DB for scenario
        HeatingTargetTemperature = 20
        CoolingTargetTemperature = 24

        # Parameters of SpaceHeatingTank
        # Mass of water in tank
        M_WaterTank = Household.SpaceHeating.TankSize
        # Surface of Tank in m2
        A_SurfaceTank = Household.SpaceHeating.TankSurfaceArea
        # insulation of tank, for calc of losses
        U_ValueTank = Household.SpaceHeating.TankLoss

        # (4.2) RC-Model

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

        # window areas in celestial directions
        Awindows_rad_east_west = Household.Building.average_effective_area_wind_west_east_red_cool
        Awindows_rad_south = Household.Building.average_effective_area_wind_south_red_cool
        Awindows_rad_north = Household.Building.average_effective_area_wind_north_red_cool
        # solar gains from different celestial directions

        Q_sol_north = np.outer(self.Radiation_SkyDirections.RadiationNorth, Awindows_rad_north)
        Q_sol_south = np.outer(self.Radiation_SkyDirections.RadiationSouth, Awindows_rad_south)
        Q_sol_east_west = np.outer(self.Radiation_SkyDirections.RadiationEast + \
                                   self.Radiation_SkyDirections.RadiationWest, Awindows_rad_east_west)
        Q_sol = (Q_sol_north + Q_sol_south + Q_sol_east_west).squeeze()

        ############################################################################################
        # (3.5) Pricing of electricity

        # (3.5.1) ElectricityPriceType == 1: constant, == 2: flexible

        if ID_ElectricityPriceType == 1:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 1].loc[:,
                               'HourlyElectricityPrice'].to_numpy()
            ElectricityPrice = list(ElectricityPrice)
            ElectricityPrice = ElectricityPrice * HoursOfSimulation

        elif ID_ElectricityPriceType == 2:
            ElectricityPrice = self.ElectricityPrice.loc[self.ElectricityPrice['ID_ElectricityPriceType'] == 2].loc[:,
                               'HourlyElectricityPrice'].to_numpy()

        # (3.5.2) FiTType == 1: constant, ==2: flexible

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

        ############################################################################################
        # (3.6) Selection of COP for SpaceHeating and HotWater

        # (3.6.1) COP for SpaceHeating: HP Air or HP Water

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

        # (3.6.2) COP for HotWater: Until 35°C HotWater is heated by SpaceHeating, rest with COP HotWater
        # Type of HP is given by SpaceHeating

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

        ############################################################################################
        # (3.7) HotWater Part1 and HotWater Part 2 from the DB

        # Hot Water Demand with Part 1 (COP SpaceHeating) and Part 2 (COP HotWater, lower)
        HotWaterProfile1 = self.HotWaterProfile.loc[self.HotWaterProfile['ID_HotWaterProfileType'] == 1].loc[:,
                           'HotWaterPart1'].to_numpy()
        HotWaterProfile2 = self.HotWaterProfile.loc[self.HotWaterProfile['ID_HotWaterProfileType'] == 1].loc[:,
                           'HotWaterPart2'].to_numpy()

        ############################################################################################
        # (4) Pyomo Model for optimisation

        m = pyo.AbstractModel()

        # (4.1) Parameters

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

        # CarAtHomeStatus
        m.CarAtHomeStatus = pyo.Param(m.t, initialize=CarAtHomeStatus)

        # Smart Technologies
        m.DishWasherTheoreticalHours = pyo.Param(m.t, within=pyo.Binary,
                                                 initialize=create_dict(DishWasherTheoreticalHours))
        m.HourOfDay = pyo.Param(m.t, initialize=create_dict(HourOfDay))
        m.WashingMachineTheoreticalHours = pyo.Param(m.t, within=pyo.Binary,
                                                     initialize=create_dict(WashingMachineTheoreticalHours))

        ############################################################################################
        # (4.2) Variables of model

        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))  # max Power of Boiler; DB?
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * M_WaterTank * (273.15 + T_TankMin),
                                                                     CWater * M_WaterTank * (273.15 + T_TankMax)))
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))

        # energy used for cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals,
                                  bounds=(0, Household.SpaceCooling.SpaceCoolingPower))  # 6kW thermal, 2 kW electrical
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals,
                           bounds=(HeatingTargetTemperature, CoolingTargetTemperature))  # Change to TargetTemp
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 60))

        # Grid, limit set by 21 kW
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))  # 380 * 32 * 1,72
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        # PV
        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 21))

        # Battery
        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.Capacity))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxChargePower))
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))

        # EV
        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals,
                                bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals,
                             bounds=(0, Household.ElectricVehicle.BatteryMaxChargePower))
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals,
                            bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals,
                           bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatterySize))

        # Smart Technologies

        m.DishWasher1 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher2 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher3 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasherStart = pyo.Var(m.t, within=pyo.Binary)

        m.WashingMachine1 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine2 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine3 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachineStart = pyo.Var(m.t, within=pyo.Binary)

        m.Dryer1 = pyo.Var(m.t, within=pyo.Binary)
        m.Dryer2 = pyo.Var(m.t, within=pyo.Binary)

        ############################################################################################
        # (4.3) Objective of model

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        ############################################################################################
        # (4.4) Constraints of model

        # (4.4.1) Electricity flow

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
                return m.Grid[t] == m.Grid2Load[t] + m.Grid2EV[t] * m.CarAtHomeStatus[t]

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (3)
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] == \
                   m.PhotovoltaicProfile[t]

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
                return m.Grid2Load[t] + m.PV2Load[t] + m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B == m.Load[t]
            else:
                return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B == \
                       m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # (6)
        def calc_SumOfLoads(m, t):
            return m.Load[t] == m.LoadProfile[t] \
                   + ((m.Q_TankHeating[t] / m.SpaceHeatingHourlyCOP[t]) / 1_000) \
                   + ((m.Q_RoomCooling[t] / Household.SpaceCooling.SpaceCoolingEfficiency/ 1_000)) \
                   + (m.HWPart1[t] / m.SpaceHeatingHourlyCOP[t]) \
                   + (m.HWPart2[t] / m.HotWaterHourlyCOP[t]) \
                   + (m.DishWasher1[t] + m.DishWasher2[t] + m.DishWasher3[t]) * DishWasherPower \
                   + (m.WashingMachine1[t] + m.WashingMachine2[t] + m.WashingMachine3[t]) * WashingMachinePower \
                   + (m.Dryer1[t] + m.Dryer2[t]) * DryerPower

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
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarAtHomeStatus[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8)
        def calc_BatCharge(m, t):
            if Household.Battery.Capacity == 0:
                return m.BatCharge[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.BatCharge[t] == m.PV2Bat[t]
            else:
                return m.BatCharge[t] == m.PV2Bat[t] + m.EV2Bat[t] * V2B

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
            elif m.CarAtHomeStatus[t] == 0:
                return m.EVCharge[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * \
                       m.CarAtHomeStatus[t]
            else:
                return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * \
                       m.CarAtHomeStatus[t] + \
                       m.Bat2EV[t] * m.CarAtHomeStatus[t]

        m.calc_EVCharge = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (11)
        def calc_EVDischarge(m, t):
            if t == 1:
                return m.EVDischarge[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.EVDischarge[t] == 0
            elif m.CarAtHomeStatus[t] == 0 and V2B == 1:
                return m.EVDischarge[t] == 0
            elif m.t[t] == HoursOfSimulation:
                return m.EVDischarge[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.EVDischarge[t] == m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B \
                       + m.EV2Bat[t] * m.CarAtHomeStatus[t] * V2B

        m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (12)
        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.EVSoC[t] == 0
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + (
                        m.EVCharge[t] * m.CarAtHomeStatus[t] * Household.ElectricVehicle.BatteryChargeEfficiency) \
                       - (m.EVDischarge[t] * m.CarAtHomeStatus[t] * (
                        1 + (1 - Household.ElectricVehicle.BatteryDischargeEfficiency))) \
                       - (EV_HourlyDemand * (1 - m.CarAtHomeStatus[t]))

        m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)

        ############################################################################################
        # (4.4.2) Smart Technologies

        def calc_DishWasherHours2(m, t):
            if t >= 8759:
                return m.DishWasher2[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and (DishWasherDuration == 2 or DishWasherDuration == 3):
                return m.DishWasher1[t] == m.DishWasher2[t + 1]
            return m.DishWasher2[t] == 0

        m.calc_DishWasherHours2 = pyo.Constraint(m.t, rule=calc_DishWasherHours2)

        def calc_DishWasherHours3(m, t):
            if t >= 8758:
                return m.DishWasher2[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and DishWasherDuration == 3:
                return m.DishWasher1[t] == m.DishWasher3[t + 2]
            return m.DishWasher3[t] == 0

        m.calc_DishWasherHours3 = pyo.Constraint(m.t, rule=calc_DishWasherHours3)

        def calc_DishWasherStartTime(m, t):
            if m.t[t] == 1:
                return m.DishWasherStart[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and m.HourOfDay[
                t] == DishWasherStartTime and DishWasherSmartStatus == 0:
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
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.HourOfDay[
                t] == WashingMachineStartTime and WashingMachineSmartStatus == 0:
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
            return m.WashingMachine1[t] == m.Dryer1[t + 3]

        m.calc_Dryer1 = pyo.Constraint(m.t, rule=calc_Dryer1)

        def calc_Dryer2(m, t):
            if t >= 8756:
                return m.Dryer2[t] == 0
            if DryerAdoption == 0:
                return m.Dryer2[t] == 0
            elif DryerDuration == 2:
                return m.WashingMachine1[t] == m.Dryer2[t + 4]
            return m.Dryer2[t + 4] == 0

        m.calc_Dryer2 = pyo.Constraint(m.t, rule=calc_Dryer2)

        ############################################################################################
        # (4.4.3) Tank and HP

        # energy input and output of tank energy
        def tank_energy(m, t):
            if t == 1:
                return m.E_tank[t] == CWater * M_WaterTank * (273.15 + T_TankStart)
            else:
                return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] \
                       - U_ValueTank * A_SurfaceTank * ((m.E_tank[t] / (M_WaterTank * CWater)) - T_TankSourounding)

        m.tank_energy_rule = pyo.Constraint(m.t, rule=tank_energy)

        ############################################################################################
        # (4.4.4) RC-Model

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
        cost = instance.OBJ()
        print(cost)
        return cost


    def run(self):
        TargetTable_list = []
        for household_id in range(0, 63):
            for environment_id in range(1, 2):
                cost = round(self.run_Optimization(household_id, environment_id),2)
                TargetTable_list.append([household_id, environment_id, cost])

        TargetTable_columns = ['ID_Household', 'ID_Environment', "TotalCost"]
        DB().write_DataFrame(TargetTable_list, REG().Res_MinimizedOperationCost, TargetTable_columns, self.Conn)
        pass