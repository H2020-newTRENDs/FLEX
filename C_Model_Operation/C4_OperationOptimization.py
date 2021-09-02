# -*- coding: utf-8 -*-
__author__ = 'Thomas, Philip, Songmin'

import pyomo.environ as pyo
import numpy as np
import sys as sys

from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C2_DataCollector import DataCollector
from B_Classes.B1_Household import Household
from A_Infrastructure.A1_CONS import CONS
import time


class OperationOptimization:

    def __init__(self, conn=DB().create_Connection(CONS().RootDB)):

        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.Conn)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.Temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)

        self.ElectricityPrice = DB().read_DataFrame(REG_Table().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile, self.Conn)

        self.DishWasherHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherHours, self.Conn)
        self.DishWasherRandomHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherStartingHours, self.Conn)
        self.Sce_Demand_DishWasher = DB().read_DataFrame(REG_Table().Sce_Demand_DishWasher, self.Conn)

        self.WashingMachineHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineHours, self.Conn)
        self.WashingMachineRandomHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineStartingHours, self.Conn)
        self.Sce_Demand_WashingMachine = DB().read_DataFrame(REG_Table().Sce_Demand_WashingMachine, self.Conn)
        self.DryerRandomHours = DB().read_DataFrame(REG_Table().Gen_Sce_DryerStartingHours, self.Conn)
        self.Demand_Dryer = DB().read_DataFrame(REG_Table().Sce_Demand_Dryer, self.Conn)

        self.PhotovoltaicProfile = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.Radiation_SkyDirections = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections,
                                                           self.Conn)
        self.HeatPump_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)
        self.AC_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_AC_HourlyCOP, self.Conn)

        self.TargetTemperature = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)
        self.EnergyCost = DB().read_DataFrame(REG_Table().Sce_Price_EnergyCost, self.Conn)

        # C_Water
        self.CPWater = 4200 / 3600

    def gen_Household(self, row_id):
        ParaSeries = self.ID_Household.iloc[row_id]
        HouseholdAgent = Household(ParaSeries, self.Conn)
        return HouseholdAgent

    def gen_Environment(self, row_id):
        EnvironmentSeries = self.ID_Environment.iloc[row_id]
        return EnvironmentSeries

    def creat_Dict(self, value_list):
        Dictionary = {}
        for index, value in enumerate(value_list, start=1):
            Dictionary[index] = value
        return Dictionary

    def get_input_data(self, household_RowID, environment_RowID):  # TODO achtung mit water-water HP gehts nicht!
        time_start = time.time()
        print('Optimization: ID_Household = ' + str(household_RowID+1) + ", ID_Environment = " + str(environment_RowID+1))
        Household = self.gen_Household(household_RowID)  # Gen_OBJ_ID_Household
        Environment = self.gen_Environment(environment_RowID)  # Gen_Sce_ID_Environment

        # ############################
        # PART I. Household definition
        # ############################

        # ------------------
        # 1. BaseLoadProfile
        # ------------------

        # BaseLoadProfile is 2376 kWh, SmartAppElectricityProfile is 1658 kWh
        BaseLoadProfile = self.BaseLoadProfile.loc[
            (self.BaseLoadProfile['ID_BaseElectricityProfileType'] == Environment["ID_BaseElectricityProfileType"]) &
            (self.BaseLoadProfile['ID_HouseholdType'] == Household.ID_HouseholdType)]['BaseElectricityProfile'].\
                              to_numpy() * 1_000  # W

        # -------------------
        # 2. Smart appliances
        # -------------------
        randomApplianceProfile = 0  # TODO select automatically random profile

        # DishWasher
        DishWasherSmartStatus = int(
            Household.ApplianceGroup.DishWasherShifting)  # 1 = dishwasher is smart, 0 = not smart
        if DishWasherSmartStatus == 0:
            DishWasherHours = self.DishWasherRandomHours["DishWasherHours " + str(randomApplianceProfile)].to_numpy()
        elif DishWasherSmartStatus == 1:
            DishWasherHours = self.DishWasherHours["DishWasherHours " + str(randomApplianceProfile)].to_numpy()
        else:
            raise ValueError('No valid DishWasher smartStatus was provided')
        DishWasherPower = float(Household.ApplianceGroup.DishWasherPower) * 1_000
        DishWasherDuration = int(self.Sce_Demand_DishWasher["DishWasherDuration"])

        # WashingMachine and Dryer
        WashingMachineSmartStatus = int(Household.ApplianceGroup.WashingMachineShifting)
        if WashingMachineSmartStatus == 0:
            WashingMachineHours = self.WashingMachineRandomHours["WashingMachineHours " + str(randomApplianceProfile)].to_numpy()
            DryerHours = self.DryerRandomHours["DryerHours " + str(randomApplianceProfile)].to_numpy()
        elif WashingMachineSmartStatus == 1:
            WashingMachineHours = self.WashingMachineHours["WashingMachineHours " + str(randomApplianceProfile)].to_numpy()
            DryerHours = np.zeros(shape=(self.OptimizationHourHorizon,))  # they will be defined in the model
        else:
            raise ValueError('No valid Washingmachine smartStatus was provided')

        WashingMachinePower = float(Household.ApplianceGroup.WashingMachinePower) * 1_000
        WashingMachineDuration = int(self.Sce_Demand_WashingMachine.WashingMachineDuration)
        DryerPower = float(Household.ApplianceGroup.DryerPower) * 1_000
        DryerDuration = int(self.Demand_Dryer["DryerDuration"])
        DryerAdoption = int(Household.ApplianceGroup.DryerAdoption)

        # ----------------------------
        # 3. Space heating and cooling
        # ----------------------------

        # (3.1) Tank

        # fixed starting values:
        T_TankStart = float(Household.SpaceHeating.TankStartTemperature)
        # min,max tank temperature for boundary of energy
        T_TankMax = float(Household.SpaceHeating.TankMaximalTemperature)
        T_TankMin = float(Household.SpaceHeating.TankMinimalTemperature)
        # surrounding temp of tank
        T_TankSurrounding = float(Household.SpaceHeating.TankSurroundingTemperature)


        # Parameters of SpaceHeatingTank
        # Mass of water in tank
        M_WaterTank = float(Household.SpaceHeating.TankSize)
        # Surface of Tank in m2
        A_SurfaceTank = float(Household.SpaceHeating.TankSurfaceArea)
        # insulation of tank, for calc of losses
        U_ValueTank = float(Household.SpaceHeating.TankLoss)

        # (3.2) RC-Model

        # Building Area
        Af = float(Household.Building.Af)  # konditionierte Nutzfläche (conditioned usable floor area)
        Atot = 4.5 * Af  # 7.2.2.2: Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen (Area of all surfaces facing the building zone)
        # Transmission Coefficient
        Hve = float(Household.Building.Hve)  # Air
        Htr_w = float(Household.Building.Htr_w)  # Wall
        Hop = float(Household.Building.Hop)  # opake Bauteile (opaque components)
        # Speicherkapazität (Storage capacity) [J/K]
        Cm = float(Household.Building.CM_factor) * Af
        # wirksame Massenbezogene Fläche (Effective mass related area) [m^2]
        Am = float(Household.Building.Am_factor) * Af
        # internal gains
        Qi = float(Household.Building.spec_int_gains_cool_watt) * Af
        # Coupling values
        his = np.float_(
            3.45)  # 7.2.2.2 - Kopplung Temp Luft mit Temp Surface Knoten s (Coupling Temp Air with Temp Surface node s)
        hms = np.float_(
            9.1)  # [W / m2K] from Equ.C.3 (from 12.2.2) - kopplung zwischen Masse und zentralen Knoten s (coupling between mass and central node s)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte (thermal coupling values) [W/K]
        Htr_is = his * Atot
        PHI_ia = 0.5 * Qi  # Equ. C.1
        Htr_1 = 1 / (1 / Hve + 1 / Htr_is)  # Equ. C.6
        Htr_2 = Htr_1 + Htr_w  # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

        # (3.3) Solar gains
        # solar gains from different celestial directions
        radiation = self.Radiation_SkyDirections
        # window areas in celestial directions
        AreaWindowEastWest = Household.Building.average_effective_area_wind_west_east_red_cool
        AreaWindowSouth = Household.Building.average_effective_area_wind_south_red_cool
        AreaWindowNorth = Household.Building.average_effective_area_wind_north_red_cool
        # solar gains from different celestial directions
        Q_sol_north = np.outer(radiation.north.to_numpy(), AreaWindowNorth)
        Q_sol_east = np.outer(radiation.east.to_numpy(), AreaWindowEastWest / 2)
        Q_sol_south = np.outer(radiation.south.to_numpy(), AreaWindowSouth)
        Q_sol_west = np.outer(radiation.west.to_numpy(), AreaWindowEastWest / 2)

        Q_sol = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())

        # (3.4) Selection of heat pump COP
        SpaceHeatingHourlyCOP = self.HeatPump_HourlyCOP.loc[
            self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] ==
            Household.SpaceHeating.ID_SpaceHeatingBoilerType]['SpaceHeatingHourlyCOP'].to_numpy()

        # Selection of AC COP
        ACHourlyCOP = self.AC_HourlyCOP.ACHourlyCOP.to_numpy()

        # ------------
        # 4. Hot water
        # ------------

        # Hot Water Demand with Part 1 (COP SpaceHeating) and Part 2 (COP HotWater, lower)
        HotWaterProfileSelect = self.HotWaterProfile.loc[
            (self.HotWaterProfile['ID_HotWaterProfileType'] == Environment["ID_HotWaterProfileType"]) &
            (self.HotWaterProfile['ID_Country'] == Household.ID_Country) &
            (self.HotWaterProfile['ID_HouseholdType'] == Household.ID_HouseholdType)]
        HotWaterProfile1 = HotWaterProfileSelect['HotWaterPart1'].to_numpy() * 1_000  # W
        HotWaterProfile2 = HotWaterProfileSelect['HotWaterPart2'].to_numpy() * 1_000  # W

        # COP for HotWater generation
        HotWaterHourlyCOP = self.HeatPump_HourlyCOP.loc[
            self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] == Household.SpaceHeating.ID_SpaceHeatingBoilerType][
            'HotWaterHourlyCOP'].to_numpy()

        # ----------------------
        # 5. PV and battery
        # ----------------------

        PhotovoltaicProfile = self.PhotovoltaicProfile.loc[
            (self.PhotovoltaicProfile['ID_PVType'] == Household.PV.ID_PVType) &
            (self.PhotovoltaicProfile['ID_Country'] == "AT")]['PVPower'].to_numpy() * 1_000  # W TODO nuts ID statt ID Country!

        # ###############################
        # PART II. Environment definition
        # ###############################

        # --------------------
        # 1. Electricity price
        # --------------------

        ElectricityPrice = self.ElectricityPrice.loc[
            (self.ElectricityPrice['ID_ElectricityPriceType'] == Environment["ID_ElectricityPriceType"]) &
            (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyElectricityPrice'].to_numpy()

        # -----------------
        # 2. Feed-in tariff
        # -----------------

        FeedinTariff = self.FeedinTariff.loc[(self.FeedinTariff['ID_FeedinTariffType'] == Environment["ID_FeedinTariffType"]) &
                       (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyFeedinTariff'].to_numpy()  # Cent/kWh

        # ---------------------
        # 3. Target temperature
        # ID_AgeGroup:  1 --> Young
        #               2 --> Old
        #               3 --> Young + night reduction
        #               4 --> Old + night reduction
        #               5 --> smart home
        # ---------------------
        # TODO add age groupts to database and other distinction!
        if Household.Name_AgeGroup == "Young":
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureYoung']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureYoung']
        elif Household.Name_AgeGroup == 2:
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureOld']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureOld']
        elif Household.Name_AgeGroup == 3:
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureYoungNightReduction']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureYoungNightReduction']
        elif Household.Name_AgeGroup == 4:
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureOldNightReduction']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureOldNightReduction']
        elif Household.Name_AgeGroup == "SmartHome":
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureSmartHome']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureSmartHome']

        BuildingMassTemperatureStartValue = Household.Building.BuildingMassTemperatureStartValue

        if Household.SpaceCooling.AdoptionStatus == 0:
            CoolingTargetTemperature = np.full((8760,), 60)  # delete limit for cooling, if no Cooling is available

        pyomo_dict = {"ElectricityPrice": self.creat_Dict(ElectricityPrice / 1_000),  # C/Wh
                      "FiT": self.creat_Dict(FeedinTariff / 1_000),  # C/Wh
                      "Q_Solar": self.creat_Dict(Q_sol),  # W
                      "T_outside": self.creat_Dict(self.Temperature["Temperature"].to_numpy()),  #°C
                      "SpaceHeatingHourlyCOP": self.creat_Dict(SpaceHeatingHourlyCOP),
                      "CoolingCOP": self.creat_Dict(ACHourlyCOP),
                      "BaseLoadProfile": self.creat_Dict(BaseLoadProfile),  # W
                      "HWPart1": self.creat_Dict(HotWaterProfile1),
                      "HWPart2": self.creat_Dict(HotWaterProfile2),
                      "HotWaterHourlyCOP": self.creat_Dict(HotWaterHourlyCOP),
                      "DayHour": self.creat_Dict(self.TimeStructure["ID_DayHour"]),
                      "DishWasherHours": self.creat_Dict(DishWasherHours),
                      "DishWasherPower": DishWasherPower,
                      "WashingMachineHours": self.creat_Dict(WashingMachineHours),
                      "WashingMachinePower": WashingMachinePower,
                      "DryerHours": self.creat_Dict(DryerHours),
                      "DryerPower": DryerPower,
                      "ChargeEfficiency": Household.Battery.ChargeEfficiency,
                      "DischargeEfficiency": Household.Battery.DischargeEfficiency,
                      "Am": Am,
                      "Atot": Atot,
                      "Qi": Qi,
                      "Htr_w": Htr_w,
                      "Htr_em": Htr_em,
                      "Htr_3": Htr_3,
                      "Htr_1": Htr_1,
                      "Htr_2": Htr_2,
                      "Hve": Hve,
                      "Htr_ms": Htr_ms,
                      "Htr_is": Htr_is,
                      "PHI_ia": PHI_ia,
                      "Cm": Cm,
                      "BuildingMassTemperatureStartValue": BuildingMassTemperatureStartValue
                      }

        return pyomo_dict, Household, Environment, HeatingTargetTemperature, CoolingTargetTemperature

        # ###############################################
        # Part III. Pyomo Optimization: cost minimization
        # ###############################################
    def create_abstract_model(self, input_parameters):

        m = pyo.AbstractModel()
        m.t = pyo.RangeSet(1, self.OptimizationHourHorizon)

        # -------------
        # 1. Parameters
        # -------------

        # price
        m.ElectricityPrice = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(ElectricityPrice / 1_000))  # C/Wh
        # Feed in Tariff of Photovoltaic
        m.FiT = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(FeedinTariff / 1_000))  # C/Wh
        # solar gains:
        m.Q_Solar = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(Q_sol))  # W
        # outside temperature
        m.T_outside = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(self.Temperature["Temperature"].to_numpy()))  # °C
        # COP of heatpump
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(SpaceHeatingHourlyCOP))
        # COP of cooling
        m.CoolingCOP = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(ACHourlyCOP))
        # electricity load profile
        m.BaseLoadProfile = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(BaseLoadProfile))  # W
        # PV profile
        m.PhotovoltaicProfile = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(PhotovoltaicProfile))  # W
        # HotWater
        m.HWPart1 = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(HotWaterProfile1))
        m.HWPart2 = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(HotWaterProfile2))
        m.HotWaterHourlyCOP = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(HotWaterHourlyCOP))

        # Smart Technologies
        m.DayHour = pyo.Param(m.t, mutable=True)#, initialize=self.creat_Dict(self.TimeStructure["ID_DayHour"]))

        # building data:
        m.Am = pyo.Param(mutable=True)
        m.Atot = pyo.Param(mutable=True)
        m.Qi = pyo.Param(mutable=True)
        m.Htr_w = pyo.Param(mutable=True)
        m.Htr_em = pyo.Param(mutable=True)
        m.Htr_3 = pyo.Param(mutable=True)
        m.Htr_1 = pyo.Param(mutable=True)
        m.Htr_2 = pyo.Param(mutable=True)
        m.Hve = pyo.Param(mutable=True)
        m.Htr_ms = pyo.Param(mutable=True)
        m.Htr_is = pyo.Param(mutable=True)
        m.PHI_ia = pyo.Param(mutable=True)
        m.Cm = pyo.Param(mutable=True)
        m.BuildingMassTemperatureStartValue = pyo.Param(mutable=True)

        # Tank data
        # Mass of water in tank
        m.M_WaterTank = pyo.Param(mutable=True)
        # Surface of Tank in m2
        m.A_SurfaceTank = pyo.Param(mutable=True)
        # insulation of tank, for calc of losses
        m.U_ValueTank = pyo.Param(mutable=True)
        m.T_TankStart = pyo.Param(mutable=True)
        # surrounding temp of tank
        m.T_TankSurrounding = pyo.Param(mutable=True)

        # Appliances
        m.DishWasherPower = pyo.Param(mutable=True)
        m.WashingMachinePower = pyo.Param(mutable=True)
        m.DryerPower = pyo.Param(mutable=True)

        # Battery data
        m.ChargeEfficiency = pyo.Param(mutable=True)
        m.DischargeEfficiency = pyo.Param(mutable=True)

        # ----------------------------
        # 2. Variables and constraints
        # ----------------------------

        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)#,
                                  #bounds=(0, float(Household.SpaceHeating.HeatPumpMaximalThermalPower)))  # W
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)#,
                                     #bounds=(0, float(Household.SpaceHeating.HeatingElementPower)))  # W


        m.E_tank = pyo.Var(m.t)#, within=pyo.NonNegativeReals, bounds=(CPWater * M_WaterTank * (273.15 + T_TankMin),
                                                                    # CPWater * M_WaterTank * (273.15 + T_TankMax)))  # Wh

        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)#,
                                  #bounds=(0, float(Household.SpaceHeating.MaximalPowerFloorHeating)))  # W

        # energy used for cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals)#,
                                  #bounds=(0, float(Household.SpaceCooling.SpaceCoolingPower)))  # W

        # def heating_cooling_bounds(m, t):
        #     return (HeatingTargetTemperature[t-1], CoolingTargetTemperature[t-1])
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals)#,  bounds=heating_cooling_bounds)

        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals)#,
                         #bounds=(0, float(Household.Building.MaximalBuildingMassTemperature)))

        # Grid, limit set by 21 kW
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))  # 380 * 32 * 1,72
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))

        m.Grid2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))


        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))

        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))
        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Building.MaximalGridPower * 1_000)))


        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Battery.Capacity * 1_000)))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Battery.MaxChargePower * 1_000)))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, Household.Battery.MaxDischargePower * 1_000))
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)#, bounds=(0, float(Household.Battery.MaxDischargePower * 1_000)))  # battery can only discharge to load

        # Smart Technologies
        m.DishWasher1 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher2 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasher3 = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasherStart = pyo.Var(m.t, within=pyo.Binary)
        m.DishWasherTheoreticalHours = pyo.Param(m.t, within=pyo.Binary, mutable=True, default=0) #,
                                                     #initialize=self.creat_Dict(DishWasherHours))


        m.WashingMachine1 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine2 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachine3 = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachineStart = pyo.Var(m.t, within=pyo.Binary)
        m.WashingMachineTheoreticalHours = pyo.Param(m.t, within=pyo.Binary, mutable=True, default=0)
                                                         #initialize=self.creat_Dict(WashingMachineHours))
        m.Dryer1 = pyo.Var(m.t, within=pyo.Binary)
        m.Dryer2 = pyo.Var(m.t, within=pyo.Binary)



        # -------------------------------------
        # 3. State transition: Electricity flow
        # -------------------------------------

        # (1) Overall energy balance
        def calc_ElectricalEnergyBalance(m, t):
            return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] == m.Feedin[t] + m.Load[t] + m.BatCharge[t]

        m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

        # (2) grid balance
        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t] #* Household.Battery.Grid2Battery

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (3) PV balance
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] == m.PhotovoltaicProfile[t]

        m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (4) Feen in
        def calc_SumOfFeedin(m, t):
            return m.Feedin[t] == m.PV2Grid[t]

        m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (5) load coverage
        def calc_SupplyOfLoads(m, t):
            return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # (6) Sum of Loads
        def calc_SumOfLoads(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.Q_TankHeating[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_RoomCooling[t] / m.CoolingCOP[t] \
                   + m.HWPart1[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.HWPart2[t] / m.HotWaterHourlyCOP[t] \
                   + (m.DishWasher1[t] + m.DishWasher2[t] + m.DishWasher3[t]) * m.DishWasherPower  \
                   + (m.WashingMachine1[t] + m.WashingMachine2[t] + m.WashingMachine3[t]) * m.WashingMachinePower  \
                   + (m.Dryer1[t] + m.Dryer2[t]) * m.DryerPower

        m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)

        # (7) Battery discharge
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == self.OptimizationHourHorizon:
                return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.BatDischarge[t] == m.Bat2Load[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8) Battery charge
        def calc_BatCharge(m, t):
                return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t] #* Household.Battery.Grid2Battery

        m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (9) Battery SOC
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.DischargeEfficiency))

        m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)


        # ---------------------------------------
        # 4. State transition: Smart Technologies
        # ---------------------------------------

        # DishWasher
        def calc_DishWasherHours2(m, t):
            if t >= 8759:
                return m.DishWasher2[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1:
                return m.DishWasher1[t] == m.DishWasher2[t + 1]
            return m.DishWasher2[t] == 0

        m.calc_DishWasherHours2 = pyo.Constraint(m.t, rule=calc_DishWasherHours2)

        def calc_DishWasherHours3(m, t):
            # if m.DishWasherTheoreticalHours[t] == 1 and DishWasherDuration == 3:
            #     return m.DishWasher1[t] == m.DishWasher3[t + 2]
            return m.DishWasher3[t] == 0

        m.calc_DishWasherHours3 = pyo.Constraint(m.t, rule=calc_DishWasherHours3)

        def calc_DishWasherStartTime(m, t):
            if m.t[t] == 1:
                return m.DishWasherStart[t] == 0
            elif m.DishWasherTheoreticalHours[t] == 1 and m.DayHour[t] == 21:
                return m.DishWasherStart[t] == m.DishWasherStart[t - 1] - 1

            return m.DishWasherStart[t] == m.DishWasherStart[t - 1] + \
                   m.DishWasher1[t] * m.DishWasherTheoreticalHours[t]

        m.calc_DishWasherStartTime = pyo.Constraint(m.t, rule=calc_DishWasherStartTime)

        # WashingMachine and Dryer
        def calc_WashingMachineHours2(m, t):
            if t >= 8759:
                return m.WashingMachine2[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1:
                return m.WashingMachine1[t] == m.WashingMachine2[t + 1]
            return m.WashingMachine2[t] == 0

        m.calc_WashingMachineHours2 = pyo.Constraint(m.t, rule=calc_WashingMachineHours2)

        def calc_WashingMachineHours3(m, t):
            # if m.WashingMachineTheoreticalHours[t] == 1 and WashingMachineDuration == 3:
            #     return m.WashingMachine1[t] == m.WashingMachine3[t + 2]
            return m.WashingMachine3[t] == 0

        m.calc_WashingMachineHours3 = pyo.Constraint(m.t, rule=calc_WashingMachineHours3)

        def calc_WashingMachineStartTime(m, t):
            if t == 1:
                return m.WashingMachineStart[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.DayHour[t] == 20:
                return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] - 1

            return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] + m.WashingMachine1[t] * \
                   m.WashingMachineTheoreticalHours[t]

        m.calc_WashingMachineStartTime = pyo.Constraint(m.t, rule=calc_WashingMachineStartTime)

        def calc_Dryer1(m, t):
            if t > 8755:
                return m.Dryer1[t] == 0
            return m.WashingMachine1[t] == m.Dryer1[t + 3]

        m.calc_Dryer1 = pyo.Constraint(m.t, rule=calc_Dryer1)

        def calc_Dryer2(m, t):
            if t > 8754:
                return m.Dryer2[t] == 0
            # if DryerDuration == 2:
            return m.WashingMachine1[t] == m.Dryer2[t + 4]
            # return m.Dryer2[t + 4] == 0

        m.calc_Dryer2 = pyo.Constraint(m.t, rule=calc_Dryer2)

        # --------------------------------------
        # 5. State transition: HeatPump and Tank
        # --------------------------------------
        def tank_energy_noTank(m, t):
            return m.Q_TankHeating[t] + m.Q_HeatingElement[t] == m.Q_RoomHeating[t]
        m.tank_energy_rule_noTank = pyo.Constraint(m.t, rule=tank_energy_noTank)


        def tank_energy(m, t):
            if t == 1:
                return m.E_tank[t] == self.CPWater * m.M_WaterTank * (273.15 + m.T_TankStart)
            else:
                return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] + m.Q_HeatingElement[t] \
                       - m.U_ValueTank * m.A_SurfaceTank * (
                                   (m.E_tank[t] / (m.M_WaterTank * self.CPWater)) - (m.T_TankSurrounding + 273.15))

        m.tank_energy_rule = pyo.Constraint(m.t, rule=tank_energy)

        # ----------------------------------------------------
        # 6. State transition: Building temperature (RC-Model)
        # ----------------------------------------------------

        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                # Equ. C.2
                PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * (PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                        ((m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.BuildingMassTemperatureStartValue *
                                     ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / (
                            (m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))

            else:
                # Equ. C.2
                PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * (PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                            ((m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / (
                            (m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.BuildingMassTemperatureStartValue) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                        T_sup + (m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve)) / (
                              m.Htr_ms + m.Htr_w + m.Htr_1)
                # Equ. C.11
                T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (m.Htr_is + m.Hve)
                #T_air = HeatingTargetTemperature[0]
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                            T_sup + (m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve)) / (
                                  m.Htr_ms + m.Htr_w + m.Htr_1)
                # Equ. C.11
                T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (m.Htr_is + m.Hve)
                # T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t]) / (Htr_is + Hve)
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        # ------------
        # 7. Objective
        # ------------

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
            return rule

        m.Objective = pyo.Objective(rule=minimize_cost)

        instance = m.create_instance(data=input_parameters)
        return instance

        # ---------
        # 8. Solver
        # ---------
    def solve_model(self):
        DC = DataCollector(self.Conn)
        Opt = pyo.SolverFactory("gurobi")
        Opt.options["TimeLimit"] = 50
        i = 0

        for household_RowID in range(0, 1):
            for environment_RowID in range(0, 1):
                input_parameters, Household, Environment, HeatingTargetTemperature, CoolingTargetTemperature = \
                    self.get_input_data(household_RowID, environment_RowID)
                if i == 0:
                    time0 = time.time()
                    instance = self.create_abstract_model(input_parameters)

                    # boundaries
                    for t in range(1, self.OptimizationHourHorizon + 1):
                        instance.Q_RoomCooling[t].setlb(0)
                        instance.Q_RoomCooling[t].setub(Household.SpaceCooling.SpaceCoolingPower)

                        instance.T_room[t].setlb(HeatingTargetTemperature[t - 1])
                        instance.T_room[t].setub(CoolingTargetTemperature[t - 1])

                        instance.Grid[t].setlb(0)
                        instance.Grid[t].setub(float(Household.Building.MaximalGridPower * 1_000))

                        instance.Grid2Load[t].setlb(0)
                        instance.Grid2Load[t].setub(float(Household.Building.MaximalGridPower * 1_000))

                        instance.Load[t].setub(float(Household.Building.MaximalGridPower * 1_000))
                        instance.Feedin[t].setub(float(Household.Building.MaximalGridPower * 1_000))

                    # special cases:
                    # Thermal storage
                    if Household.SpaceHeating.TankSize == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.E_tank[t].fix(0)
                            instance.Q_RoomHeating[t].setlb(0)
                            instance.Q_RoomHeating[t].setub(Household.SpaceHeating.HeatPumpMaximalThermalPower +
                                                            Household.SpaceHeating.HeatingElementPower)
                            instance.tank_energy_rule.deactivate()
                            instance.tank_energy_rule_noTank.activate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.E_tank[t].setlb(
                                self.CPWater * float(Household.SpaceHeating.TankSize) * (273.15 +
                                                                                         float(
                                                                                             Household.SpaceHeating.TankMinimalTemperature)))
                            instance.E_tank[t].setub(
                                self.CPWater * float(Household.SpaceHeating.TankSize) * (273.15 +
                                                                                         float(
                                                                                             Household.SpaceHeating.TankMaximalTemperature)))
                            instance.Q_RoomHeating[t].setlb(0)
                            instance.Q_RoomHeating[t].setub(Household.SpaceHeating.MaximalPowerFloorHeating)
                            instance.tank_energy_rule.activate()
                            instance.tank_energy_rule_noTank.deactivate()

                    # Battery
                    if Household.Battery.Capacity == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Grid2Bat[t].fix(0)

                            instance.Bat2Load[t].fix(0)
                            instance.BatSoC[t].fix(0)
                            instance.BatCharge[t].fix(0)
                            instance.BatDischarge[t].fix(0)

                        instance.calc_BatCharge.deactivate()
                        instance.calc_BatSoC.deactivate()
                        instance.calc_BatDischarge.deactivate()

                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Grid2Bat[t].setlb(0)
                            instance.Grid2Bat[t].setub(float(Household.Building.MaximalGridPower * 1_000))

                            instance.Bat2Load[t].setub(float(Household.Battery.MaxDischargePower * 1_000))
                            instance.BatSoC[t].setub(float(Household.Battery.Capacity * 1_000))
                            instance.BatCharge[t].setub(float(Household.Battery.MaxChargePower * 1_000))
                            instance.BatDischarge[t].setub(float(Household.Battery.MaxDischargePower * 1_000))

                            instance.calc_BatCharge.activate()
                            instance.calc_BatSoC.activate()
                            instance.calc_BatDischarge.activate()

                    # PV
                    if Household.PV.Power == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.PV2Load[t].fix(0)
                            instance.PV2Bat[t].fix(0)
                            instance.PV2Grid[t].fix(0)
                        instance.calc_UseOfPV.deactivate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.PV2Load[t].setub(float(Household.Building.MaximalGridPower * 1_000))
                            instance.PV2Bat[t].setub(float(Household.Building.MaximalGridPower * 1_000))
                            instance.PV2Grid[t].setub(float(Household.Building.MaximalGridPower * 1_000))
                        instance.calc_UseOfPV.activate()

                    # Dishwasher
                    if int(Household.ApplianceGroup.DishWasherShifting) == 1:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.DishWasherTheoreticalHours[t] = input_parameters["DishWasherHours"][t]
                            instance.calc_DishWasherHours2.activate()
                            instance.calc_DishWasherHours3.activate()
                            instance.calc_DishWasherStartTime.activate()

                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.DishWasher1[t] = input_parameters["DishWasherHours"][t]
                            instance.DishWasher2[t].fix(0)
                            instance.DishWasher3[t].fix(0)
                            instance.calc_DishWasherHours2.deactivate()
                            instance.calc_DishWasherHours3.deactivate()
                            instance.calc_DishWasherStartTime.deactivate()

                    # washingmachine
                    if int(Household.ApplianceGroup.WashingMachineShifting) == 1:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.WashingMachineTheoreticalHours[t] = input_parameters["WashingMachineHours"][t]
                            instance.calc_WashingMachineHours2.activate()
                            instance.calc_WashingMachineHours3.activate()
                            instance.calc_Dryer1.activate()
                            instance.calc_Dryer2.activate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.WashingMachine1[t] = input_parameters["WashingMachineHours"][t]
                            instance.WashingMachine2[t].fix(0)
                            instance.WashingMachine3[t].fix(0)
                            instance.Dryer1[t] = input_parameters["DryerHours"][t]
                            instance.Dryer2[t].fix(0)

                            instance.calc_WashingMachineHours2.deactivate()
                            instance.calc_WashingMachineHours3.deactivate()
                            instance.calc_WashingMachineStartTime.deactivate()

                            instance.calc_Dryer1.deactivate()
                            instance.calc_Dryer2.deactivate()

                    # dryer
                    if int(Household.ApplianceGroup.DryerAdoption) == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Dryer1[t].fix(0)
                            instance.Dryer2[t].fix(0)
                            instance.calc_Dryer1.deactivate()
                            instance.calc_Dryer2.deactivate()
                    else:
                        instance.calc_Dryer1.activate()
                        instance.calc_Dryer2.activate()

                else:
                    time0 = time.time()
                    # update the instance
                    for t in range(1, self.OptimizationHourHorizon+1):
                        instance.ElectricityPrice[t] = input_parameters["ElectricityPrice"][t]
                        instance.FiT[t] = input_parameters["FiT"][t]
                        instance.Q_solar[t] = input_parameters["Q_solar"][t]
                        instance.T_outside[t] = input_parameters["T_outside"][t]
                        instance.SpaceHeatingHourlyCOP[t] = input_parameters["SpaceHeatingHourlyCOP"][t]
                        instance.CoolingCOP[t] = input_parameters["CoolingCOP"][t]
                        instance.BaseLoadProfile[t] = input_parameters["BaseLoadProfile"][t]
                        instance.PhotovoltaicProfile[t] = input_parameters["PhotovoltaicProfile"][t]
                        instance.HWPart1[t] = input_parameters["HWPart1"][t]
                        instance.HWPart2[t] = input_parameters["HWPart2"][t]
                        instance.HotWaterHourlyCOP[t] = input_parameters["HotWaterHourlyCOP"][t]
                        instance.DayHour[t] = input_parameters["DayHour"][t]

                        instance.DishWasherHours[t] = input_parameters["DishWasherHours"][t]
                        instance.WashingMachineHours[t] = input_parameters["WashingMachineHours"][t]

                        # Boundaries:
                        instance.Q_RoomCooling[t].setlb(0)
                        instance.Q_RoomCooling[t].setub(Household.SpaceCooling.SpaceCoolingPower)

                        instance.T_room[t].setlb(HeatingTargetTemperature[t-1])
                        instance.T_room[t].setub(CoolingTargetTemperature[t-1])

                        instance.Grid[t].setlb(0)
                        instance.Grid[t].setub(float(Household.Building.MaximalGridPower*1_000))

                        instance.Grid2Load[t].setlb(0)
                        instance.Grid2Load[t].setub(float(Household.Building.MaximalGridPower*1_000))

                        instance.Load[t].setub(float(Household.Building.MaximalGridPower*1_000))
                        instance.Feedin[t].setub(float(Household.Building.MaximalGridPower*1_000))

                    # special cases:
                    # Thermal storage
                    if Household.SpaceHeating.TankSize == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.E_tank[t].fix(0)
                            instance.Q_RoomHeating[t].setlb(0)
                            instance.Q_RoomHeating[t].setub(Household.SpaceHeating.HeatPumpMaximalThermalPower +
                                                            Household.SpaceHeating.HeatingElementPower)
                            instance.tank_energy_rule.deactivate()
                            instance.tank_energy_rule_noTank.activate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.E_tank[t].setlb(self.CPWater*float(Household.SpaceHeating.TankSize)*(273.15 +
                                                                  float(Household.SpaceHeating.TankMinimalTemperature)))
                            instance.E_tank[t].setub(self.CPWater*float(Household.SpaceHeating.TankSize)*(273.15 +
                                                                  float(Household.SpaceHeating.TankMaximalTemperature)))
                            instance.Q_RoomHeating[t].setlb(0)
                            instance.Q_RoomHeating[t].setub(Household.SpaceHeating.MaximalPowerFloorHeating)
                            instance.tank_energy_rule.activate()
                            instance.tank_energy_rule_noTank.deactivate()

                    # Battery
                    if Household.Battery.Capacity == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Grid2Bat[t].fix(0)

                            instance.Bat2Load[t].fix(0)
                            instance.BatSoC[t].fix(0)
                            instance.BatCharge[t].fix(0)
                            instance.BatDischarge[t].fix(0)

                        instance.calc_BatCharge.deactivate()
                        instance.calc_BatSoC.deactivate()
                        instance.calc_BatDischarge.deactivate()

                        instance.calc_UseOfGrid_noBat.activate()
                        instance.calc_UseOfGrid.deactivate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Grid2Bat[t].setlb(0)
                            instance.Grid2Bat[t].setub(float(Household.Building.MaximalGridPower*1_000))

                            instance.Bat2Load[t].setub(float(Household.Battery.MaxDischargePower * 1_000))
                            instance.BatSoC[t].setub(float(Household.Battery.Capacity * 1_000))
                            instance.BatCharge[t].setub(float(Household.Battery.MaxChargePower * 1_000))
                            instance.BatDischarge[t].setub(float(Household.Battery.MaxDischargePower * 1_000))

                            instance.calc_BatCharge.activate()
                            instance.calc_BatSoC.activate()
                            instance.calc_BatDischarge.activate()

                            instance.calc_UseOfGrid_noBat.deactivate()
                            instance.calc_UseOfGrid.activate()

                    # PV
                    if Household.PV.Power == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.PV2Load[t].fix(0)
                            instance.PV2Bat[t].fix(0)
                            instance.PV2Grid[t].fix(0)
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.PV2Load[t].setub(float(Household.Building.MaximalGridPower*1_000))
                            instance.PV2Bat[t].setub(float(Household.Building.MaximalGridPower*1_000))
                            instance.PV2Grid[t].setub(float(Household.Building.MaximalGridPower*1_000))


                    # Dishwasher
                    if int(Household.ApplianceGroup.DishWasherShifting) == 1:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.DishWasherTheoreticalHours[t] = input_parameters["DishWasherHours"][t]
                            instance.calc_DishWasherHours2.activate()
                            instance.calc_DishWasherHours3.activate()
                            instance.calc_DishWasherStartTime.activate()

                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.DishWasher1[t] = input_parameters["DishWasherHours"][t]
                            instance.DishWasher2[t].fix(0)
                            instance.DishWasher3[t].fix(0)
                            instance.calc_DishWasherHours2.deactivate()
                            instance.calc_DishWasherHours3.deactivate()
                            instance.calc_DishWasherStartTime.deactivate()

                    # washingmachine
                    if int(Household.ApplianceGroup.WashingMachineShifting) == 1:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.WashingMachineTheoreticalHours[t] = input_parameters["WashingMachineHours"][t]
                            instance.calc_WashingMachineHours2.activate()
                            instance.calc_WashingMachineHours3.activate()
                            instance.calc_Dryer1.activate()
                            instance.calc_Dryer2.activate()
                    else:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.WashingMachine1[t] = input_parameters["WashingMachineHours"][t]
                            instance.WashingMachine2[t].fix(0)
                            instance.WashingMachine3[t].fix(0)
                            instance.Dryer1[t] = input_parameters["DryerHours"][t]
                            instance.Dryer2[t].fix(0)

                            instance.calc_WashingMachineHours2.deactivate()
                            instance.calc_WashingMachineHours3.deactivate()
                            instance.calc_WashingMachineStartTime.deactivate()

                            instance.calc_Dryer1.deactivate()
                            instance.calc_Dryer2.deactivate()

                    # dryer
                    if int(Household.ApplianceGroup.DryerAdoption) == 0:
                        for t in range(1, self.OptimizationHourHorizon + 1):
                            instance.Dryer1[t].fix(0)
                            instance.Dryer2[t].fix(0)
                            instance.calc_Dryer1.deactivate()
                            instance.calc_Dryer2.deactivate()
                    else:
                        instance.calc_Dryer1.activate()
                        instance.calc_Dryer2.activate()



                    # time independent parameters
                    instance.Am = input_parameters["Am"]
                    instance.Atot = input_parameters["Atot"]
                    instance.Qi = input_parameters["Qi"]
                    instance.Htr_w = input_parameters["Htr_w"]
                    instance.Htr_em = input_parameters["Htr_em"]
                    instance.Htr_3 = input_parameters["Htr_3"]
                    instance.Htr_1 = input_parameters["Htr_1"]
                    instance.Htr_2 = input_parameters["Htr_2"]
                    instance.Hve = input_parameters["Hve"]
                    instance.Htr_ms = input_parameters["Htr_ms"]
                    instance.Htr_is = input_parameters["Htr_is"]
                    instance.PHI_ia = input_parameters["PHI_ia"]
                    instance.Cm = input_parameters["Cm"]
                    instance.BuildingMassTemperatureStartValue = input_parameters["BuildingMassTemperatureStartValue"]

                    instance.ChargeEfficiency = input_parameters["ChargeEfficiency"]
                    instance.DischargeEfficiency = input_parameters["DischargeEfficiency"]




                result = Opt.solve(instance, tee=True)
                instance.display("./log.txt")
                DC.collect_OptimizationResult(Household, Environment, instance)
                i+=1
                time1 = time.time()
                print("time optimization:" + str(time1 - time0))


        # PyomoModelInstance = m.create_instance(report_timing=False)
        # Opt = pyo.SolverFactory("gurobi")#, solver_io='python') schneller ohne solver_io=python um 10 sekunden
        # Opt.options["TimeLimit"] = 360
        # results = Opt.solve(PyomoModelInstance, tee=False)
        # # save log file
        # PyomoModelInstance.display("./log.txt")
        # print('Total Operation Cost: ' + str(round(PyomoModelInstance.Objective(), 2)))
        #

        # return Household, Environment, PyomoModelInstance

    def run(self):
        DC = DataCollector(self.Conn)
        runs = len(self.ID_Household)
        for household_RowID in range(0, 1):
            for environment_RowID in range(0, 1):
                Household, Environment, PyomoModelInstance = self.run_Optimization(household_RowID, environment_RowID)
                DC.collect_OptimizationResult(Household, Environment, PyomoModelInstance)
        # DC.save_OptimizationResult()


if __name__ == "__main__":
    Conn = DB().create_Connection(CONS().RootDB)
    OperationOptimization(Conn).solve_model()

    # OperationOptimization(Conn).run()
