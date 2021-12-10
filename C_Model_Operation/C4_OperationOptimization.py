# -*- coding: utf-8 -*-
__author__ = 'Thomas, Philip, Songmin'

import pyomo.environ as pyo
import numpy as np
import sys as sys

from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C2_DataCollector import DataCollector
from B_Classes.B1_Household import Household



class OperationOptimization:

    def __init__(self, conn):

        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.Conn)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.Temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)
        self.Radiation = DB().read_DataFrame(REG_Table().Sce_Weather_Radiation, self.Conn)

        self.ElectricityPrice = DB().read_DataFrame(REG_Table().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.Demand_EV = DB().read_DataFrame(REG_Table().Sce_Demand_ElectricVehicleBehavior, self.Conn)

        self.DishWasherHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherHours, self.Conn)
        self.Sce_Demand_DishWasher = DB().read_DataFrame(REG_Table().Sce_Demand_DishWasher, self.Conn)
        self.WashingMachineHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineHours, self.Conn)
        self.Sce_Demand_WashingMachine = DB().read_DataFrame(REG_Table().Sce_Demand_WashingMachine, self.Conn)
        self.Sce_Demand_Dryer = DB().read_DataFrame(REG_Table().Sce_Demand_Dryer, self.Conn)

        self.CarAtHomeStatus = DB().read_DataFrame(REG_Table().Gen_Sce_CarAtHomeHours, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.Radiation_SkyDirections = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections, self.Conn)
        self.HeatPump_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)

        self.TargetTemperature = DB().read_DataFrame(REG_Table().Sce_ID_TargetTemperatureType, self.Conn)
        self.EnergyCost = DB().read_DataFrame(REG_Table().Sce_Price_EnergyCost, self.Conn)

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

    def run_Optimization(self, household_RowID, environment_RowID):

        print('Optimization: ID_Household = ' + str(household_RowID + 1) + ", ID_Environment = " + str(environment_RowID + 1))
        Household = self.gen_Household(household_RowID)
        Environment = self.gen_Environment(environment_RowID)

        # ############################
        # PART I. Household definition
        # ############################

        # ------------------
        # 1. BaseLoadProfile
        # ------------------

        # BaseLoadProfile is 2376 kWh, SmartAppElectricityProfile is 1658 kWh
        BaseLoadProfile = self.BaseLoadProfile.loc[(self.BaseLoadProfile['ID_BaseElectricityProfileType'] == Environment["ID_BaseElectricityProfileType"]) &
                                                   (self.BaseLoadProfile['ID_HouseholdType'] == Household.ID_HouseholdType)]['BaseElectricityProfile']

        # -------------------
        # 2. Smart appliances
        # -------------------

        # DishWasher
        DishWasherAdoption = Household.ApplianceGroup.DishWasherAdoption
        DishWasherTheoreticalHours = self.DishWasherHours["DishWasherHours"] * DishWasherAdoption
        DishWasherPower = Household.ApplianceGroup.DishWasherPower
        DishWasherSmartStatus = Household.ApplianceGroup.DishWasherShifting
        DishWasherDuration = int(self.Sce_Demand_DishWasher["DishWasherDuration"])
        DishWasherStartTime = int(self.Sce_Demand_DishWasher["DishWasherStartTime"])

        # WashingMachine and Dryer
        WashingMachineAdoption = Household.ApplianceGroup.WashingMachineAdoption
        WashingMachineTheoreticalHours = self.WashingMachineHours["WashingMachineHours"] * WashingMachineAdoption
        WashingMachineDuration = int(self.Sce_Demand_WashingMachine["WashingMachineDuration"])
        WashingMachineStartTime = int(self.Sce_Demand_WashingMachine["WashingMachineStartTime"])
        WashingMachinePower = Household.ApplianceGroup.WashingMachinePower
        WashingMachineSmartStatus = Household.ApplianceGroup.WashingMachineShifting
        DryerPower = Household.ApplianceGroup.DryerPower
        DryerDuration = int(self.Sce_Demand_Dryer["DryerDuration"])
        DryerAdoption = Household.ApplianceGroup.DryerAdoption

        # ----------------------------
        # 3. Space heating and cooling
        # ----------------------------

        # (3.1) Tank

        # fixed starting values:
        T_TankStart = Household.SpaceHeating.TankStartTemperature
        # min,max tank temperature for boundary of energy
        T_TankMax = Household.SpaceHeating.TankMaximalTemperature
        T_TankMin = Household.SpaceHeating.TankMinimalTemperature
        # sourounding temp of tank
        T_TankSourounding = Household.SpaceHeating.TankSurroundingTemperature
        # C_Water
        CWater = 4200 / 3600

        # Parameters of SpaceHeatingTank
        # Mass of water in tank
        M_WaterTank = Household.SpaceHeating.TankSize
        # Surface of Tank in m2
        A_SurfaceTank = Household.SpaceHeating.TankSurfaceArea
        # insulation of tank, for calc of losses
        U_ValueTank = Household.SpaceHeating.TankLoss

        # (3.2) RC-Model

        # Building Area
        Af = Household.Building.Af # konditionierte Nutzfläche (conditioned usable floor area)
        Atot = 4.5 * Af  # 7.2.2.2: Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen (Area of all surfaces facing the building zone)
        # Transmission Coefficient
        Hve = Household.Building.Hve # Air
        Htr_w = Household.Building.Htr_w # Wall
        Hop = Household.Building.Hop # opake Bauteile (opaque components)
        # Speicherkapazität (Storage capacity) [J/K]
        Cm = Household.Building.CM_factor * Af
        # wirksame Massenbezogene Fläche (Effective mass related area) [m^2]
        Am = float(Household.Building.Am_factor) * Af
        # internal gains
        Qi = Household.Building.spec_int_gains_cool_watt * Af
        # Coupling values
        his = np.float_(3.45)  # 7.2.2.2 - Kopplung Temp Luft mit Temp Surface Knoten s (Coupling Temp Air with Temp Surface node s)
        hms = np.float_(9.1)  # [W / m2K] from Equ.C.3 (from 12.2.2) - kopplung zwischen Masse und zentralen Knoten s (coupling between mass and central node s)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte (thermal coupling values) [W/K]
        Htr_is = his * Atot
        PHI_ia = 0.5 * Qi # Equ. C.1
        Htr_1 = 1 / (1 / Hve + 1 / Htr_is) # Equ. C.6
        Htr_2 = Htr_1 + Htr_w # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms) # Equ.C.8

        # (3.3) Solar gains
        # window areas in celestial directions
        Awindows_rad_east_west = Household.Building.average_effective_area_wind_west_east_red_cool
        Awindows_rad_south = Household.Building.average_effective_area_wind_south_red_cool
        Awindows_rad_north = Household.Building.average_effective_area_wind_north_red_cool

        # solar gains from different celestial directions
        Q_sol_north = np.outer(self.Radiation_SkyDirections.RadiationNorth, Awindows_rad_north)
        Q_sol_east = np.outer(self.Radiation_SkyDirections.RadiationEast, Awindows_rad_east_west /2)
        Q_sol_south = np.outer(self.Radiation_SkyDirections.RadiationSouth, Awindows_rad_south)
        Q_sol_west = np.outer(self.Radiation_SkyDirections.RadiationWest, Awindows_rad_east_west /2)

        Q_sol = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())

        # (3.4) Selection of heat pump COP
        SpaceHeatingHourlyCOP = self.HeatPump_HourlyCOP.loc[self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] ==Household.SpaceHeating.ID_SpaceHeatingBoilerType]['SpaceHeatingHourlyCOP']

        # ------------
        # 4. Hot water
        # ------------

        # Hot Water Demand with Part 1 (COP SpaceHeating) and Part 2 (COP HotWater, lower)
        HotWaterProfileSelect = self.HotWaterProfile.loc[(self.HotWaterProfile['ID_HotWaterProfileType'] == Environment["ID_HotWaterProfileType"]) &
                                                         (self.HotWaterProfile['ID_Country'] == Household.ID_Country) &
                                                         (self.HotWaterProfile['ID_HouseholdType'] == Household.ID_HouseholdType)]
        HotWaterProfile1 = HotWaterProfileSelect['HotWaterPart1']
        HotWaterProfile2 = HotWaterProfileSelect['HotWaterPart2']

        # Selection of COP
        HotWaterHourlyCOP = self.HeatPump_HourlyCOP.loc[self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] == Household.SpaceHeating.ID_SpaceHeatingBoilerType]['HotWaterHourlyCOP']

        # ----------------------
        # 5. PV and battery
        # ----------------------

        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.loc[(self.PhotovoltaicProfile['ID_PhotovoltaicProfileType'] == Environment["ID_PhotovoltaicProfileType"]) &
                                                               (self.PhotovoltaicProfile['ID_Country'] == Household.ID_Country)]['PhotovoltaicProfile']
        PhotovoltaicProfile = PhotovoltaicBaseProfile * Household.PV.PVPower

        # -----
        # 6. EV
        # -----

        # (6.1) Calculation of the hourly EV demand for the discharge of the EV, if not at home
        KilometerPerWorkday = int(self.Demand_EV.KilometerPerWorkday)
        ConsumptionPer100km = Household.ElectricVehicle.ConsumptionPer100km
        EV_DailyDemand = KilometerPerWorkday * ConsumptionPer100km / 100
        EV_LeaveHour = int(self.Demand_EV.EVLeaveHomeClock)
        EV_ArriveHour = int(self.Demand_EV.EVArriveHomeClock)
        EV_AwayHours = EV_ArriveHour - EV_LeaveHour
        EV_HourlyDemand = EV_DailyDemand / EV_AwayHours

        # (6.2) Check if the EV adopted, by checking the capacity of the EV
        # Case: BatteryCapacity = 0: EV not adopted - Petrol Car is used
        if Household.ElectricVehicle.BatterySize == 0:
            CarAtHomeStatus = self.creat_Dict([0] * self.OptimizationHourHorizon)
            V2B = 0  # Vif EV is not adopted, V2B have to be 0
            EV_HourlyDemand = 0
            # This value is hard coded for now, have to be chanced with DrivingProfiles
            PetrolCarYearlyDemand = KilometerPerWorkday * ConsumptionPer100km * 5 * 52 / 100  # 5 WorkdaysPerWeek
            PetrolCostPerLiter = float(self.EnergyCost.loc[self.EnergyCost['ID_EnergyCostType'] == Environment["ID_EnergyCostType"]].loc[:, 'PetrolCost'])
            PetrolCostPerYear = PetrolCarYearlyDemand * PetrolCostPerLiter
        # Case: BatteryCapacity > 0: EV is adopted
        else:
            CarAtHomeStatus = self.creat_Dict(self.CarAtHomeStatus["CarAtHomeHours"])
            V2B = Household.ElectricVehicle.V2B
            PetrolCostPerYear = 0

        # ###############################
        # PART II. Environment definition
        # ###############################

        # --------------------
        # 1. Electricity price
        # --------------------

        ElectricityPrice = self.ElectricityPrice.loc[(self.ElectricityPrice['ID_ElectricityPriceType'] == Environment["ID_ElectricityPriceType"]) &
                                                     (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyElectricityPrice']

        # -----------------
        # 2. Feed-in tariff
        # -----------------

        FeedinTariff = self.FeedinTariff.loc[(self.FeedinTariff['ID_FeedinTariffType'] == Environment["ID_FeedinTariffType"]) &
                                             (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyFeedinTariff']

        # ---------------------
        # 3. Target temperature
        # ---------------------

        ID_TargetTemperatureType = Environment["ID_TargetTemperatureType"]
        TargetTemperatureSelect = self.TargetTemperature.loc[self.TargetTemperature['ID_TargetTemperatureType'] == ID_TargetTemperatureType].iloc[0]
        if Household.ID_AgeGroup == 1:
            HeatingTargetTemperature = int(TargetTemperatureSelect['HeatingTargetTemperatureYoung'])
            CoolingTargetTemperature = int(TargetTemperatureSelect['CoolingTargetTemperatureYoung'])
        elif Household.ID_AgeGroup == 2:
            HeatingTargetTemperature = int(TargetTemperatureSelect['HeatingTargetTemperatureOld'])
            CoolingTargetTemperature = int(TargetTemperatureSelect['CoolingTargetTemperatureOld'])
        if Household.SpaceCooling.AdoptionStatus == 0:
            CoolingTargetTemperature = 60  # delete limit for cooling, if no Cooling is available
        if CoolingTargetTemperature < HeatingTargetTemperature:
            print('ERROR: CoolingTargetTemperature !>= HeatingTargetemperature! Change value in databank: Sce_ID_TargetTemperature')
            sys.exit()

        # ###############################################
        # Part III. Pyomo Optimization: cost minimization
        # ###############################################

        m = pyo.AbstractModel()
        m.t = pyo.RangeSet(1, self.OptimizationHourHorizon)

        # -------------
        # 1. Parameters
        # -------------

        # price
        m.ElectricityPrice = pyo.Param(m.t, initialize=self.creat_Dict(ElectricityPrice))
        # Feed in Tariff of Photovoltaic
        m.FiT = pyo.Param(m.t, initialize=self.creat_Dict(FeedinTariff))
        # solar gains:
        m.Q_Solar = pyo.Param(m.t, initialize=self.creat_Dict(Q_sol))
        # outside temperature
        m.T_outside = pyo.Param(m.t, initialize=self.creat_Dict(self.Temperature["Temperature"]))
        # COP of heatpump
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t, initialize=self.creat_Dict(SpaceHeatingHourlyCOP))
        # electricity load profile
        m.BaseLoadProfile = pyo.Param(m.t, initialize=self.creat_Dict(BaseLoadProfile))
        # PV profile
        m.PhotovoltaicProfile = pyo.Param(m.t, initialize=self.creat_Dict(PhotovoltaicProfile))
        # HotWater
        m.HWPart1 = pyo.Param(m.t, initialize=self.creat_Dict(HotWaterProfile1))
        m.HWPart2 = pyo.Param(m.t, initialize=self.creat_Dict(HotWaterProfile2))
        m.HotWaterHourlyCOP = pyo.Param(m.t, initialize=self.creat_Dict(HotWaterHourlyCOP))
        # CarAtHomeStatus
        m.CarAtHomeStatus = pyo.Param(m.t, initialize=CarAtHomeStatus)
        # Smart Technologies
        m.DayHour = pyo.Param(m.t, initialize=self.creat_Dict(self.TimeStructure["ID_DayHour"]))
        m.DishWasherTheoreticalHours = pyo.Param(m.t, within=pyo.Binary, initialize=self.creat_Dict(DishWasherTheoreticalHours))
        m.WashingMachineTheoreticalHours = pyo.Param(m.t, within=pyo.Binary, initialize=self.creat_Dict(WashingMachineTheoreticalHours))

        # ----------------------------
        # 2. Variables and constraints
        # ----------------------------

        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.SpaceHeating.HeatPumpMaximalThermalPower))
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.SpaceHeating.HeatingElementPower))
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * M_WaterTank * (273.15 + T_TankMin),
                                                                     CWater * M_WaterTank * (273.15 + T_TankMax)))
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.SpaceHeating.MaximalPowerFloorHeating))

        # energy used for cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.SpaceCooling.SpaceCoolingPower))  # 6kW thermal, 2 kW electrical
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(HeatingTargetTemperature, CoolingTargetTemperature))  # Change to TargetTemp
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalBuildingMassTemperature))

        # Grid, limit set by 21 kW
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))  # 380 * 32 * 1,72
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.Grid2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))

        # PV
        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))
        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Building.MaximalGridPower))

        # Battery
        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.Capacity))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxChargePower))
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.Battery.MaxDischargePower))

        # EV
        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatteryMaxChargePower))
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, Household.ElectricVehicle.BatteryMaxDischargePower))
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

        # -------------------------------------
        # 3. State transition: Electricity flow
        # -------------------------------------

        # (1) Overall energy balance
        def calc_ElectricalEnergyBalance(m, t):
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] == m.Feedin[t] + m.Load[t] + m.BatCharge[t]
            elif Household.Battery.Capacity == 0:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.EVDischarge[t] == m.Feedin[t] + m.Load[t] + m.EVCharge[t]
            else:
                return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] + m.EVDischarge[t] == m.Feedin[t] + m.Load[t] + m.BatCharge[t] + m.EVCharge[t]

        m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

        # (2)
        def calc_UseOfGrid(m, t):
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t]* Household.Battery.Grid2Battery
            else:
                return m.Grid[t] == m.Grid2Load[t] + m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.Grid2Bat[t] * Household.Battery.Grid2Battery

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (3)
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] == m.PhotovoltaicProfile[t]

        m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (4)
        def calc_SumOfFeedin(m, t):
            return m.Feedin[t] == m.PV2Grid[t]

        m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (5) EV is extra in EVSoC
        def calc_SupplyOfLoads(m, t):
            if Household.ElectricVehicle.BatterySize == 0 and Household.Battery.Capacity == 0:
                return m.Grid2Load[t] + m.PV2Load[t] == m.Load[t]
            if Household.ElectricVehicle.BatterySize == 0:
                return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]
            elif Household.Battery.Capacity == 0:
                return m.Grid2Load[t] + m.PV2Load[t] + m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B == m.Load[t]
            else:
                return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B == m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # (6)
        def calc_SumOfLoads(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                                + ((m.Q_TankHeating[t] / m.SpaceHeatingHourlyCOP[t]) / 1_000) \
                                + (m.Q_HeatingElement[t] / 1_000) \
                                + ((m.Q_RoomCooling[t] / Household.SpaceCooling.SpaceCoolingEfficiency / 1_000)) \
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
            elif m.t[t] == self.OptimizationHourHorizon:
                return m.BatDischarge[t] == 0
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarAtHomeStatus[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8)
        def calc_BatCharge(m, t):
            if Household.Battery.Capacity == 0:
                return m.BatCharge[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t]* Household.Battery.Grid2Battery
            else:
                return m.BatCharge[t] == m.PV2Bat[t] + m.EV2Bat[t] * V2B + m.Grid2Bat[t]* Household.Battery.Grid2Battery

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
                return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * m.CarAtHomeStatus[t]
            else:
                return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] + \
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
            elif m.t[t] == self.OptimizationHourHorizon:
                return m.EVDischarge[t] == 0
            elif Household.Battery.Capacity == 0:
                return m.EVDischarge[t] == m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.CarAtHomeStatus[t] * V2B + m.EV2Bat[t] * m.CarAtHomeStatus[t] * V2B

        m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (12)
        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0
            elif Household.ElectricVehicle.BatterySize == 0:
                return m.EVSoC[t] == 0
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + (m.EVCharge[t] * m.CarAtHomeStatus[t] * Household.ElectricVehicle.BatteryChargeEfficiency) \
                                     - (m.EVDischarge[t] * m.CarAtHomeStatus[t] * (1 + (1 - Household.ElectricVehicle.BatteryDischargeEfficiency))) \
                                     - (EV_HourlyDemand * (1 - m.CarAtHomeStatus[t]))

        m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)

        # ---------------------------------------
        # 4. State transition: Smart Technologies
        # ---------------------------------------

        # DishWasher
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
            elif m.DishWasherTheoreticalHours[t] == 1 and m.DayHour[t] == DishWasherStartTime and DishWasherSmartStatus == 0:
                return m.DishWasher1[t] == 1
            elif m.DishWasherTheoreticalHours[t] == 1 and m.DayHour[t] == 24:
                return m.DishWasherStart[t] == m.DishWasherStart[t - 1] - 1 * DishWasherSmartStatus
            return m.DishWasherStart[t] == m.DishWasherStart[t - 1] \
                   + m.DishWasher1[t] * m.DishWasherTheoreticalHours[t] * DishWasherSmartStatus

        m.calc_DishWasherStartTime = pyo.Constraint(m.t, rule=calc_DishWasherStartTime)

        # WashingMachine and Dryer
        def calc_WashingMachineHours2(m, t):
            if t >= 8759:
                return m.WashingMachine2[t] == 0
            elif m.WashingMachineTheoreticalHours[t] == 1 and (WashingMachineDuration == 2 or WashingMachineDuration == 3):
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
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.DayHour[t] == WashingMachineStartTime and WashingMachineSmartStatus == 0:
                return m.WashingMachine1[t] == 1
            elif m.WashingMachineTheoreticalHours[t] == 1 and m.DayHour[t] == 24:
                return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] - 1 * WashingMachineSmartStatus
            return m.WashingMachineStart[t] == m.WashingMachineStart[t - 1] + m.WashingMachine1[t] * m.WashingMachineTheoreticalHours[t] * WashingMachineSmartStatus

        m.calc_WashingMachineStartTime = pyo.Constraint(m.t, rule=calc_WashingMachineStartTime)

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

        # --------------------------------------
        # 5. State transition: HeatPump and Tank
        # --------------------------------------

        def tank_energy(m, t):
            if t == 1:
                return m.E_tank[t] == CWater * M_WaterTank * (273.15 + T_TankStart)
            else:
                return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] + m.Q_HeatingElement[t] \
                                      - U_ValueTank * A_SurfaceTank * ((m.E_tank[t] / (M_WaterTank * CWater)) - (T_TankSourounding+273.15))

        m.tank_energy_rule = pyo.Constraint(m.t, rule=tank_energy)

        # ----------------------------------------------------
        # 6. State transition: Building temperature (RC-Model)
        # ----------------------------------------------------

        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                return m.Tm_t[t] == Household.Building.BuildingMassTemperatureStartValue

            else:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_Solar[t])
                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / Hve) + T_sup)) / Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / ((Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                T_air = HeatingTargetTemperature
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (Htr_is + Hve)
                # T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t]) / (Htr_is + Hve)
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        # ------------
        # 7. Objective
        # ------------

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t) + PetrolCostPerYear
            return rule
        m.Objective = pyo.Objective(rule=minimize_cost)

        # ---------
        # 8. Solver
        # ---------

        PyomoModelInstance = m.create_instance(report_timing=False)
        Opt = pyo.SolverFactory("gurobi")
        results = Opt.solve(PyomoModelInstance, tee=False)
        print('Total Operation Cost: ' + str(round(PyomoModelInstance.Objective(), 2)))

        return Household, Environment, PyomoModelInstance

    def run(self):
        DC = DataCollector(self.Conn)
        for household_RowID in range(0, 1):
            for environment_RowID in range(0,1):
                Household, Environment, PyomoModelInstance = self.run_Optimization(household_RowID, environment_RowID)
                DC.collect_OptimizationResult(Household, Environment, PyomoModelInstance)
        DC.save_OptimizationResult()




