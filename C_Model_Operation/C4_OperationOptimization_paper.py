# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import pyomo.environ as pyo
import numpy as np
import sys as sys

from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C2_DataCollector import DataCollector
from C_Model_Operation.C2_DataCollectorNoSQlite import DataCollector_noSQlite
from B_Classes.B1_Household import Household
from A_Infrastructure.A1_CONS import CONS
import time
from pathos.multiprocessing import ProcessingPool, cpu_count
from functools import wraps
from pathos.pools import ParallelPool
import multiprocessing


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        print("function >>{}<< time for execution: {}".format(func.__name__, t_end - t_start))
        return result

    return wrapper


class DataSetUp:

    def __init__(self):

        self.Conn = DB().create_Connection(CONS().RootDB)
        self.ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.Conn)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.Temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)

        self.ElectricityPrice = DB().read_DataFrame(REG_Table().Gen_Sce_ElectricityProfile, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile,
                                                   self.Conn).BaseElectricityProfile.to_numpy()

        self.PhotovoltaicProfile = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.Radiation_SkyDirections = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections,
                                                           self.Conn)
        # self.HeatPump_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)
        self.AC_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_AC_HourlyCOP, self.Conn)

        self.TargetTemperature = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)

        # C_Water
        self.CPWater = 4200 / 3600

    def COP_HP(self, outside_temperature, supply_temperature, efficiency, source):
        """
        calculates the COP based on a performance factor and the massflow temperatures
        """
        if source == "Air_HP":
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - temp) for temp in
                            outside_temperature])
        elif source == "Water_HP":
            # for the ground source heat pump the temperature of the source is 10°C
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - 10) for temp in
                            outside_temperature])
        else:
            assert "only >>air<< and >>ground<< are valid arguments"
        return COP

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
        # print('Getting input data: \n ID_Household = ' + str(household_RowID + 1) + ", ID_Environment = " + str(
        #     environment_RowID + 1))
        Household = self.gen_Household(household_RowID)  # Gen_OBJ_ID_Household
        Environment = self.gen_Environment(environment_RowID)  # Gen_Sce_ID_Environment

        # ############################
        # PART I. Household definition
        # ############################

        # ------------------
        # 1. BaseLoadProfile
        # ------------------

        # BaseLoadProfile is 2376 kWh, SmartAppElectricityProfile is 1658 kWh
        BaseLoadProfile = self.BaseLoadProfile * 1_000  # W

        # ----------------------------
        # 3. Space heating and cooling
        # ----------------------------

        # (3.1) Tank

        # fixed starting values:
        T_TankStart = float(Household.SpaceHeating.TankMinimalTemperature)  # start temperature is min temp (empty tank)
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

        space_heating_supply_temperature = 35  # TODO implement this info in Household
        hot_water_supply_temperature = 55  # TODO implement this info in Household
        SpaceHeatingHourlyCOP = self.COP_HP(outside_temperature=self.Temperature["Temperature"].to_numpy(),
                                            supply_temperature=space_heating_supply_temperature,
                                            efficiency=Household.SpaceHeating.CarnotEfficiencyFactor,
                                            source=Household.SpaceHeating.Name_SpaceHeatingPumpType)

        # Selection of AC COP
        ACHourlyCOP = self.AC_HourlyCOP.ACHourlyCOP.to_numpy()

        # ------------
        # 4. Hot water
        # ------------

        # Hot Water Demand with Part 1 (COP SpaceHeating) and Part 2 (COP HotWater, lower)
        HotWaterProfile = self.HotWaterProfile.HotWater.to_numpy() * 1_000  # Wh

        # COP for HotWater generation
        HotWaterHourlyCOP = self.COP_HP(outside_temperature=self.Temperature["Temperature"].to_numpy(),
                                        supply_temperature=hot_water_supply_temperature,
                                        efficiency=Household.SpaceHeating.CarnotEfficiencyFactor,
                                        source=Household.SpaceHeating.Name_SpaceHeatingPumpType)

        # ----------------------
        # 5. PV and battery
        # ----------------------
        if Household.PV.ID_PVType in self.PhotovoltaicProfile["ID_PVType"].to_numpy():
            PhotovoltaicProfile = self.PhotovoltaicProfile.loc[
                                      (self.PhotovoltaicProfile['ID_PVType'] == Household.PV.ID_PVType) &
                                      (self.PhotovoltaicProfile['ID_Country'] == "AT")][
                                      'PVPower'].to_numpy() * 1_000  # W TODO nuts ID statt ID Country!
        else:
            PhotovoltaicProfile = np.full((8760,), 0)

        # ###############################
        # PART II. Environment definition
        # ###############################

        # --------------------
        # 1. Electricity price
        # --------------------
        ElectricityPrice = \
        self.ElectricityPrice.loc[self.ElectricityPrice['ID_PriceType'] == Environment["ID_ElectricityPriceType"]][
            "ElectricityPrice"].to_numpy() / 100
        # -----------------
        # 2. Feed-in tariff
        # -----------------
        FeedinTariff = self.FeedinTariff.loc[(self.FeedinTariff['ID_FeedinTariffType'] ==
                                              Environment["ID_FeedinTariffType"])][
            "HourlyFeedinTariff"].to_numpy()  # Cent/kWh

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
        elif Household.Name_AgeGroup == "Conventional":
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureYoungNightReduction']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureYoung']
        elif Household.Name_AgeGroup == 4:
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureOldNightReduction']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureOldNightReduction']
        elif Household.Name_AgeGroup == "SmartHome":
            HeatingTargetTemperature = self.TargetTemperature['HeatingTargetTemperatureSmartHome']
            CoolingTargetTemperature = self.TargetTemperature['CoolingTargetTemperatureSmartHome']

        BuildingMassTemperatureStartValue = 15  # °C

        if Household.SpaceCooling.AdoptionStatus == 0:
            CoolingTargetTemperature = np.full((8760,), 60)  # delete limit for cooling, if no Cooling is available

        pyomo_dict = {None: {"t": {None: np.arange(1, self.OptimizationHourHorizon + 1)},
                             "ElectricityPrice": self.creat_Dict(ElectricityPrice / 1_000),  # C/Wh
                             "FiT": self.creat_Dict(FeedinTariff / 1_000),  # C/Wh
                             "Q_Solar": self.creat_Dict(Q_sol),  # W
                             "PhotovoltaicProfile": self.creat_Dict(PhotovoltaicProfile),
                             "T_outside": self.creat_Dict(self.Temperature["Temperature"].to_numpy()),  # °C
                             "SpaceHeatingHourlyCOP": self.creat_Dict(SpaceHeatingHourlyCOP),
                             "CoolingCOP": self.creat_Dict(ACHourlyCOP),
                             "BaseLoadProfile": self.creat_Dict(BaseLoadProfile),  # W
                             "HotWater": self.creat_Dict(HotWaterProfile),
                             "HotWaterHourlyCOP": self.creat_Dict(HotWaterHourlyCOP),
                             "DayHour": self.creat_Dict(self.TimeStructure["ID_DayHour"]),
                             "ChargeEfficiency": {None: Household.Battery.ChargeEfficiency},
                             "DischargeEfficiency": {None: Household.Battery.DischargeEfficiency},
                             "Am": {None: Am},
                             "Atot": {None: Atot},
                             "Qi": {None: Qi},
                             "Htr_w": {None: Htr_w},
                             "Htr_em": {None: Htr_em},
                             "Htr_3": {None: Htr_3},
                             "Htr_1": {None: Htr_1},
                             "Htr_2": {None: Htr_2},
                             "Hve": {None: Hve},
                             "Htr_ms": {None: Htr_ms},
                             "Htr_is": {None: Htr_is},
                             "PHI_ia": {None: PHI_ia},
                             "Cm": {None: Cm},
                             "BuildingMassTemperatureStartValue": {None: BuildingMassTemperatureStartValue},
                             "T_TankStart": {None: T_TankStart},
                             "M_WaterTank": {None: M_WaterTank},
                             "U_ValueTank": {None: U_ValueTank},
                             "T_TankSurrounding": {None: T_TankSurrounding},
                             "A_SurfaceTank": {None: A_SurfaceTank}
                             }}

        # break up household and environment because they contain sqlite connection which can not be parallelized:
        household_dict = {
            "SpaceHeating_HeatPumpMaximalThermalPower": float(
                Household.SpaceHeating.HeatPumpMaximalThermalPower),
            "SpaceHeating_HeatingElementPower": float(Household.SpaceHeating.HeatingElementPower),
            "SpaceCooling_SpaceCoolingPower": float(Household.SpaceCooling.SpaceCoolingPower),
            "Building_MaximalGridPower": float(Household.Building.MaximalGridPower * 1_000),
            "SpaceHeating_TankSize": float(Household.SpaceHeating.TankSize),
            "SpaceHeating_TankMinimalTemperature": float(Household.SpaceHeating.TankMinimalTemperature),
            "SpaceHeating_TankMaximalTemperature": float(Household.SpaceHeating.TankMaximalTemperature),
            "Battery_MaxDischargePower": float(Household.Battery.MaxDischargePower * 1_000),
            "Battery_Capacity": float(Household.Battery.Capacity * 1_000),
            "Battery_MaxChargePower": float(Household.Battery.MaxChargePower * 1_000),
            "Battery_MaxDischargePower": float(Household.Battery.MaxDischargePower * 1_000),
            "ID": int(Household.ID),
            "ID_Building": int(Household.ID_Building),
            "Building_hwb_norm1": float(Household.Building.hwb_norm1),
            "SpaceCooling_AdoptionStatus": int(Household.SpaceCooling.AdoptionStatus),
            "PV_PVPower": float(Household.PV.PVPower),
            "Name_SpaceHeatingPumpType": Household.SpaceHeating.Name_SpaceHeatingPumpType,
            "ID_AgeGroup": int(Household.ID_AgeGroup)
        }

        environment_dict = {"ID": Environment["ID"],
                            "ID_ElectricityPriceType": Environment["ID_ElectricityPriceType"]
                            }
        all_input_parameters = {"input_parameters": pyomo_dict,
                                "Household": household_dict,
                                "Environment": environment_dict, \
                                "HeatingTargetTemperature": HeatingTargetTemperature, \
                                "CoolingTargetTemperature": CoolingTargetTemperature}

        return all_input_parameters


# ###############################################
# Part III. Pyomo Optimization: cost minimization
# ###############################################
@performance_counter
def create_abstract_model():
    CPWater = 4200 / 3600
    m = pyo.AbstractModel()
    m.t = pyo.Set()
    # -------------
    # 1. Parameters
    # -------------

    # price
    m.ElectricityPrice = pyo.Param(m.t, mutable=True)  # C/Wh
    # Feed in Tariff of Photovoltaic
    m.FiT = pyo.Param(m.t, mutable=True)  # C/Wh
    # solar gains:
    m.Q_Solar = pyo.Param(m.t, mutable=True)  # W
    # outside temperature
    m.T_outside = pyo.Param(m.t, mutable=True)  # °C
    # COP of heatpump
    m.SpaceHeatingHourlyCOP = pyo.Param(m.t, mutable=True)
    # COP of cooling
    m.CoolingCOP = pyo.Param(m.t, mutable=True)
    # electricity load profile
    m.BaseLoadProfile = pyo.Param(m.t, mutable=True)
    # PV profile
    m.PhotovoltaicProfile = pyo.Param(m.t, mutable=True)
    # HotWater
    m.HotWater = pyo.Param(m.t, mutable=True)
    m.HotWaterHourlyCOP = pyo.Param(m.t, mutable=True)

    # Smart Technologies
    m.DayHour = pyo.Param(m.t, mutable=True)

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

    # Battery data
    m.ChargeEfficiency = pyo.Param(mutable=True)
    m.DischargeEfficiency = pyo.Param(mutable=True)

    # ----------------------------
    # 2. Variables
    # ----------------------------
    # Variables SpaceHeating
    m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # Variable space cooling
    m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # Temperatures (room and thermal mass)
    m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # Grid variables
    m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Grid2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # PV variables
    m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # Electric Load and Electricity fed back to the grid
    m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # Battery variables
    m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # -------------------------------------
    # 3. Constraints:
    # -------------------------------------

    # (1) Overall energy balance
    def calc_ElectricalEnergyBalance(m, t):
        return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] == m.Feedin[t] + m.Load[t] + m.BatCharge[t]

    m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

    # (2) grid balance
    def calc_UseOfGrid(m, t):
        return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t]

    m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

    # (3) PV balance
    def calc_UseOfPV(m, t):
        return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] == m.PhotovoltaicProfile[t]

    m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

    # (4) Feed in
    def calc_SumOfFeedin(m, t):
        return m.Feedin[t] == m.PV2Grid[t]

    m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

    # (5) load coverage
    def calc_SupplyOfLoads(m, t):
        return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

    m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

    # (6) Sum of Loads
    def calc_SumOfLoads_with_cooling(m, t):
        return m.Load[t] == m.BaseLoadProfile[t] \
               + m.Q_TankHeating[t] / m.SpaceHeatingHourlyCOP[t] \
               + m.Q_HeatingElement[t] \
               + m.Q_RoomCooling[t] / m.CoolingCOP[t] \
               + m.HotWater[t] / m.HotWaterHourlyCOP[t]

    m.calc_SumOfLoads_with_cooling = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

    def calc_SumOfLoads_without_cooling(m, t):
        return m.Load[t] == m.BaseLoadProfile[t] \
               + m.Q_TankHeating[t] / m.SpaceHeatingHourlyCOP[t] \
               + m.Q_HeatingElement[t] \
               + m.HotWater[t] / m.HotWaterHourlyCOP[t]

    m.calc_SumOfLoads_without_cooling = pyo.Constraint(m.t, rule=calc_SumOfLoads_without_cooling)

    # (7) Battery discharge
    def calc_BatDischarge(m, t):
        if m.t[t] == 1:
            return m.BatDischarge[t] == 0  # start of simulation, battery is empty
        elif m.t[t] == m.t[-1]:
            return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
        else:
            return m.BatDischarge[t] == m.Bat2Load[t]

    m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

    # (8) Battery charge
    def calc_BatCharge(m, t):
        return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t]  # * Household.Battery.Grid2Battery

    m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

    # (9) Battery SOC
    def calc_BatSoC(m, t):
        if t == 1:
            return m.BatSoC[t] == 0  # start of simulation, battery is empty
        else:
            return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.ChargeEfficiency - \
                   m.BatDischarge[t] * (1 + (1 - m.DischargeEfficiency))

    m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)

    # --------------------------------------
    # 5. State transition: HeatPump and Tank
    # --------------------------------------

    def tank_energy(m, t):
        if t == 1:
            return m.E_tank[t] == CPWater * m.M_WaterTank * (273.15 + m.T_TankStart)
        else:
            return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] + m.Q_HeatingElement[t] \
                   - m.U_ValueTank * m.A_SurfaceTank * (
                           (m.E_tank[t] / (m.M_WaterTank * CPWater)) - (m.T_TankSurrounding + 273.15))

    m.tank_energy_rule = pyo.Constraint(m.t, rule=tank_energy)

    def tank_energy_noTank(m, t):
        return m.E_tank[t] == + m.Q_TankHeating[t] + m.Q_HeatingElement[t] - m.Q_RoomHeating[t]  # E_tank = 0

    m.tank_energy_rule_noTank = pyo.Constraint(m.t, rule=tank_energy_noTank)

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
            T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (
                    m.Htr_is + m.Hve)
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
            T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / (
                    m.Htr_is + m.Hve)
            return m.T_room[t] == T_air

    m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

    # ------------
    # 7. Objective
    # ------------

    def minimize_cost(m):
        rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
        return rule

    m.Objective = pyo.Objective(rule=minimize_cost)
    # return the abstract model
    return m


@performance_counter
def create_instance(input_parameters):
    # create the instance once:
    model = create_abstract_model()
    instance = model.create_instance(data=input_parameters)
    return instance


@performance_counter
def update_instance(total_input, instance):
    """
    Function takes the instance and updates its parameters as well as fixes various parameters to 0 if they are
    not used because there is no storage available for example. Solves the instance and returns the solved instance.
    """

    input_parameters = total_input["input_parameters"]
    household = total_input["Household"]
    Environment = total_input["Environment"]
    HeatingTargetTemperature = total_input["HeatingTargetTemperature"]
    CoolingTargetTemperature = total_input["CoolingTargetTemperature"]
    # print("Household ID: " + str(household["ID"]))
    # print("Environment ID: " + str(Environment["ID"]))
    CPWater = 4200 / 3600
    # update the instance
    for t in range(1, len(HeatingTargetTemperature) + 1):
        # time dependent parameters are updated:
        instance.ElectricityPrice[t] = input_parameters[None]["ElectricityPrice"][t]
        instance.FiT[t] = input_parameters[None]["FiT"][t]
        instance.Q_Solar[t] = input_parameters[None]["Q_Solar"][t]
        instance.PhotovoltaicProfile[t] = input_parameters[None]["PhotovoltaicProfile"][t]
        instance.T_outside[t] = input_parameters[None]["T_outside"][t]
        instance.SpaceHeatingHourlyCOP[t] = input_parameters[None]["SpaceHeatingHourlyCOP"][t]
        instance.CoolingCOP[t] = input_parameters[None]["CoolingCOP"][t]
        instance.BaseLoadProfile[t] = input_parameters[None]["BaseLoadProfile"][t]
        instance.HotWater[t] = input_parameters[None]["HotWater"][t]
        instance.HotWaterHourlyCOP[t] = input_parameters[None]["HotWaterHourlyCOP"][t]
        instance.DayHour[t] = input_parameters[None]["DayHour"][t]

        # Boundaries:
        # Heating
        instance.Q_TankHeating[t].setub(household["SpaceHeating_HeatPumpMaximalThermalPower"])
        instance.Q_HeatingElement[t].setub(household["SpaceHeating_HeatingElementPower"])
        # room heating is handled in if cases
        # Temperatures for RC model
        instance.T_room[t].setlb(HeatingTargetTemperature[t - 1])

        instance.Tm_t[t].setub(100)  # so it wont be infeasabe when no cooling
        # maximum Grid load
        instance.Grid[t].setub(household["Building_MaximalGridPower"])
        instance.Grid2Load[t].setub(household["Building_MaximalGridPower"])
        # maximum load of house and electricity fed back to the grid
        instance.Load[t].setub(household["Building_MaximalGridPower"])
        instance.Feedin[t].setub(household["Building_MaximalGridPower"])

    # special cases:
    # Room Cooling:
    if household["SpaceCooling_AdoptionStatus"] == 0:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            instance.Q_RoomCooling[t].setub(0)
            instance.T_room[t].setub(100)
        instance.calc_SumOfLoads_without_cooling.activate()
        instance.calc_SumOfLoads_with_cooling.deactivate()

    else:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            instance.Q_RoomCooling[t].fixed = False
            instance.Q_RoomCooling[t].setub(household["SpaceCooling_SpaceCoolingPower"])
            instance.T_room[t].setub(CoolingTargetTemperature[t - 1])
        instance.calc_SumOfLoads_without_cooling.deactivate()
        instance.calc_SumOfLoads_with_cooling.activate()
    # Thermal storage
    if household["SpaceHeating_TankSize"] == 0:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            instance.E_tank[t].fix(0)
            instance.Q_RoomHeating[t].setub(household["SpaceHeating_HeatPumpMaximalThermalPower"] +
                                            household["SpaceHeating_HeatingElementPower"])

        instance.tank_energy_rule_noTank.activate()
        instance.tank_energy_rule.deactivate()
    else:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            instance.E_tank[t].fixed = False
            instance.E_tank[t].setlb(CPWater * float(household["SpaceHeating_TankSize"]) * (273.15 +
                                                                                            float(household[
                                                                                                      "SpaceHeating_TankMinimalTemperature"])))
            instance.E_tank[t].setub(
                CPWater * household["SpaceHeating_TankSize"] * (273.15 +
                                                                household["SpaceHeating_TankMaximalTemperature"]))

            instance.Q_RoomHeating[t].setub(household["SpaceHeating_HeatPumpMaximalThermalPower"])  # W
            pass
        instance.tank_energy_rule_noTank.deactivate()
        instance.tank_energy_rule.activate()

    # Battery
    if household["Battery_Capacity"] == 0:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            # fix the parameters to 0
            instance.Grid2Bat[t].fix(0)
            instance.Bat2Load[t].fix(0)
            instance.BatSoC[t].fix(0)
            instance.BatCharge[t].fix(0)
            instance.BatDischarge[t].fix(0)
    else:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            # variables have to be unfixed in case they were fixed in a previous run
            instance.Grid2Bat[t].fixed = False
            instance.Bat2Load[t].fixed = False
            instance.BatSoC[t].fixed = False
            instance.BatCharge[t].fixed = False
            instance.BatDischarge[t].fixed = False
            # set upper bounds
            instance.Grid2Bat[t].setub(household["Building_MaximalGridPower"])

            instance.Bat2Load[t].setub(household["Battery_MaxDischargePower"])
            instance.BatSoC[t].setub(household["Battery_Capacity"])
            instance.BatCharge[t].setub(household["Battery_MaxChargePower"])
            instance.BatDischarge[t].setub(household["Battery_MaxDischargePower"])
            pass
    # PV
    if household["PV_PVPower"] == 0:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            instance.PV2Load[t].fix(0)
            instance.PV2Bat[t].fix(0)
            instance.PV2Grid[t].fix(0)

    else:
        for t in range(1, len(HeatingTargetTemperature) + 1):
            # variables have to be unfixed in case they were fixed in a previous run
            instance.PV2Load[t].fixed = False
            instance.PV2Bat[t].fixed = False
            instance.PV2Grid[t].fixed = False
            # set upper bounds
            instance.PV2Load[t].setub(household["Building_MaximalGridPower"])
            instance.PV2Bat[t].setub(household["Building_MaximalGridPower"])
            instance.PV2Grid[t].setub(household["Building_MaximalGridPower"])
            pass

    # update time independent parameters
    # building parameters:
    instance.Am = input_parameters[None]["Am"][None]
    instance.Atot = input_parameters[None]["Atot"][None]
    instance.Qi = input_parameters[None]["Qi"][None]
    instance.Htr_w = input_parameters[None]["Htr_w"][None]
    instance.Htr_em = input_parameters[None]["Htr_em"][None]
    instance.Htr_3 = input_parameters[None]["Htr_3"][None]
    instance.Htr_1 = input_parameters[None]["Htr_1"][None]
    instance.Htr_2 = input_parameters[None]["Htr_2"][None]
    instance.Hve = input_parameters[None]["Hve"][None]
    instance.Htr_ms = input_parameters[None]["Htr_ms"][None]
    instance.Htr_is = input_parameters[None]["Htr_is"][None]
    instance.PHI_ia = input_parameters[None]["PHI_ia"][None]
    instance.Cm = input_parameters[None]["Cm"][None]
    instance.BuildingMassTemperatureStartValue = input_parameters[None]["BuildingMassTemperatureStartValue"][None]
    # Battery parameters
    instance.ChargeEfficiency = input_parameters[None]["ChargeEfficiency"][None]
    instance.DischargeEfficiency = input_parameters[None]["DischargeEfficiency"][None]
    # Thermal storage parameters
    instance.T_TankStart = input_parameters[None]["T_TankStart"][None]
    instance.M_WaterTank = input_parameters[None]["M_WaterTank"][None]
    instance.U_ValueTank = input_parameters[None]["U_ValueTank"][None]
    instance.T_TankSurrounding = input_parameters[None]["T_TankSurrounding"][None]
    instance.A_SurfaceTank = input_parameters[None]["A_SurfaceTank"][None]

    return instance

    # def run(self):
    #     DC = DataCollector(self.Conn)
    #     runs = len(self.ID_Household)
    #     for household_RowID in range(0, 1):
    #         for environment_RowID in range(0, 1):
    #             Household, Environment, PyomoModelInstance = self.run_Optimization(household_RowID, environment_RowID)
    #             DC.collect_OptimizationResult(Household, Environment, PyomoModelInstance)
    #     # DC.save_OptimizationResult()


@performance_counter
def create_instances2solve(all_input_parameters, instance):
    ergebnisse2 = []
    for key in all_input_parameters.keys():
        instance2calculate = update_instance(all_input_parameters[key], instance)
        ergebnisse2.append(instance2calculate)
    return ergebnisse2


def run():
    # Conn = DB().create_Connection(CONS().RootDB)
    # DataGettingMachine = DataSetUp()
    # model input data
    DC = DataCollector()
    Opt = pyo.SolverFactory("gurobi")
    Opt.options["TimeLimit"] = 180
    runs = len(DataSetUp().ID_Household)
    input_data_initial = DataSetUp().get_input_data(0, 0)
    initial_parameters = input_data_initial["input_parameters"]
    pyomo_instance = create_instance(initial_parameters)
    for household_RowID in range(0, runs):
        for environment_RowID in range(0, 2):
            # print("Houshold ID: " + str(household_RowID + 1))
            # print("Environment ID: " + str(environment_RowID + 1))

            input_data = DataSetUp().get_input_data(household_RowID, environment_RowID)
            instance2solve = update_instance(input_data, pyomo_instance)

            # solve the instance:
            starttime = time.perf_counter()
            print("solving household {} in environment {}...".format(household_RowID, environment_RowID))
            result = Opt.solve(instance2solve, tee=False)
            print('Total Operation Cost: ' + str(round(instance2solve.Objective(), 2)))
            print("time for optimization: {}".format(time.perf_counter() - starttime))
            DC.collect_OptimizationResult(input_data["Household"], input_data["Environment"], instance2solve)


if __name__ == "__main__":
    # Conn = DB().create_Connection(CONS().RootDB)
    # OperationOptimization(Conn).solve_model()

    starttime2 = time.time()
    result_serial = run()
    endtime2 = time.time()
    print("time for execution serial:" + str(endtime2 - starttime2))


# import matplotlib.pyplot as plt
# plt.plot(np.arange(8760), (instance2solve.Q_RoomCooling.extract_values().values()))
# plt.show()