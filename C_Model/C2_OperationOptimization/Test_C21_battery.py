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

        # PV, battery and electricity load profile
        LoadProfile = self.LoadProfile.BaseElectricityProfile.to_numpy()
        PhotovoltaicBaseProfile = self.PhotovoltaicProfile.PhotovoltaicProfile
        PhotovoltaicProfileHH = PhotovoltaicBaseProfile * Household.PV.PVPower
        PhotovoltaicProfile = PhotovoltaicProfileHH.to_numpy()


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
        m.p = pyo.Param(m.t, initialize=create_dict(ElectricityPrice))
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




        # Variables SpaceHeating
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * M_WaterTank * (273.15 + T_TankMin), \
                                                                     CWater * M_WaterTank * (273.15 + T_TankMax)))

        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 17_000))
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 50))
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 60))

        # Variables PV and battery

        m.StorageFillLevel = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
        m.GridCover = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))
        m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))
        m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))

        # objective

        #m.Q_TankHeating[t] / m.COP_dynamic[t] * m.p[t]
        def minimize_cost(m):
            rule = sum(m.GridCover[t] * m.p[t] - m.PhotovoltaicFeedin[t] * 0.08 for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        # Battery
        def calc_StorageFillLevel(m, t):
            if t == 1:
                return m.StorageFillLevel[t] == 1
            else:
                return m.StorageFillLevel[t] == m.StorageFillLevel[t-1]*0.9999 + m.BatCharge[t] - m.BatDischarge[t]
        m.calc_StorageFillLevel = pyo.Constraint(m.t, rule=calc_StorageFillLevel)

        def calc_BatteryCharge(m, t):
            return m.BatCharge[t]*1.05 == m.PhotovoltaicProfile[t] - m.PhotovoltaicDirect[t] - m.PhotovoltaicFeedin[t]
        m.calc_BatteryCharge = pyo.Constraint(m.t, rule=calc_BatteryCharge)

        def calc_BatteryDischarge(m,t):
            return m.BatDischarge[t]*0.95 == m.LoadProfile[t] - m.GridCover[t] - m.PhotovoltaicDirect[t] + ((m.Q_TankHeating[t] / m.COP_dynamic[t])/1000)
        m.calc_BatterDischarge = pyo.Constraint(m.t, rule=calc_BatteryDischarge)



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
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_solar[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_solar[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_RoomHeating[t]) / Hve) + T_sup)) / \
                           Htr_2

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
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                        T_sup + (PHI_ia + m.Q_RoomHeating[t]) / Hve)) / \
                      (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t]) / (Htr_is + Hve)
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        instance = m.create_instance(report_timing=True)
        # opt = pyo.SolverFactory("glpk")
        opt = pyo.SolverFactory("gurobi")
        results = opt.solve(instance, tee=True)
        instance.display("./log.txt")
        print(results)
        # return relevant data
        return instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile

    def run(self):
        for household_id in range(515999, 516000):
            for environment_id in range(3, 4):
                instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile = self.run_Optimization(
                    household_id,
                    environment_id)
                return instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile


# create plots to visualize results price
def show_results(instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile):
    # calculation of JAZ
    EnergyThermal = np.array([instance.Q_TankHeating[t]() for t in range(1, HoursOfSimulation + 1)])
    EnergyElectric = []

    for i in range(1, len(list(EnergyThermal))):
        HP = EnergyThermal[i] / ListOfDynamicCOP[i]
        EnergyElectric.append(HP)

    JAZ = (np.sum(EnergyThermal)) / (np.sum(EnergyElectric))
    print('Test JAZ = ' + str(JAZ))

    # total values
    total_cost = instance.OBJ()
    total_energy = np.sum(EnergyThermal)

    # Visualization
    starttime = 0
    endtime = 8760

    Q_TankHeating = np.array([instance.Q_TankHeating[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    Q_RoomHeating = np.array([instance.Q_RoomHeating[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    T_room = np.array([instance.T_room[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    T_BuildingMass = np.array([instance.Tm_t[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    E_tank = np.array([instance.E_tank[t]() for t in range(1, HoursOfSimulation + 1)])
    GridCover =  np.array([instance.GridCover[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    StorageFillLevel = np.array([instance.StorageFillLevel[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    PhotovoltaicFeedin = np.array([instance.PhotovoltaicFeedin[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]
    PhotovoltaicDirect = np.array([instance.PhotovoltaicDirect[t]() for t in range(1, HoursOfSimulation + 1)])[starttime: endtime]

    # Generate Tank temperature
    TankTemperature = []
    for i in range(0, 8760):
        T_tank = (E_tank[i] / (M_WaterTank* CWater)) - 273.15
        TankTemperature.append(T_tank)
    T_tank = [TankTemperature[i] for i in range(0, HoursOfSimulation)]


    # Plot
    x_achse = np.arange(starttime, endtime)

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax1.plot(x_achse, Q_TankHeating, label="Q_TankHeating", color='blue', linewidth=0.15)
    ax1.plot(x_achse, Q_RoomHeating, label="Q_RoomHeating", color="green", linewidth=0.25)
    ax2.plot(x_achse, ElectricityPrice[starttime:endtime], color="black", label="Price", linewidth=0.50,
             linestyle=':')

    ax1.set_ylabel("Energy Wh")
    ax2.set_ylabel("Price per kWh in Ct/€")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="TempRoom", color="orange", linewidth=0.5)
    ax3.plot(x_achse, T_BuildingMass, label='TempMassBuild.', linewidth=0.15, color='black')
    ax4.plot(x_achse, T_tank[starttime: endtime], label="TempTank", color="red", linewidth=0.25)

    ax3.set_ylabel("Temp in °C", color='black')
    ax4.set_ylabel("Tank Temperature in °C", color='black')

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc=0)

    plt.grid()

    ax1.set_title("Total costs: " + str(round(total_cost, 2)) + ' € and total thermal energy: ' + str(
        round(total_energy / 1000, 2)) + ' kWh')

    plt.show()

    plt.figure()
    #plt.plot(StorageFillLevel, linewidth=0.15, label ='StorageFillLevel', color = 'gray')
    plt.plot(GridCover, linewidth=0.15, label ='GridCover', color = 'red')
    plt.plot(PhotovoltaicFeedin, linewidth=0.15, label ='PhotovoltaicFeedin', color = 'green')
    plt.plot(PhotovoltaicDirect, linewidth=0.4, label ='PhotovoltaicDirect', linestyle=':', color = 'black')
    #plt.plot(PhotovoltaicProfile[starttime:endtime], linewidth=0.15, label ='PhotovoltaicProfile', color = 'orange')
    plt.legend()
    plt.xlabel('Hour of year')
    plt.ylabel('Power in kW')

    plt.show()



if __name__ == "__main__":
    A = OperationOptimization(DB().create_Connection(CONS().RootDB))
    instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile = A.run()
    show_results(instance, ElectricityPrice, HoursOfSimulation, ListOfDynamicCOP, M_WaterTank, CWater, PhotovoltaicProfile)
