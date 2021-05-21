import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints
import random

from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from B_Classes.B1_Household import Household


# %%
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

    # %%
    def __init__(self, conn):
        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG().Gen_Sce_ID_Environment, self.Conn)

        # later stage: this is included in the scenario
        self.TimeStructure = DB().read_DataFrame(REG().Sce_ID_TimeStructure, self.Conn)
        self.LoadProfile = DB().read_DataFrame(REG().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.PhotovoltaicProfile = DB().read_DataFrame(REG().Sce_BasePhotovoltaicProfile, self.Conn)
        self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature_test, self.Conn)
        # self.Weather = DB().read_DataFrame(REG().Sce_Weather_Temperature, self.Conn)
        self.Radiation = DB().read_DataFrame(REG().Sce_Weather_Radiation, self.Conn)
        self.Building = DB().read_DataFrame(REG().ID_BuildingOption, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, self.Conn)
        # self.HeatPumpCOP = DB().read_DataFrame(REG().Sce_Technology_HeatPumpCOP, self.Conn)
        # you can import all the necessary tables into the memory here.
        # Then, it can be faster when we run the optimization problem for many "household - environment" combinations.

    # %%

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
        # Householdtype = Household.ID_HouseholdType
        # print(Householdtype)
        Environment = self.gen_Environment(environment_id)
        print(Environment.EVDailyElectricityConsumption)

        HoursOfSimulation = 8760

        # toDO:
        # error in Htr_w ... may false imported in DB

        def create_dict(liste):
            dictionary = {}
            for index, value in enumerate(liste, start=1):
                dictionary[index] = value
            return dictionary


        # fixed starting values:
        T_TankStart = 25
        # max tank temperature for boundary of energy
        T_TankMax = 50
        # sourounding temp of tank
        T_TankSourounding = 20
        # starting temperature of thermal mass 20°C
        thermal_mass_starting_temp = 20

        CWater = 4200 / 3600

        # Parameters of SpaceHeatingTank
        # Mass of water in tank
        M_WaterTank = Household.SpaceHeating.TankSize
        # Surface of Tank in m2
        A_SurfaceTank = Household.SpaceHeating.TankSurfaceArea
        # insulation of tank, for calc of losses
        U_ValueTank = Household.SpaceHeating.TankLoss

        # Building data for indoor temp calculation
        data_building = self.Building
        # data_building = Household.Building

        # konditionierte Nutzfläche
        Af = data_building.loc[:, "Af"].to_numpy()
        # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
        Atot = 4.5 * Af  # 7.2.2.2
        # Airtransfercoefficient
        Hve = data_building.loc[:, "Hve"].to_numpy()
        # Transmissioncoefficient wall
        Htr_w = data_building.loc[:, "Htr_w"].to_numpy()
        # Transmissioncoefficient opake Bauteile
        Hop = data_building.loc[:, "Hop"].to_numpy()
        # Speicherkapazität J/K
        Cm = data_building.loc[:, "CM_factor"].to_numpy() * Af
        # wirksame Massenbezogene Fläche [m^2]
        Am = data_building.loc[:, "Am_factor"].to_numpy() * Af
        # internal gains
        Qi = data_building.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
        At = 4.5  # 7.2.2.2
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

        # %%

        # Solar Gains, but to small calculated: in W/m²
        Q_sol = self.Radiation.loc[self.Radiation["ID_Country"] == 5].loc[:, "Radiation"].to_numpy()


        def random_price(size):
            return [random.randint(16, 40) for i in size * [None]]
        # generates a random price 16-40 ct, for simulation of RTP
        ElectricityPrice = random_price(HoursOfSimulation)
        # %%

        # model
        m = pyo.AbstractModel()

        # parameters
        m.t = pyo.RangeSet(1, HoursOfSimulation)
        # price
        m.p = pyo.Param(m.t, initialize=create_dict(ElectricityPrice))
        # solar gains:
        m.Q_solar = pyo.Param(m.t, initialize=create_dict(Q_sol))
        # outside temperature
        m.T_outside = pyo.Param(m.t, initialize=create_dict(self.Weather.Temperature))

        # Variables SpaceHeating
        # Energy of HeatPump
        m.Q_TankHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10_000))

        # Energy Tank
        m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(CWater * Household.SpaceHeating.TankSize * \
                                                                     (273.15 + T_TankStart), \
                                                                     CWater * Household.SpaceHeating.TankSize * \
                                                                     (273.15 + T_TankMax)))

        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10_000))
        # room temperature
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 28))
        # thermal mass temperature
        m.Tm_t = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 50))

        # objective
        def minimize_cost(m):
            rule = sum(m.Q_TankHeating[t] * m.p[t] for t in m.t)
            return rule

        m.OBJ = pyo.Objective(rule=minimize_cost)

        def tank_energy(m, t):
            if t == 1:
                return m.E_tank[t] == CWater * M_WaterTank * (273.15 + T_TankStart)
            else:
                return m.E_tank[t] == m.E_tank[t - 1] - m.Q_RoomHeating[t] + m.Q_TankHeating[t] \
                       - U_ValueTank * A_SurfaceTank * \
                       ((m.E_tank[t] + (M_WaterTank * CWater * T_TankSourounding)) / (M_WaterTank * CWater) \
                        - T_TankSourounding)

        # Calc_TempTank = (E_tank + m*c*t_sourround) / M_water*C_Water
        m.tank_energy = pyo.Constraint(m.t, rule=tank_energy)



        # 5R 1C model:
        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                        ((PHI_ia + m.Q_RoomHeating[t]) / Hve) + m.T_outside[t])) / \
                           Htr_2

                # Equ. C.4
                return m.Tm_t[t] == (thermal_mass_starting_temp * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

            else:
                # Equ. C.2
                PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                        PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_RoomHeating[t]) / Hve) + T_sup)) / \
                           Htr_2

                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                        (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.time, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + thermal_mass_starting_temp) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (
                        T_sup + (PHI_ia + m.Q_RoomHeating[t]) / Hve)) / \
                      (Htr_ms + Htr_w + Htr_1)
                # Equ. C.11
                T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_RoomHeating[t]) / (Htr_is + Hve)
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
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

        m.room_temperature_rule = pyo.Constraint(m.time, rule=room_temperature_rc)

        instance = m.create_instance(report_timing=True)
        opt = pyo.SolverFactory("glpk")
        results = opt.solve(instance, tee=True)
        print(results)

        # create plots to visualize resultsprice
        def show_results():
            Q_e = [instance.Q_TankHeating[t]() for t in m.t]
            price = [var2[i] for i in range(1, HoursOfSimulation + 1)]
            Q_a = [instance.Q_RoomHeating[t]() for t in m.t]

            T_room = [instance.T_room[t]() for t in m.t]
            T_tank = [instance.T_tank[t]() for t in m.t]
            T_a = [tout[i] for i in range(1, HoursOfSimulation + 1)]

            # indoor temperature is constant 20°C
            cost = [list(price)[i] * Q_e[i] for i in range(HoursOfSimulation)]
            total_cost = instance.OBJ()
            x_achse = np.arange(HoursOfSimulation)

            fig, (ax1, ax3) = plt.subplots(2, 1)
            ax2 = ax1.twinx()
            ax4 = ax3.twinx()
            #    ax5 = ax3.twinx()

            ax1.bar(x_achse, Q_e, label="HP power", color='blue')
            ax1.plot(x_achse, Q_a, label="Floot heating power", color="green", linewidth=0.75)
            ax2.plot(x_achse, price, color="red", label="price", linewidth=0.75, linestyle=':')

            ax1.set_ylabel("Energy Wh")
            ax2.set_ylabel("Price per kWh in Ct/€")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)

            ax3.plot(x_achse, T_room, label="Room temperature", color="orange", linewidth=0.5)
            ax4.plot(x_achse, T_tank, label="Tank temperature", color="red", linewidth=0.5)
            #    ax5.plot(x_achse, T_a, label='Outside temperature', color= 'black')

            ax3.set_ylabel("room temperature °C")
            ax4.set_ylabel("tank temperature °C")
            #    ax5.set_ylabel('outside temperature °C')
            ax3.yaxis.label.set_color('orange')
            ax4.yaxis.label.set_color('red')
            #    ax5.yaxis.label.set_color('black')
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax4.get_legend_handles_labels()
            ax4.legend(lines + lines2, labels + labels2, loc=0)
            plt.grid()

            ax1.set_title("Total costs un Ct/€ " + str(round(total_cost / 1000, 3)))
            plt.show()

        show_results()
        pass

    def run(self):
        for household_id in range(0, 1):
            for environment_id in range(0, 1):
                self.run_Optimization(household_id, environment_id)
        pass
