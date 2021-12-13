import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pyomo.environ as pyo
from _Philipp.Radiation import calculate_angels_of_sun
from B_Classes.B2_Building import HeatingCoolingNoSEMS


def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary

def showResults_noDR(Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR, plot_EPlus=False):
    if plot_EPlus==True:
        B1_EPlus = pd.read_csv("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Prosumager\\_Philipp\\inputdata\\Building1_EPlus.csv", sep=";")
        B1_cooling = B1_EPlus.loc[:, "DistrictCooling:Facility [J](Hourly)"].to_numpy()
        B1_heating = B1_EPlus.loc[:, "DistrictHeating:Facility [J](Hourly) "].to_numpy()
    red = '#F47070'
    dark_red = '#a10606'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    plt.style.use('ggplot')
    colors = [red, blue, green]
    colors2 = [dark_red, dark_blue, dark_green]
    x_achse = np.arange(len(Q_Heating_noDR))
    fig1 = plt.figure()
    for i in range(1):
        plt.plot(x_achse, Q_Heating_noDR[:, i] / 1_000, label="household " + str(i+1), color=colors[i], alpha=0.8)
    plt.plot(x_achse, B1_heating / 3_600 / 1_000, label="EPlus 1", color=colors2[0], linestyle=":")
    plt.legend()
    plt.title("Heating loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig1.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichHeizlast.svg")
    plt.show()

    fig2 = plt.figure()
    for i in [2, 1, 0]:
        plt.plot(x_achse, Q_Cooling_noDR[:, i] / 1_000, label="household " + str(i+1), color=colors[i], alpha=0.8)
    plt.legend()
    plt.title("Cooling loads")
    plt.ylabel("heating load in kWh")
    plt.xlabel("time in hours")
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\VergleichKuehlast.svg")
    plt.show()

    # room temperatures:
    fig3 = plt.figure()
    for i in range(3):
        plt.plot(x_achse, T_Room_noDR[:, i], label="indoor temperature " + str(i+1), color=colors[i], alpha=0.8)
        plt.plot(x_achse, T_thermalMass_noDR[:, i], label="thermal mass temperature " + str(i+1), color=colors2[i], alpha=0.8, linewidth=0.5, linestyle=":")
    plt.legend()
    plt.ylim(T_thermalMass_noDR.min(), T_thermalMass_noDR.max())
    plt.title("temperatures")
    plt.ylabel("temperature in °C")
    plt.xlabel("time in hours")
    fig3.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Temperaturen.svg")
    plt.show()



def create_pyomo_model(elec_price, tout, Qsol, Am, Atot, Cm, Hop, Htr_1, Htr_2, Htr_3, Htr_em, Htr_is, Htr_ms,
                       Htr_w, Hve, PHI_ia, Qi, COP):
    # model
    m = pyo.AbstractModel()

    # parameters
    m.time = pyo.RangeSet(len(elec_price))   # later just len(elec_price) for whole year
    # electricity price
    m.p = pyo.Param(m.time, initialize=create_dict(elec_price))
    # outside temperature
    m.T_outside = pyo.Param(m.time, initialize=create_dict(tout))
    # solar gains
    m.Q_sol = pyo.Param(m.time, initialize=create_dict(Qsol))


    # variables
    # energy used for heating
    m.Q_heating = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # energy used for cooling
    m.Q_cooling = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # real indoor temperature
    m.T_room = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(T_air_min, T_air_max))
    # thermal mass temperature
    m.Tm_t = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 50))


    # objective
    def minimize_cost(m):
        return sum((m.Q_heating[t] + m.Q_cooling[t]) / COP * m.ElectricityPrice[t] for t in m.time)
    m.OBJ = pyo.Objective(rule=minimize_cost)


    # constraints
    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            return m.Tm_t[t] == 15

        else:
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = m.T_outside[t]
            # Equ. C.5
            PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                    PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve) +
                                                               T_sup)) / Htr_2

            # Equ. C.4
            return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm/3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                                (Cm/3600) + 0.5 * (Htr_3 + Htr_em))
    m.thermal_mass_temperature_rule = pyo.Constraint(m.time, rule=thermal_mass_temperature_rc)


    def room_temperature_rc(m, t):
        if t == 1:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + thermal_mass_starting_temp) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
        else:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + m.Tm_t[t-1]) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
    m.room_temperature_rule = pyo.Constraint(m.time, rule=room_temperature_rc)


    instance = m.create_instance(report_timing=True)
    opt = pyo.SolverFactory("gurobi")
    results = opt.solve(instance)#, tee=True)
    print(results)
    instance.display("./log.txt")
    return instance, m


def showResults_Sprungantwort(Q_noDR, T_Room_noDR, T_thermalMass_noDR, heatORcool):
    red = '#F47070'
    dark_red = '#a10606'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    plt.style.use('ggplot')
    colors = [red, blue, green]
    colors2 = [dark_red, dark_blue, dark_green]
    x_achse = np.arange(len(Q_noDR))
    plt.style.use('ggplot')
    width = 1

    # room temperatures:
    fig3 = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for i in range(3):
        ax1.plot(x_achse, T_thermalMass_noDR[:, i], label="thermal mass temperature " + str(i+1), color=colors2[i], alpha=0.8, linewidth=width, linestyle=":")
        ax2.plot(x_achse, Q_noDR[:, i] / 1_000, label=heatORcool+" power " + str(i + 1), color=colors[i], alpha=0.8, linestyle="--", linewidth=width)
    ax1.plot(x_achse, T_Room_noDR[:, 0], label="indoor temperature ", color="black", alpha=0.8, linewidth=1.5)

    ax1.set_ylim(T_thermalMass_noDR.min(), T_thermalMass_noDR.max())
    ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    plt.title("thermal response 5R1C "+heatORcool)
    ax1.set_ylabel("temperature in °C")
    ax2.set_ylabel(heatORcool+" power in kW")
    plt.xlabel("time in hours")
    plt.tight_layout()
    fig3.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Sprungantwort_"+heatORcool+".svg")
    plt.show()

    # fig2 = plt.figure()
    # for i in range(3):
    #     plt.plot(T_thermalMass_noDR[:, i], Q_noDR[:, i] / 1_000, color=colors[i])
    # plt.show()


def get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor):
    # start time steps to calculate constant values
    runtime = 100
    # starte mit 100 stunde, dann checken ob sich konstantes Tm_t eingestellt hat
    Temperature_outside = np.array([T_outside] * runtime)
    Q_sol = np.array([[0] * 3] * runtime)
    # initial thermal mass temperature should be between T_min_indoor and T_max_indoor (22°C should always be ok)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = Buildings.ref_HeatingCooling(Temperature_outside,
                                                                                           Q_solar=Q_sol,
                                                                                           initial_thermal_mass_temp=22,
                                                                                           T_air_min=T_min_indoor,
                                                                                           T_air_max=T_max_indoor)
    # check if stationary condition has set in
    for i in range(T_thermalMass_noDR.shape[1]):
        while -1e-6 >= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i] or \
               1e-6 <= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i]:
            runtime += 100
            Temperature_outside = np.array([T_outside] * runtime)
            Q_sol = np.array([[0] * 3] * runtime)
            Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = Buildings.ref_HeatingCooling(Temperature_outside,
                                                                               Q_solar=Q_sol,
                                                                               initial_thermal_mass_temp=22,
                                                                               T_air_min=T_min_indoor,
                                                                               T_air_max=T_max_indoor)



    return Q_Heating_noDR[-1, :], Q_Cooling_noDR[-1, :], T_Room_noDR[-1, :], T_thermalMass_noDR[-1, :]


def calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, T_outside,
                                    T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=2):
    # calculate the thermal mass temperature when there is thermal equilibrium:
    Q_Heating_noDR_constant, Q_Cooling_noDR_constant, T_Room_noDR_constant, T_thermalMass_noDR_constant = \
        get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor)

    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_preheating)
    # no solar gains are considered
    Q_sol = np.array([[0] * 3] * hours_of_preheating)
    # calculate the thermal mass temperature after the time of preheating as well as heating/cooling power
    # this is done by raising the minimum indoor temperature for heating and lowering it for cooling by 2°C
    Q_PreHeating_noDR, Q_PreCooling_noDR, T_PreRoom_noDR, T_PrethermalMass_noDR = Buildings.ref_HeatingCooling(Temperature_outside,
                                                                                           Q_solar=Q_sol,
                                                                                           initial_thermal_mass_temp=T_thermalMass_noDR_constant,
                                                                                           T_air_min=T_min_indoor+T_offset_indoor,
                                                                                           T_air_max=T_max_indoor-T_offset_indoor)

    # now calculate the heating/cooling power with the old indoor temperature settings starting from the values
    # calculated in the preheating:
    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_shifting)
    # no solar gains are considered
    Q_sol = np.array([[0] * 3] * hours_of_shifting)
    Q_ReducedHeating_noDR, Q_ReducedCooling_noDR, T_ReducedRoom_noDR, T_ReducedthermalMass_noDR = Buildings.ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_PrethermalMass_noDR[-1, :],  # take the last one
        T_air_min=T_min_indoor,
        T_air_max=T_max_indoor)

    # calculate and plot the difference between steady state and shifting demand:
    # total thermal power during preheating when load is constant:
    Q_Heating_constant_total_preheating = Q_Heating_noDR_constant * hours_of_preheating
    Q_Heating_constant_total_shifting = Q_Heating_noDR_constant * hours_of_shifting
    Q_PreHeating_noDR_total = Q_PreHeating_noDR.sum(axis=0)
    Q_ReducedHeating_noDR_total = Q_ReducedHeating_noDR.sum(axis=0)

    ExcessHeatPreheat = Q_PreHeating_noDR_total - Q_Heating_constant_total_preheating
    SaveHeatShifting = Q_Heating_constant_total_shifting - Q_ReducedHeating_noDR_total

    fig0 = plt.figure()
    plt.bar(["preheating", "discharging"], [ExcessHeatPreheat[0], SaveHeatShifting[0]], color=["red", "green"])
    plt.show()

    # plot results for one building:
    x_achse = np.arange(hours_of_preheating+hours_of_shifting)
    Q_Heating_plot = np.append(Q_PreHeating_noDR[:, HouseNr-1], Q_ReducedHeating_noDR[:, HouseNr-1])
    T_thermalMass_plot = np.append(T_PrethermalMass_noDR[:, HouseNr-1], T_ReducedthermalMass_noDR[:, HouseNr-1])
    T_Room_plot = np.append(T_PreRoom_noDR[:, HouseNr-1], T_ReducedRoom_noDR[:, HouseNr-1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.bar(x_achse, Q_Heating_plot, label="heating power", color="red")
    ax1.axhline(Q_Heating_noDR_constant[0], xmin=0, xmax=1, label="constant heating power", color="black")
    ax1.vlines(x=0, ymin=Q_Heating_noDR_constant[0], ymax=Q_Heating_plot[0], color="red")
    ax1.axvline(x=hours_of_preheating-1, color="black", linestyle="--", linewidth=0.5)

    ax2.plot(x_achse, T_thermalMass_plot, label="thermal mass temp", color="blue")
    ax2.hlines(T_thermalMass_noDR_constant[0], xmin=0, xmax=hours_of_preheating+hours_of_shifting-1, color="purple", label="constant thermal mass temp")
    ax2.vlines(x=0, ymin=T_thermalMass_noDR_constant[0], ymax=T_thermalMass_plot[0], color="blue")
    ax2.axvline(x=hours_of_preheating-1, color="black", linestyle="--", linewidth=0.5)

    ax3.plot(x_achse, T_Room_plot, label="Room temp", color="green")
    ax3.hlines(T_Room_noDR_constant[0], xmin=0, xmax=hours_of_preheating+hours_of_shifting-1,  color="skyblue", label="constant room temp")
    ax3.vlines(x=0, ymin=T_Room_noDR_constant[0], ymax=T_Room_plot[0], color="green")
    ax3.axvline(x=hours_of_preheating-1, color="black", linestyle="--", linewidth=0.5)

    ax1.legend(loc="lower left")
    ax2.legend()
    ax3.legend()
    ax1.set_ylabel("heating power in W")
    ax2.set_ylabel("temperature in °C")
    ax3.set_ylabel("temperature in °C")
    ax3.set_xlabel("hours")
    ax1.set_title("Load shift at " + str(T_outside) + " °C, House Nr " + str(HouseNr))
    plt.tight_layout()
    fig.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Subplot_energy_shifted_" + str(HouseNr) + ".png")
    plt.show()



def Sprungantwort():
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    # # Heizen:
    # define building data
    buildingData = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")
    B = HeatingCoolingNoSEMS(buildingData)

    stunden_anzahl = 24
    outdoorTemp_heating = -5
    Temperature_outside = np.array([outdoorTemp_heating] * stunden_anzahl)
    Q_sol = np.array([[0] * 3] * stunden_anzahl)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                       Q_solar=Q_sol,
                                                                       initial_thermal_mass_temp=21,
                                                                       T_air_min=20,
                                                                       T_air_max=26)
    showResults_Sprungantwort(Q_Heating_noDR, T_Room_noDR, T_thermalMass_noDR, "heating")

    stunden_anzahl = 48
    outdoorTemp_cooling = 30
    Temperature_outside = np.array([outdoorTemp_cooling] * stunden_anzahl)
    Q_sol = np.array([[0] * 3] * stunden_anzahl)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                       Q_solar=Q_sol,
                                                                       initial_thermal_mass_temp=21,
                                                                       T_air_min=20,
                                                                       T_air_max=26)
    showResults_Sprungantwort(Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR, "cooling")


def Sprungantwort_Strompreis(base_input_path):
    # define building data
    buildingData = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")
    B = HeatingCoolingNoSEMS(buildingData)

    stunden_anzahl = 24
    outdoorTemp = -5
    Temperature_outside = np.array([outdoorTemp] * stunden_anzahl)
    Q_sol = np.array([[0] * 3] * stunden_anzahl)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                       Q_solar=Q_sol,
                                                                       initial_thermal_mass_temp=21,
                                                                       T_air_min=20,
                                                                       T_air_max=26)
    # Sprungantwort mit Strompreis
    strompreis = 0.2
    elec_price = np.array([strompreis] * stunden_anzahl)
    elec_price[:int(stunden_anzahl/2)] = 0

    # konditionierte Nutzfläche
    Af = buildingData.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af  # 7.2.2.2
    # Airtransfercoefficient
    Hve = buildingData.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = buildingData.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = buildingData.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = buildingData.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = buildingData.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = buildingData.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    # HWB_norm = data.loc[:, "hwb_norm"].to_numpy()
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
    # COP is only used in the objetive function (min Q_heating+Q_cooling * elecprice / COP)
    COP = 3
    instance = create_pyomo_model(elec_price, Temperature_outside, Q_sol, Am, Atot, Cm, Hop, Htr_1, Htr_2, Htr_3,
                                     Htr_em, Htr_is, Htr_ms, Htr_w, Hve, PHI_ia, Qi, COP)

    T_room = np.array(list(instance.T_room.extract_values().values()))
    Q_cooling = np.array(list(instance.Q_cooling.extract_values().values())) / 1_000  # kW
    Q_heating = np.array(list(instance.Q_heating.extract_values().values())) / 1_000  # kW
    T_mass_mean = np.array(list(instance.Tm_t.extract_values().values()))

    # # Kühlen:


def compare_solar_radation():
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    Temperature_data = pd.read_csv(base_input_path / "Frankfurt_WeatherData.csv", engine="python", sep=None, header=17)
    Temperature_outside = pd.to_numeric(Temperature_data.loc[:, "DryBulb {C}"].drop(0)).to_numpy()

    # location : Frankfurt
    latitude = 50.11
    longitude = 8.680965
    # year is 2010 even if the radiation is of a representative year..
    timearray = pd.date_range("01-01-2010 00:00:00", "01-01-2011 00:00:00", freq="H", closed="left",
                              tz=datetime.timezone.utc)
    GlobalHorizontalRadiation = Temperature_data.loc[:, "GloHorzRad {Wh/m2}"].drop(0).to_numpy().astype(np.float)
    E_diff = Temperature_data.loc[:, "DifHorzRad {Wh/m2}"].drop(0).to_numpy().astype(np.float)
    E_dir = GlobalHorizontalRadiation - E_diff

    azimuth_sun, altitude_sun, E_nord, E_sued, E_ost, E_west = \
        calculate_angels_of_sun(latitude, longitude, timearray, E_dir, E_diff)
    # define building data
    buildingData = pd.read_excel(base_input_path / "Building_data_for_EPlus.xlsx", engine="openpyxl")

    Q_sol_north = np.outer(E_nord, buildingData.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy())
    Q_sol_south = np.outer(E_sued, buildingData.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy())
    Q_sol_east_west = np.outer((E_ost + E_ost),
                               buildingData.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy())
    Q_sol = Q_sol_north + Q_sol_south + Q_sol_east_west

    # # # radiation EPlus:
    Q_sol_EPlus = pd.to_numeric(
        pd.read_csv(base_input_path / "solarRadiationEPlus.csv", sep=";").loc[:, "solar radiation"])
    Q_sol_EPlus = pd.concat([Q_sol_EPlus, Q_sol_EPlus, Q_sol_EPlus], axis=1).to_numpy()

    B = HeatingCoolingNoSEMS(buildingData)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = B.ref_HeatingCooling(Temperature_outside,
                                                                                           Q_solar=Q_sol,
                                                                                           initial_thermal_mass_temp=20,
                                                                                           T_air_min=20,
                                                                                           T_air_max=26)

    Q_Heating_noDR_Eplus, Q_Cooling_noDR_Eplus, T_Room_noDR_Eplus, T_thermalMass_noDR_Eplus = B.ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol_EPlus,
        initial_thermal_mass_temp=20,
        T_air_min=20,
        T_air_max=26)
    x_achse = np.arange(len(Q_Heating_noDR))
    fig2, ax1 = plt.subplots()
    ax1.plot(x_achse, E_dir+E_diff, label="rad gains pysolar")
    ax1.plot(x_achse, Q_sol_EPlus[:, 0], label="rad gains E+", alpha=0.5)
    ax1.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(x_achse, Q_sol[:, 0], label="rad gains pysolar")
    ax1.plot(x_achse, Q_sol_EPlus[:, 0], label="rad gains E+", alpha=0.5)
    ax1.legend()
    ax1.set_ylim(0, 10_000)

    ax2.bar(x_achse, Q_Heating_noDR[:, 0], label="heating pysolar")
    ax2.bar(x_achse, Q_Heating_noDR_Eplus[:, 0], label="heating E+ solar", alpha=0.5)
    ax2.legend()
    fig.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\SolarRadiationVergleichEPlus.svg")
    plt.show()
    print(f"totale solar Gewinne im Jahr Pysolar: {Q_sol[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale solar Gewinne im Jahr E+: {Q_sol_EPlus[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale Heizlast Pysolar: {Q_Heating_noDR[:, 0].sum() / 1_000_000:.4} MWh")
    print(f"totale Heitlast E+: {Q_Heating_noDR_Eplus[:, 0].sum() / 1_000_000:.4} MWh")

if __name__=="__main__":
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    # define building data
    buildingData = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")
    Buildings = HeatingCoolingNoSEMS(buildingData)

    # Sprungantwort()
    compare_solar_radation()

    # hours_of_preheating = 3
    # hours_of_shifting = 3
    # T_outside = 12
    # T_min_indoor = 20
    # T_max_indoor = 26
    # T_offset_indoor = 2
    # HouseNr = 3  # startet bei 1! nicht bei 0
    # calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, T_outside,
    #                              T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=T_offset_indoor)






