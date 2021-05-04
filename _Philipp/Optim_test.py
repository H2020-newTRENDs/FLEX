import pyomo.environ as pyo
import numpy as np
from pathlib import Path
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints

def get_invert_data():
    project_directory_path = Path(__file__).parent.resolve()
    base_results_path = project_directory_path / "inputdata"
    scn = r'_scen_aut_a00_WAM_plus_v6_ren_mue'

    results_path = base_results_path / scn
    run_number_str = "001"
    results_path_rcm = results_path / "Dynamic_Calc_Input_Data"
    results_path_temperatures = results_path / "Annual_Climate_Data"
    results_path_FWKWK_data = results_path / "FWKWK_REGIONAL_DATA"

    # check if paths exist:
    print("rcm path exists: " + str(results_path_rcm.exists()))
    print("temperatures path exists: " + str(results_path_temperatures.exists()))
    print("FWKWK path exists: " + str(results_path_FWKWK_data.exists()))

    # load outside temperature data:
    temp_data = pd.read_excel(results_path_temperatures / "Input_Weather2015_AT.xlsx", engine="openpyxl")
    temp_outside = temp_data.loc[:, "Temperature"].to_numpy()

    # load building stock data exportet by invert run:
    YEAR = 2020
    datei = str(run_number_str) + "__dynamic_calc_data_bc_" + str(YEAR) + ".csv"
    data = pd.read_csv(results_path_rcm / datei).drop(0, axis=0)


    # data = pd.DataFrame(data=data_np["arr_0"], columns=data_np["arr_1"][0])
    # data.columns = data.columns.str.decode("utf-8")
    # data.columns = data.columns.str.replace(" ", "")

    datei = run_number_str + '__climate_data_solar_rad_' + str(YEAR) + ".csv"
    sol_rad = pd.read_csv(results_path_rcm / datei)
    sol_rad = sol_rad.drop(columns=["climate_region_index"]).to_numpy()
    # climate region index - 1 because python starts to count at zero:
    climate_region_index = data.loc[:, "climate_region_index"].to_numpy().astype(int) - 1
    unique_climate_region_index = np.unique(climate_region_index).astype(int)
    sol_rad_north = np.empty((1, 12), float)
    sol_rad_east_west = np.empty((1, 12))
    sol_rad_south = np.empty((1, 12))
    for climate_region in unique_climate_region_index:
        anzahl = len(np.where(climate_region_index == climate_region)[0])
        sol_rad_north = np.append(sol_rad_north,
                                  np.tile(sol_rad[int(climate_region), np.arange(12)], (anzahl, 1)),
                                  axis=0)  # weil von 0 gezählt
        sol_rad_east_west = np.append(sol_rad_east_west,
                                      np.tile(sol_rad[int(climate_region), np.arange(12, 24)], (anzahl, 1)), axis=0)
        sol_rad_south = np.append(sol_rad_south,
                                  np.tile(sol_rad[int(climate_region), np.arange(24, 36)], (anzahl, 1)), axis=0)

    # delete first row in solar radiation as its just zeros
    sol_rad_north = np.delete(sol_rad_north, 0, axis=0)
    sol_rad_east_west = np.delete(sol_rad_east_west, 0, axis=0)
    sol_rad_south = np.delete(sol_rad_south, 0, axis=0)

    # konditionierte Nutzfläche
    Af = data.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af  # 7.2.2.2
    # Airtransfercoefficient
    Hve = data.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = data.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = data.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = data.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = data.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = data.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    # hours per day
    h = np.arange(1, 25)
    # sol_hx = min(14, max(0, h + 0.5 - 6.5))
    # Verteilung der Sonneneinstrahlung auf die einzelnen Stunden mit fixem Profil:
    # Minimalwert = 0 und Maximalwert = 14
    sol_hx = []
    for i in h:
        if h[i - 1] + 0.5 - 6.5 < 0:
            sol_hx.append(0)
        elif h[i - 1] + 0.5 - 6.5 > 14:
            sol_hx.append(14)
        else:
            sol_hx.append(h[i - 1] + 0.5 - 6.5)

    # sol_rad_norm = max(0, np.sin(3.1415 * sol_hx / 13)) / 8.2360
    # Sinusprofil für die Sonneneinstrahlung:
    sol_rad_norm = []
    for i in sol_hx:
        if np.sin(3.1415 * i / 13) / 8.2360 < 0:
            sol_rad_norm.append(0)
        else:
            sol_rad_norm.append(np.sin(3.1415 * i / 13) / 8.2360)

    sol_rad_norm = np.array(sol_rad_norm)

    # solar radiation: Norm profile multiplied by typical radiation of month times 24 hours for one day
    # TODO create 8760 vector not only for first 24 hours
    sol_rad_n = sol_rad_norm * sol_rad_north[0, 0] * 24
    sol_rad_ea = sol_rad_norm * sol_rad_east_west[0, 0] * 24
    sol_rad_s = sol_rad_norm * sol_rad_south[0, 0] * 24
    # solar gains through windows
    Qsol = Awindows_rad_north[0] * sol_rad_n + Awindows_rad_east_west[0] * sol_rad_ea + Awindows_rad_south[0] * sol_rad_s

    # electricity price for 24 hours:
    elec_price = pd.read_excel(base_results_path / "Elec_price_per_hour.xlsx", engine="openpyxl")
    elec_price = elec_price.loc[:, "Price (€/MWh)"].dropna().to_numpy()
    return Atot[0], Hve[0], Htr_w[0], Hop[0], Cm[0], Am[0], Qi[0], \
           temp_outside[0:24], Qsol, elec_price


# testing pyomo very simple example for 6 hours: (6 steps)
# Hot water tank with input, output, losses depending on inside temperature
# surrounding temperature T_a = 20°C constant
# losses Q_loss = 0.003 * (T_inside - T_a) kWh
# water mass in the tank m = 1000kg, c_p water = 4.2 kJ/kgK
# Useful energy in the Tank Q = m * cp * (T_inside - T_a)
# heat demand supplied by the tank is equal to heat transfered to the room
# input into the tank Q_e = ??? (variable to be determined)
# price for buying energy every hour p = [1, 6, 8, 8, 2] [price units / kWh]
# Goal: Minimize operation cost of hot water tank!
# room temperature is function of thermal capacity, gains and losses
# room temperature has to stay above 20°C
# outside temperature is [5, 4, 0, -1, -2, -2] °C


def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary

Atot, Hve, Htr_w, Hop, Cm, Am, Qi, temp_outside, Q_solar, price = get_invert_data()

# fixed starting values:
tank_starting_temp = 50
indoor_starting_temp = 22

# constants:
# water mass in storage
m_water = 10_000  # l
# thermal capacity water
cp_water = 4.2  # kJ/kgK
# constant surrounding temp for water tank
T_a = 20

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

# given values changing over time:
# electricity price
elec_price = create_dict(price)
# outside temperature
tout = create_dict(temp_outside)
# solar gains
Qsol = create_dict(Q_solar)


# TODO Dimensionskontrolle!
# model
m = pyo.AbstractModel()

# parameters
m.time = pyo.RangeSet(24)
# electricity price
m.p = pyo.Param(m.time, initialize=elec_price)
# outside temperature
m.T_outside = pyo.Param(m.time, initialize=tout)
# solar gains
m.Q_sol = pyo.Param(m.time, initialize=Qsol)

# variables
# electricity into boiler
# m.Q_e = pyo.Var(m.t, within=pyo.NonNegativeReals)
# temperature inside hot water tank
# m.T_tank = pyo.Var(m.t, within=pyo.NonNegativeReals)
# energy used for heating
m.Q_heating = pyo.Var(m.time, within=pyo.NonNegativeReals)
# real indoor temperature
m.T_room = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(18, 28))
# # mean indoor temperature
m.Tm_t = pyo.Var(m.time, within=pyo.NonNegativeReals)


# objective
def minimize_cost(m):
    rule = sum(m.Q_heating[t] * m.p[t] for t in m.time)
    return rule
m.OBJ = pyo.Objective(rule=minimize_cost)


# constraints
def mean_room_temperature_rc(m, t):
    if t == 1:
        # Equ. C.2
        PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

        # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
        # Equ. C.5
        PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t]) / Hve) + m.T_outside[t])) / \
                   Htr_2

        # Equ. C.4
        return m.Tm_t[t] == (indoor_starting_temp * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
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
                PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t]) / Hve) + T_sup)) / \
                   Htr_2

        # Equ. C.4
        return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                    (Cm / 3600) + 0.5 * (Htr_3 + Htr_em))
m.mean_room_temperature = pyo.Constraint(m.time, rule=mean_room_temperature_rc)


def room_temperature_rc(m, t):
    if t == 1:
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.9
        T_m = (m.Tm_t[t] + indoor_starting_temp) / 2
        T_sup = m.T_outside[t]
        # Euq. C.10
        T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (PHI_ia + m.Q_heating[t]) / Hve)) / (
                Htr_ms + Htr_w + Htr_1)
        # Equ. C.11
        T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t]) / (Htr_is + Hve)
        # Equ. C.12
        T_op = 0.3 * T_air + 0.7 * T_s
        # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
        return m.T_room[t] == T_op
    else:
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
        # Equ. C.9
        T_m = (m.Tm_t[t] + m.Tm_t[t-1]) / 2
        T_sup = m.T_outside[t]
        # Euq. C.10
        T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (PHI_ia + m.Q_heating[t]) / Hve)) / (
                Htr_ms + Htr_w + Htr_1)
        # Equ. C.11
        T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t]) / (Htr_is + Hve)
        # Equ. C.12
        T_op = 0.3 * T_air + 0.7 * T_s
        # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
        return m.T_room[t] == T_op
m.room_temperature = pyo.Constraint(m.time, rule=room_temperature_rc)


# def maximum_temperature_tank(m, t):
#     return m.T_tank[t] <= 100
# m.maximum_temperature_tank = pyo.Constraint(m.t, rule=maximum_temperature_tank)


# def minimum_temperature_tank(m, t):
#     return m.T_tank[t] >= 20
# m.minimum_temperature_tank = pyo.Constraint(m.t, rule=minimum_temperature_tank)


# def maximum_temperature_room(m, t):
#     return m.T_room[t] <= 28
# m.maximum_temperature_raum = pyo.Constraint(m.t, rule=maximum_temperature_room)
#
#
# def minimum_temperature_room(m, t):
#     return m.T_room[t] >= 20
# m.minimum_temperature_raum = pyo.Constraint(m.t, rule=minimum_temperature_room)


# def tank_temperatur(m, t):
#     if t == 1:
#         return m.T_tank[t] == tank_starting_temp - m.Q_heating[t] / m_water / cp_water * 3600 + \
#                m.Q_e[t] / m_water / cp_water * 3600 - 0.003 * (tank_starting_temp - T_a)
#     else:
#         return m.T_tank[t] == m.T_tank[t - 1] - m.Q_heating[t] / m_water / cp_water * 3600 + \
#                m.Q_e[t] / m_water / cp_water * 3600 - 0.003 * (m.T_tank[t] - T_a)


# m.tank_temperatur = pyo.Constraint(m.t, rule=tank_temperatur)


# def max_power_tank(m, t):
#     return m.Q_e[t] <= 10_000_000  # W
# m.max_power = pyo.Constraint(m.t, rule=max_power_tank)
#
#
# def min_power_tank(m, t):
#     return m.Q_e[t] >= 0
# m.min_power = pyo.Constraint(m.t, rule=min_power_tank)


# def max_power_heating(m, t):
#     return m.Q_heating[t] <= 10_000_000
# m.max_power_heating = pyo.Constraint(m.t, rule=max_power_heating)
#
#
# def min_power_heating(m, t):
#     return m.Q_heating[t] >= 0
# m.min_power_heating = pyo.Constraint(m.t, rule=min_power_heating)


# def max_mean_indoor_temp(m,t):
#     return m.Tm_t[t] <= 29
# m.max_mean_indoor_temp = pyo.Constraint(m.t, rule=max_mean_indoor_temp)
#
#
# def min_mean_indoor_temp(m,t):
#     return m.Tm_t[t] >= 18
# m.min_mean_indoor_temp = pyo.Constraint(m.t, rule=min_mean_indoor_temp)


instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
print(results)


# create plots to visualize results
def show_results():
    Q_a = [instance.Q_heating[t]() for t in m.time]
    T_room = [instance.T_room[t]() for t in m.time]
    T_room_mean = [instance.Tm_t[t]() for t in m.time]
    total_cost = instance.OBJ()

    # total_cost = instance.Objective()
    x_achse = np.arange(24)

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()

    # ax1.bar(x_achse, Q_e, label="boiler power")
    ax1.plot(x_achse, Q_a, label="heating power", color="red")
    ax1.plot(x_achse, Q_solar, label="solar power", color="green")
    ax2.plot(x_achse, price, color="orange", label="price")

    ax1.set_ylabel("energy Wh")
    # ax1.yaxis.label.set_color('green')

    ax2.set_ylabel("price per kWh")
    ax2.yaxis.label.set_color('orange')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="room temperature", color="blue")
    ax3.plot(x_achse, T_room_mean, label="mean room temperature 0", color="grey")

    # ax4.plot(x_achse, T_tank, label="tank temperature", color="orange")

    ax3.set_ylabel("room temperature °C")

    # ax4.set_ylabel("tank temperature °C")
    ax3.yaxis.label.set_color('blue')
    ax3.legend()
    plt.grid()

    ax1.set_title("Total costs: " + str(round(total_cost / 1000, 3)))
    plt.show()


show_results()

