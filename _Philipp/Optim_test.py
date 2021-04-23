import pyomo.environ as pyo
import numpy as np
from pathlib import Path
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt


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
    datei = str(run_number_str) + "__dynamic_calc_data_bc_" + str(YEAR) + ".npz"
    data_np = np.load(results_path_rcm / datei)
    data = pd.DataFrame(data=data_np["arr_0"], columns=data_np["arr_1"][0])
    data.columns = data.columns.str.decode("utf-8")
    data.columns = data.columns.str.replace(" ", "")

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

    return Atot[0], Hve[0], Htr_w[0], Hop[0], Cm[0], Am[0], Qi[0], \
           temp_outside[0:24], Qsol


def room_temperature_rc(Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Qsol, Q_heating,
                        T_outside, Tm_t_prev):
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
    # Equ. C.2
    PHI_m = Am / At * (0.5 * Qi + Qsol)
    # Equ. C.3
    PHI_st = (1 - Am / At - Htr_w / 9.1 / At) * (0.5 * Qi + Qsol)
    # Equ. C.6
    Htr_1 = np.float_(1) / (np.float_(1) / Hve + np.float_(1) / Htr_is)
    # Equ. C.7
    Htr_2 = Htr_1 + Htr_w
    # Equ.C.8
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)

    # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
    T_sup = T_outside
    # Equ. C.5
    PHI_mtot = PHI_m + Htr_em * T_outside + Htr_3 * (
            PHI_st + Htr_w * T_outside + Htr_1 * (((PHI_ia + Q_heating) / Hve) + T_sup)) / \
               Htr_2

    # Equ. C.4
    Tm_t = (Tm_t_prev * ((Cm / 3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / ((Cm / 3600) + 0.5 * (Htr_3 + Htr_em))

    # Equ. C.9
    T_m = (Tm_t + Tm_t_prev) / 2
    # Euq. C.10
    T_s = (Htr_ms * T_m + PHI_st + Htr_w * T_outside + Htr_1 * (T_sup + (PHI_ia + Q_heating) / Hve)) / (
            Htr_ms + Htr_w + Htr_1)
    # Equ. C.11
    T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + Q_heating) / (Htr_is + Hve)
    # Equ. C.12
    T_op = 0.3 * T_air + 0.7 * T_s
    # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
    return Tm_t, T_op


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

Atot, Hve, Htr_w, Hop, Cm, Am, Qi, temp_outside, Qsol = get_invert_data()

# fixed starting values:
tank_starting_temp = 50
indoor_starting_temp = 22

# water mass in storage
m_water = 1_000  # l
# thermal capacity water
cp_water = 4.2  # kJ/kgK

# var = {1: 20, 2: 25, 3: 15, 4: 12, 5: 25, 6: 20}  # Q_a
var2 = create_dict([1, 6, 8, 8, 2, 1])  # price
ta = create_dict([20, 20, 20, 20, 20, 20])  # constant surrounding temp
tout = create_dict([5, 4, 0, -1, -2, -2])  # outside temperature
# Q_loss = create_dict([4, 3, 5, 3.2, 3, 2])  # Losses of the building


# model
m = pyo.AbstractModel()

# parameters
m.t = pyo.RangeSet(1, 6)
m.p = pyo.Param(m.t, initialize=var2)
m.T_a = pyo.Param(m.t, initialize=ta)
m.T_out = pyo.Param(m.t, initialize=tout)

# variables
m.Q_e = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.T_tank = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.Q_a = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals)


# objective
def minimize_cost(m):
    rule = sum(m.Q_e[t] * m.p[t] for t in m.t)
    return rule


m.OBJ = pyo.Objective(rule=minimize_cost)


# constraints
def maximum_temperature_tank(m, t):
    return m.T_tank[t] <= 100


m.maximum_temperature_tank = pyo.Constraint(m.t, rule=maximum_temperature_tank)


def minimum_temperature_tank(m, t):
    return m.T_tank[t] >= 20


m.minimum_temperature_tank = pyo.Constraint(m.t, rule=minimum_temperature_tank)


def maximum_temperature_room(m, t):
    return m.T_room[t] <= 28


m.maximum_temperature_raum = pyo.Constraint(m.t, rule=maximum_temperature_room)


def minimum_temperature_room(m, t):
    return m.T_room[t] >= 20


m.minimum_temperature_raum = pyo.Constraint(m.t, rule=minimum_temperature_room)


def room_temperature(m, t):
    if t == 1:
        return m.T_room[t] == indoor_starting_temp - U * Am / Cm * (m.T_room[t] - m.T_out[t]) + m.Q_a[t] / Cm
    else:
        return m.T_room[t] == m.T_room[t - 1] - U * Am / Cm * (m.T_room[t] - m.T_out[t]) + m.Q_a[t] / Cm


m.room_temperature = pyo.Constraint(m.t, rule=room_temperature)


def tank_temperatur(m, t):
    if t == 1:
        return m.T_tank[t] == tank_starting_temp - m.Q_a[t] / m_water / cp_water * 3600 + \
               m.Q_e[t] / m_water / cp_water * 3600 - 0.003 * (tank_starting_temp - m.T_a[t])
    else:
        return m.T_tank[t] == m.T_tank[t - 1] - m.Q_a[t] / m_water / cp_water * 3600 + \
               m.Q_e[t] / m_water / cp_water * 3600 - 0.003 * (m.T_tank[t] - m.T_a[t])


m.tank_temperatur = pyo.Constraint(m.t, rule=tank_temperatur)


def max_power_tank(m, t):
    return m.Q_e[t] <= 10_000  # W


m.max_power = pyo.Constraint(m.t, rule=max_power_tank)


def min_power_tank(m, t):
    return m.Q_e[t] >= 0


m.min_power = pyo.Constraint(m.t, rule=min_power_tank)


def max_power_heating(m, t):
    return m.Q_a[t] <= 10_000


m.max_power_heating = pyo.Constraint(m.t, rule=max_power_heating)


def min_power_heating(m, t):
    return m.Q_a[t] >= 0


m.min_power_heating = pyo.Constraint(m.t, rule=min_power_heating)

instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
print(results)


# create plots to visualize results
def show_results():
    Q_e = [instance.Q_e[t]() for t in m.t]
    price = [var2[i] for i in range(1, 7)]
    Q_a = [instance.Q_a[t]() for t in m.t]
    T_room = [instance.T_room[t]() for t in m.t]
    T_tank = [instance.T_tank[t]() for t in m.t]

    # indoor temperature is constant 20°C
    cost = [list(price)[i] * Q_e[i] for i in range(6)]
    total_cost = instance.OBJ()
    x_achse = np.arange(6)

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    ax1.bar(x_achse, Q_e, label="boiler power")
    ax1.plot(x_achse, Q_a, label="heating power", color="green")
    ax2.plot(x_achse, price, color="orange", label="price")

    ax1.set_ylabel("energy Wh")
    ax2.set_ylabel("price per kWh")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="room temperature", color="blue")
    ax4.plot(x_achse, T_tank, label="tank temperature", color="orange")

    ax3.set_ylabel("room temperature °C")
    ax4.set_ylabel("tank temperature °C")
    ax3.yaxis.label.set_color('blue')
    ax4.yaxis.label.set_color('orange')
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc=0)
    plt.grid()

    ax1.set_title("Total costs: " + str(round(total_cost / 1000, 3)))
    plt.show()


if __name__ == "__main__":
    sol_rad_north, sol_rad_south, sol_rad_east_west, sol_rad_norm, \
    Atot, Hve, Htr_w, Hop, Cm, Am, Qi, \
    Awindows_rad_east_west, Awindows_rad_north, Awindows_rad_south, \
    temp_outside = \
        get_invert_data()




    show_results()
