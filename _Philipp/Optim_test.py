import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints


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


# fixed starting values:
tank_starting_temp = 50
indoor_starting_temp = 22
# U-Value
U = 0.2  # W/m2K
# heat capacity room
C_h = 2000  # J/K
# useful area
Am = 425  # m2

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
        return m.T_room[t] == indoor_starting_temp - U * Am / C_h * (m.T_room[t] - m.T_out[t]) + m.Q_a[t] / C_h
    else:
        return m.T_room[t] == m.T_room[t - 1] - U * Am / C_h * (m.T_room[t] - m.T_out[t]) + m.Q_a[t] / C_h
m.room_temperature = pyo.Constraint(m.t, rule=room_temperature)


def tank_temperatur(m, t):
    if t == 1:
        return m.T_tank[t] == tank_starting_temp - m.Q_a[t] / 1000 / 4.2 * 3600 + m.Q_e[t] / 1000 / 4.2 * 3600 - 0.003 \
               * (tank_starting_temp - m.T_a[t])
    else:
        return m.T_tank[t] == m.T_tank[t - 1] - m.Q_a[t] / 1000 / 4.2 * 3600 + m.Q_e[t] / 1000 / 4.2 * 3600 - \
               0.003 * (m.T_tank[t] - m.T_a[t])
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
# opt = pyo.SolverFactory("gurobi")
opt = pyo.SolverFactory("glpk")
results = opt.solve(instance, tee=True)
print(results)


# create plots to visualize resultsprice
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

    ax1.set_title("Total costs: " + str(round(total_cost/1000, 3)))
    plt.show()


show_results()
