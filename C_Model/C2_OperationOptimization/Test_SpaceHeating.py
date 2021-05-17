import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints
import random





def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary


# fixed starting values:
tank_starting_temp = 50
indoor_starting_temp = 22
t_base = 20

c_water = 4200/3600
c_air = 718/3600
density_air = 1.225

V_building = 400
A_building = 300
U_building = 0.2
m_air = V_building*density_air

V_tank = 1
A_Tank = 10
U_Tank = 0.4
m_water = V_tank*1000

builiding_starting_energy = m_air*c_air*(indoor_starting_temp-t_base)
tank_starting_energy = m_water*c_water*(tank_starting_temp - t_base)


def random_price(size):
    return [random.randint(16, 40) for i in size * [None]]

hours_of_simulation = 24

def random_tout(size):
    return [random.randint(-10, 15) for i in size * [None]]
tout = create_dict(random_tout(hours_of_simulation))


var2 = (create_dict(random_price(hours_of_simulation)))
ta = create_dict([20]*hours_of_simulation)  # constant surrounding temp



# model
m = pyo.AbstractModel()

# parameters
m.t = pyo.RangeSet(1, hours_of_simulation)
m.p = pyo.Param(m.t, initialize=var2)
m.T_a = pyo.Param(m.t, initialize=ta)
m.T_out = pyo.Param(m.t, initialize=tout)

# variables
m.Q_e = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.E_tank = pyo.Var(m.t, within=pyo.NonNegativeReals)

m.Q_a = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.E_room = pyo.Var(m.t, within=pyo.NonNegativeReals)

m.T_tank = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals)


# objective
def minimize_cost(m):
    rule = sum(m.Q_e[t] * m.p[t] for t in m.t)
    return rule
m.OBJ = pyo.Objective(rule=minimize_cost)


# constraints
#def maximum_energy_tank(m, t):
#    return m.E_tank[t] <= 35000 #50°C
#m.maximum_energy_tank = pyo.Constraint(m.t, rule=maximum_energy_tank)

def maximum_temperature_tank(m,t):
    return m.T_tank[t] <= 50
m.maximum_temperature_tank = pyo.Constraint(m.t, rule = maximum_temperature_tank)


def minimum_temperature_tank(m, t):
    return m.T_tank[t] >= 25
m.minimum_temperature_tank = pyo.Constraint(m.t, rule=minimum_temperature_tank)


def maximum_temperature_room(m,t):
    return m.T_room[t] <= 23
m.maximum_temperatur_room = pyo.Constraint(m.t, rule = maximum_temperature_room)

#def maximum_energy_room(m, t):
#    return m.E_room[t] <= 586.36
#m.maximum_energy_raum = pyo.Constraint(m.t, rule=maximum_energy_room)


def minimum_energy_room(m, t):
    return m.E_room[t] >= 0
m.minimum_energy_raum = pyo.Constraint(m.t, rule=minimum_energy_room)


def room_energy(m, t):
    if t == 1:
        return m.E_room[t] == builiding_starting_energy
    else:
        return m.E_room[t] == m.E_room[t-1] - U_building*A_building*(m.T_room[t-1]-tout[t-1]) + m.Q_a[t-1]
m.room_energy = pyo.Constraint(m.t, rule=room_energy)

def room_temperature(m, t,):
    if t == 1:
        return m.T_room[t] == indoor_starting_temp
    else:
        return m.T_room[t] == (m.E_room[t]+(m_air*c_air*t_base))/(m_air*c_air)
m.room_temperature = pyo.Constraint(m.t, rule=room_temperature)



def tank_energy(m, t):
    if t == 1:
        return m.E_tank[t] == tank_starting_energy
    else:
        return m.E_tank[t] == m.E_tank[t-1] - m.Q_a[t-1] + m.Q_e[t-1] - U_Tank*A_Tank*(m.T_tank[t-1]-t_base)
m.tank_energy = pyo.Constraint(m.t, rule=tank_energy)


def tank_temperature(m, t,):
    if t == 1:
        return m.T_tank[t] == tank_starting_temp
    else:
        return m.T_tank[t] == (m.E_tank[t]+(m_water*c_water*t_base))/(m_water*c_water)
m.tank_temperature = pyo.Constraint(m.t, rule=tank_temperature)





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
opt = pyo.SolverFactory("glpk")
results = opt.solve(instance, tee=True)
print(results)


# create plots to visualize resultsprice
def show_results():
    Q_e = [instance.Q_e[t]() for t in m.t]
    price = [var2[i] for i in range(1, hours_of_simulation+1)]
    Q_a = [instance.Q_a[t]() for t in m.t]


    T_room = [instance.T_room[t]() for t in m.t]
    T_tank = [instance.T_tank[t]() for t in m.t]
    T_a = [tout[i] for i in range(1, hours_of_simulation + 1)]

    # indoor temperature is constant 20°C
    cost = [list(price)[i] * Q_e[i] for i in range(hours_of_simulation)]
    total_cost = instance.Object()
    x_achse = np.arange(hours_of_simulation)

    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
#    ax5 = ax3.twinx()

    ax1.bar(x_achse, Q_e, label="HP power", color = 'blue')
    ax1.plot(x_achse, Q_a, label="Floot heating power", color="green", linewidth =0.75)
    ax2.plot(x_achse, price, color="red", label="price", linewidth =0.75, linestyle =':')

    ax1.set_ylabel("Energy Wh")
    ax2.set_ylabel("Price per kWh in Ct/€")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax3.plot(x_achse, T_room, label="Room temperature", color="orange", linewidth =0.5)
    ax4.plot(x_achse, T_tank, label="Tank temperature", color="red", linewidth =0.5)
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

    ax1.set_title("Total costs un Ct/€ " + str(round(total_cost/1000, 3)))
    plt.show()


show_results()