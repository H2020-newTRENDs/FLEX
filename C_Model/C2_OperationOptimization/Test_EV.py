import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt
import numpy as np


def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary

# toDo:
# Generate Input with 1 / 0 with Leave and come time from DB
# Include stationary battery system in model
# EV Demand calculation




CarAtHome = create_dict([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

LoadProfile = create_dict([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

PhotovoltaicProfile = create_dict([0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0])

HoursOfSimulation = 48
FiT = create_dict([0.08] * HoursOfSimulation)
GridPrice = create_dict([0.30] * HoursOfSimulation)

m = pyo.AbstractModel()

# Parameters
m.t = pyo.RangeSet(HoursOfSimulation)
m.LoadProfile = pyo.Param(m.t, initialize=LoadProfile)
m.PhotovoltaicProfile = pyo.Param(m.t, initialize=PhotovoltaicProfile)
m.FiT = pyo.Param(m.t, initialize=FiT)
m.GridPrice = pyo.Param(m.t, initialize=GridPrice)
m.CarAtHome = pyo.Param(m.t, initialize=CarAtHome)

# Variables
# EV
m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10))
m.SolarEVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.GridEVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EVBatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))



# Grid and PV
m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.GridDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))


# objective
def minimize_cost(m):
    rule = sum(m.Grid[t] * m.GridPrice[t] - m.PhotovoltaicFeedin[t] * m.FiT[t] for t in m.t)
    return rule

m.OBJ = pyo.Objective(rule=minimize_cost)

# (1) Energy balance of supply == demand
def calc_ElectricalEnergyBalance(m, t):
    return m.Grid[t] + m.PhotovoltaicProfile[t] + m.EVDischarge[t] * CarAtHome[t]  == m.PhotovoltaicFeedin[t] + \
           m.LoadProfile[t] + m.EVBatCharge[t] * CarAtHome[t]

m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)

# (2) Use of Photovoltaic
def calc_UseOfPhotovoltaic(m, t):
    return m.SolarEVCharge[t] * CarAtHome[t] + m.PhotovoltaicFeedin[t] + m.PhotovoltaicDirect[t]  == \
           m.PhotovoltaicProfile[t]

m.calc_UseOfPhotovoltaic = pyo.Constraint(m.t, rule=calc_UseOfPhotovoltaic)

# (3) Use of Grid
def calc_UseOfGrid(m, t):
    return m.Grid[t] == m.GridDirect[t] + m.GridEVCharge[t] * CarAtHome[t]

m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

# (4) Supply of Load
def calc_SupplyOfLoads(m, t):
    return m.EVDischarge[t] * CarAtHome[t] + m.PhotovoltaicDirect[t] + m.GridDirect[t] == m.LoadProfile[t]
m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

# (5) Charge of EVBattery
def calc_EVBatteryCharge(m, t):
    return m.EVBatCharge[t] == m.SolarEVCharge[t] * CarAtHome[t] + m.GridEVCharge[t] * CarAtHome[t]
m.calc_EVBatteryCharge = pyo.Constraint(m.t, rule=calc_EVBatteryCharge)

# (6) SoC of EVBattery
def calc_EVSoC(m, t):
    if t == 1:
        return m.EVSoC[t] == 0
    else:
        return m.EVSoC[t] == m.EVSoC[t - 1] - 1 * (1 - CarAtHome[t]) \
               + m.EVBatCharge[t - 1] * 0.95 * CarAtHome[t] - m.EVDischarge[t - 1] * 1.05 * \
               CarAtHome[t]

m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)




instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
instance.display("./log.txt")
print(results)
total_cost = instance.OBJ()

print(total_cost)

# Visualization
PhotovoltaicProfile = np.array(list(instance.PhotovoltaicProfile.extract_values().values()))
CarAtHome = np.array(list(instance.CarAtHome.extract_values().values()))

#EVBatCharge = np.array(list(instance.EVBatCharge.extract_values().values()))
EVBatCharge = np.nan_to_num(np.array(np.array([instance.EVBatCharge[t]() for t in m.t]), dtype=np.float), nan=0)
EVBatDischarge = np.nan_to_num(np.array(np.array([instance.EVDischarge[t]() for t in m.t]), dtype=np.float), nan=0)

#EVBatDischarge = np.array(list(instance.EVDischarge.extract_values().values()))

Grid = np.array(list(instance.Grid.extract_values().values()))
EVSoC = np.array(list(instance.EVSoC.extract_values().values()))

#




x_achse = np.arange(48)
fig, ax1 = plt.subplots()

ax1.bar(x_achse, Grid, linewidth=1, label='Grid', color='black', alpha=0.2)
ax1.bar(x_achse, EVBatCharge, bottom = Grid, linewidth=1, label='EVBatCharge', color='green', alpha=0.2)
ax1.bar(x_achse, EVBatDischarge, bottom = Grid, linewidth=1, label='EVBatDischarge', color='red', alpha=0.2)
ax1.plot(x_achse, EVSoC, linewidth=2, label='EVSoC', color= 'grey', alpha=0.3, linestyle = 'dotted')
ax1.plot(x_achse, PhotovoltaicProfile, linewidth=3, label='PV', color= 'orange', alpha=0.1)




ax1.set_ylabel("Power kW")

ax2 = ax1.twinx()
ax2.plot(x_achse, CarAtHome, linewidth=1, label='CarAtHome', color='blue', alpha=0.1)
ax2.fill_between(x_achse, CarAtHome, alpha = 0.05)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_ylabel("Car At Home, 1 or 0")

plt.title('Integration of EV')
plt.tight_layout()
plt.show()
