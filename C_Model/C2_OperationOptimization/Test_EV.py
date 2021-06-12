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




BatCapacity = 10
EVCapacity = 50
EVDischargeDriving = 5
HoursOfSimulation = 96

if EVCapacity == 0:
    CarAtHome = create_dict([1] * HoursOfSimulation)
else:
    CarAtHome = create_dict([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


LoadProfile = create_dict([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

PhotovoltaicProfile = create_dict([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 5, 5, 5, 3, 2, 2, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 5, 5, 5, 3, 2, 2, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 5, 5, 5, 3, 2, 2, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 5, 5, 5, 3, 2, 2, 1, 0, 0, 0, 0, 0])


FiT = create_dict([0.08] * HoursOfSimulation)
GridPrice = create_dict([0.30] * HoursOfSimulation)
V2G = create_dict([0] * HoursOfSimulation)          # from database?



m = pyo.AbstractModel()

# Parameters
m.t = pyo.RangeSet(HoursOfSimulation)
m.LoadProfile = pyo.Param(m.t, initialize=LoadProfile)
m.PhotovoltaicProfile = pyo.Param(m.t, initialize=PhotovoltaicProfile)
m.FiT = pyo.Param(m.t, initialize=FiT)
m.GridPrice = pyo.Param(m.t, initialize=GridPrice)
m.CarStatus = pyo.Param(m.t, initialize=CarAtHome)
m.V2G = pyo.Param(m.t, initialize=V2G)

# Variables
# EV
m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))

m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, BatCapacity))
m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 5))
# m.Bat2Grid


m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 11))
m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, EVCapacity))


# m.EV2Grid
# objective
def minimize_cost(m):
    rule = sum(m.Grid[t] * m.GridPrice[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
    return rule


m.OBJ = pyo.Objective(rule=minimize_cost)


# (1) Energy balance of supply == demand
def calc_ElectricalEnergyBalance(m, t):
    return m.Grid[t] + m.PhotovoltaicProfile[t] + m.BatDischarge[t] + m.EVDischarge[t] == m.Feedin[t] + \
           m.Load[t] + m.BatCharge[t] + m.EVCharge[t]


m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)


# (2)
def calc_UseOfGrid(m, t):
    return m.Grid[t] == m.Grid2Load[t] + m.Grid2EV[t] * m.CarAtHomeStatus[t]


m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)


# (3)
def calc_UseOfPV(m, t):
    return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] == m.PhotovoltaicProfile[t]


m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

# (4)
def calc_SumOfFeedin(m, t):
    return m.Feedin[t] == m.PV2Grid[t]


m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

# (5)
def calc_SupplyOfLoads(m, t):
    return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t] * m.V2B[t] * m.CarAtHomeStatus[t] == m.Load[t]


m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)


# (6)
def calc_SumOfLoads(m, t):
    return m.Load[t] == m.LoadProfile[t]


m.calc_SumOfLoads = pyo.Constraint(m.t, rule=calc_SumOfLoads)


# (7)
def calc_BatDischarge(m, t):
    if BatCapacity == 0:
        return m.BatDischarge[t] == 0
    elif m.t[t] == 1:
        return m.BatDischarge[t] == 0
    elif m.t[t] == HoursOfSimulation:
        return m.BatDischarge[t] == 0
    else:
        return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.CarAtHomeStatus[t]


m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

# (8)
def calc_BatCharge(m, t):
    if BatCapacity == 0:
        return m.BatCharge[t] == 0
    else:
        return m.BatCharge[t] == m.PV2Bat[t] + m.EV2Bat[t] * m.V2B[t]


m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

# (9)
def calc_BatSoC(m, t):
    if t == 1:
        return m.BatSoC[t] == 0
    elif BatCapacity == 0:
        return m.BatSoC[t] == 0
    else:
        return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * 0.95 - m.BatDischarge[t] * 1.05


m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)


# (10)
def calc_EVCharge(m, t):
    if EVCapacity == 0:
        return m.EVCharge[t] == 0
    elif m.CarAtHomeStatus[t] == 0:
        return m.EVCharge[t] == 0
    else:
        return m.EVCharge[t] * m.CarAtHomeStatus[t] == m.Grid2EV[t] * m.CarAtHomeStatus[t] + m.PV2EV[t] * m.CarAtHomeStatus[t] + m.Bat2EV[t] * m.CarAtHomeStatus[t]


m.calc_EVCharge = pyo.Constraint(m.t, rule=calc_EVCharge)


# (11)
def calc_EVDischarge(m, t):
    if t == 1:
        return m.EVDischarge[t] == 0
    elif EVCapacity == 0:
        return m.EVDischarge[t] == 0
    elif m.CarAtHomeStatus[t] == 0 and m.V2B[t] == 1:
        return m.EVDischarge[t] == 0
    elif m.t[t] == HoursOfSimulation:
        return m.EVDischarge[t] == 0
    else:
        return m.EVDischarge[t] == m.EV2Load[t] * m.V2B[t] * m.CarAtHomeStatus[t] + m.EV2Bat[t] * m.V2B[t] * m.CarAtHomeStatus[t]


m.calc_EVDischarge = pyo.Constraint(m.t, rule=calc_EVDischarge)


# (12)
def calc_EVSoC(m, t):
    if t == 1:
        return m.EVSoC[t] == 0
    elif EVCapacity == 0:
        return m.EVSoC[t] == 0
    else:
        return m.EVSoC[t] == m.EVSoC[t - 1] + (m.EVCharge[t] * m.CarAtHomeStatus[t] * 0.95) - (m.EVDischarge[t] * m.CarAtHomeStatus[
            t] * 1.05) - (3 * (1 - m.CarAtHomeStatus[t]))


m.calc_EVSoC = pyo.Constraint(m.t, rule=calc_EVSoC)

instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
instance.display("./log.txt")
print(results)
total_cost = instance.OBJ()

print(total_cost)

# # Visualization
PhotovoltaicProfile = np.array(list(instance.PhotovoltaicProfile.extract_values().values()))

# EVDischarge = np.nan_to_num(np.array(np.array([instance.EVDischarge[t]() for t in m.t]), dtype=np.float), nan=0)

BatCharge = np.nan_to_num(np.array(np.array([instance.BatCharge[t]() for t in m.t]), dtype=np.float), nan=0)
BatDischarge = np.nan_to_num(np.array(np.array([instance.BatDischarge[t]() for t in m.t]), dtype=np.float), nan=0)
BatSoC = np.array(list(instance.BatSoC.extract_values().values()))

PV2Load = np.nan_to_num(np.array(np.array([instance.PV2Load[t]() for t in m.t]), dtype=np.float), nan=0)
PV2Bat = np.nan_to_num(np.array(np.array([instance.PV2Bat[t]() for t in m.t]), dtype=np.float), nan=0)
PV2Grid = np.nan_to_num(np.array(np.array([instance.PV2Grid[t]() for t in m.t]), dtype=np.float), nan=0)
PV2EV = np.nan_to_num(np.array(np.array([instance.PV2EV[t]() for t in m.t]), dtype=np.float), nan=0)

EVSoC = np.nan_to_num(np.array(np.array([instance.EVSoC[t]() for t in m.t]), dtype=np.float), nan=0)
EVCharge = np.nan_to_num(np.array(np.array([instance.EVCharge[t]() for t in m.t]), dtype=np.float), nan=0)
EVDischarge = np.nan_to_num(np.array(np.array([instance.EVDischarge[t]() for t in m.t]), dtype=np.float), nan=0)
Car = np.array(list(instance.CarAtHomeStatus.extract_values().values()))

EV2Load = np.nan_to_num(np.array(np.array([instance.EV2Load[t]() for t in m.t]), dtype=np.float), nan=0)
EV2Bat = np.nan_to_num(np.array(np.array([instance.EV2Bat[t]() for t in m.t]), dtype=np.float), nan=0)

Grid2EV = np.nan_to_num(np.array(np.array([instance.Grid2EV[t]() for t in m.t]), dtype=np.float), nan=0)
PV2EV = np.nan_to_num(np.array(np.array([instance.PV2EV[t]() for t in m.t]), dtype=np.float), nan=0)
Bat2EV = np.nan_to_num(np.array(np.array([instance.Bat2EV[t]() for t in m.t]), dtype=np.float), nan=0)

x_achse = np.arange(96)

# Battery
fig, ax1 = plt.subplots()

ax1.plot(x_achse, PhotovoltaicProfile, linewidth=3, label='PV', color='orange', alpha=0.1)
#
ax1.bar(x_achse, BatDischarge, linewidth=2, label='BatDischarge', color='red', alpha=0.5, linestyle='dotted')
ax1.bar(x_achse, BatCharge, linewidth=2, label='BatCharge', color='green', alpha=0.5, linestyle='dotted')
ax1.set_ylabel("Power kW")
#
ax2 = ax1.twinx()
ax2.bar(x_achse, BatSoC, linewidth=1, label='BatSoC', color='blue', alpha=0.07)
# ax2.fill_between(x_achse, BatSoC, alpha=0.05, color='blue')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_ylabel("SoC in kWh")
#
plt.title('Battery')
plt.tight_layout()
plt.show()

# Photvoltaic

fig, ax1 = plt.subplots()

ax1.plot(x_achse, PhotovoltaicProfile, linewidth=3, label='PV', color='orange', alpha=0.5)
# ax1.fill_between(x_achse, PhotovoltaicProfile, alpha=0.1, color='orange')

ax1.bar(x_achse, PV2Load, linewidth=2, label='PV2Load', color='green', alpha=0.5)
ax1.bar(x_achse, PV2Bat, bottom=PV2Load, linewidth=2, label='PV2Bat', color='blue', alpha=0.5)
ax1.bar(x_achse, PV2Grid, bottom=PV2Load + PV2Bat, linewidth=2, label='PV2Grid', color='red', alpha=0.5)
ax1.bar(x_achse, PV2EV, bottom=PV2Load + PV2Bat + PV2Grid, linewidth=2, label='PV2EV', color='grey', alpha=0.5)

ax1.set_ylabel("Power kW")
#

lines, labels = ax1.get_legend_handles_labels()

ax1.legend(lines, labels, loc='upper right')

#
plt.title('Use of Photovoltaic')
plt.tight_layout()
plt.show()

# EV

fig, ax1 = plt.subplots()

ax1.plot(x_achse, PhotovoltaicProfile, linewidth=3, label='PV', color='orange', alpha=0.3)

ax1.bar(x_achse, EVSoC, linewidth=2, label='EVSoC', color='grey', alpha=0.4, linestyle='dotted')
ax1.bar(x_achse, EVCharge, linewidth=2, label='EVCharge', color='green', alpha=0.5)
ax1.bar(x_achse, EVDischarge, linewidth=2, label='EVDischarge', color='red', alpha=0.5)
ax1.set_ylabel("Power kW")

#
ax2 = ax1.twinx()
ax2.bar(x_achse, Car, linewidth=1, label='Car@Home', color='pink', alpha=0.3)
# ax2.fill_between(x_achse, Car, alpha=0.05, color='blue')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_ylabel("Car@Home 1 or 0")

#
plt.title('EV')
plt.tight_layout()
plt.show()

# EV Discharge

fig, ax1 = plt.subplots()

# ax1.bar(x_achse, Car, linewidth=3, label='Car', color='blue', alpha=0.07)
# ax1.fill_between(x_achse, Car, alpha=0.1, color='blue')

ax1.plot(x_achse, EVDischarge, linewidth=2, label='EVDischarge', color='red', alpha=0.5)
ax1.bar(x_achse, EV2Load, linewidth=2, label='EV2Load', color='green', alpha=0.5)
ax1.bar(x_achse, EV2Bat, bottom=EV2Load, linewidth=2, label='EV2Bat', color='blue', alpha=0.5)

ax1.plot(x_achse, EVCharge, linewidth=2, label='EVCharge', color='darkgreen', alpha=0.5)
ax1.bar(x_achse, Grid2EV, linewidth=2, label='Grid2EV', color='darkred', alpha=0.5)
ax1.bar(x_achse, PV2EV, bottom=Grid2EV, linewidth=2, label='PV2EV', color='black', alpha=0.5)
ax1.bar(x_achse, Bat2EV, bottom=Grid2EV + PV2EV, linewidth=2, label='Bat2EV', color='orange', alpha=0.5)
ax1.set_ylabel("Power kW")

ax2 = ax1.twinx()
ax2.bar(x_achse, Car, linewidth=1, label='Car@Home', color='blue', alpha=0.05)
# ax2.fill_between(x_achse, Car, alpha=0.05, color='blue')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')
ax2.set_ylabel("Car@Home 1 or 0")
#

#
plt.title('EV Discharge and Charge')
plt.tight_layout()
plt.show()
