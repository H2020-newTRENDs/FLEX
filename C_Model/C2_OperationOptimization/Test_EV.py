import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt


def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary


HoD = create_dict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

CarAtHome = create_dict([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

LoadProfile = create_dict([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

PhotovoltaicProfile = create_dict([0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0])

HoursOfSimulation = 48
FiT = create_dict([0.08] * HoursOfSimulation)
GridPrice = create_dict([0.30] * HoursOfSimulation)

print(LoadProfile)
print(PhotovoltaicProfile)
print(FiT)
print(GridPrice)

m = pyo.AbstractModel()

# Parameters
m.t = pyo.RangeSet(1, HoursOfSimulation)
m.LoadProfile = pyo.Param(m.t, initialize=LoadProfile)
m.PhotovoltaicProfile = pyo.Param(m.t, initialize=PhotovoltaicProfile)
m.FiT = pyo.Param(m.t, initialize=FiT)
m.GridPrice = pyo.Param(m.t, initialize=GridPrice)
m.CarAtHome = pyo.Param(m.t, initialize=CarAtHome)
m.HoD = pyo.Param(m.t, initialize=HoD)

# Variables
m.StateOfCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 10))
m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 20))
m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 20))
m.GridCover = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 100))
m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 100))
m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 100))
m.SumOfLoads = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 100))


# objective
def minimize_cost(m):
    rule = sum(m.GridCover[t] * m.GridPrice[t] - m.PhotovoltaicFeedin[t] * m.FiT[t] for t in m.t)
    return rule


m.OBJ = pyo.Objective(rule=minimize_cost)


# # Photovoltaic
def calc_UseOfPhotovoltaic(m, t):
    if m.CarAtHome[t] == 1:
        return m.PhotovoltaicDirect[t] + m.BatCharge[t] + m.PhotovoltaicFeedin[t] == m.PhotovoltaicProfile[t]
    if m.CarAtHome[t] == 0:
        return m.PhotovoltaicDirect[t] + m.PhotovoltaicFeedin[t] == m.PhotovoltaicProfile[t]

m.calc_UseOfPhotovoltaic_rule = pyo.Constraint(m.t, rule=calc_UseOfPhotovoltaic)


# Electrical energy balance: supply and demand
def calc_ElectricalEnergyBalance(m, t):
    if m.CarAtHome[t] == 1:
        return m.PhotovoltaicFeedin[t] + m.BatCharge[t]  == m.GridCover[t] + m.BatDischarge[t] + \
                m.PhotovoltaicProfile[t] - m.LoadProfile[t]
    if m.CarAtHome[t] == 0:
        return m.PhotovoltaicFeedin[t]  == m.GridCover[t] + m.PhotovoltaicProfile[t] - m.LoadProfile[t]

m.calc_ElectricalEnergyBalance = pyo.Constraint(m.t, rule=calc_ElectricalEnergyBalance)


def calc_SoC(m, t):
    if m.CarAtHome[t] == 1:
        if t == 1:
            return m.StateOfCharge[t] == 0
        elif t == 8:
            return m.StateOfCharge[t] == 8
        else:
            return m.StateOfCharge[t] == m.StateOfCharge[t - 1] + m.BatCharge[t - 1] - m.BatDischarge[t - 1]

    elif m.CarAtHome[t] == 0:
        if t == 1:
            return m.StateOfCharge[t] == 0
        else:
            return m.StateOfCharge[t] == m.StateOfCharge[t - 1] * 0.8


m.calc_SoC = pyo.Constraint(m.t, rule=calc_SoC)


def ChargetoZero(m, t):
    if m.CarAtHome[t] == 1:
        return m.PhotovoltaicDirect[t] + m.BatCharge[t] + m.PhotovoltaicFeedin[t] == m.PhotovoltaicProfile[t]
    elif m.CarAtHome[t] == 0:
        return m.BatCharge[t] == 0


m.ChargetoZero = pyo.Constraint(m.t, rule=ChargetoZero)

instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("gurobi")
results = opt.solve(instance, tee=True)
instance.display("./log.txt")
print(results)
total_cost = instance.OBJ()

print(total_cost)

PhotovoltaicProfile = [PhotovoltaicProfile[i] for i in range(1, 48)]
LoadProfile = [LoadProfile[i] for i in range(1, 48)]
CarAtHome = [CarAtHome[i] for i in range(1, 48)]

StateOfCharge = [instance.StateOfCharge[t]() for t in m.t]
BatCharge = [instance.BatCharge[t]() for t in m.t]
BatDischarge = [instance.BatDischarge[t]() for t in m.t]
GridCover = [instance.GridCover[t]() for t in m.t]
PhotovoltaicDirect = [instance.PhotovoltaicDirect[t]() for t in m.t]
StateOfCharge = [instance.StateOfCharge[t]() for t in m.t]

plt.figure()
plt.plot(BatCharge, color='green', label='BatCharge', linewidth=2, alpha=0.3)
plt.plot(BatDischarge, color='red', label='BatDischarge', linewidth=1, alpha=0.3)
plt.plot(StateOfCharge, color='black', label='SoC', linewidth=2, alpha=0.4, linestyle='dashed')
plt.plot(CarAtHome, color='grey', label='ControlTime', linewidth=2, alpha=0.2)
plt.plot(PhotovoltaicProfile, color='orange', label='PhotovoltaicProfile', alpha=0.2, linewidth=2)

plt.title('PV')
plt.legend()

plt.show()

plt.figure()
plt.plot(PhotovoltaicProfile, color='grey', label='PhotovoltaicProfile', linestyle='dashed')
plt.plot(GridCover, color='red', label='GridCover', linestyle='dashed', linewidth=0.5)
plt.plot(PhotovoltaicDirect, color='green', label='PhotovoltaicDirect', linewidth=0.5)
plt.plot(BatDischarge, color='blue', label='BatDischarge', linewidth=0.5)
plt.plot(LoadProfile, color='pink', label='LoadProfile', linewidth=0.5)

plt.title('Check')
plt.legend()

plt.show()
