import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt

def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary


LoadProfile = create_dict([1,1,1,1, 2,2,2,2])
PhotovoltaicProfile = create_dict([3,3,3,3, 0,0,0,0])


HoursOfSimulation = 8
FiT = create_dict([0.0792] * HoursOfSimulation)
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


# Variables
m.StorageFillLevel = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 50))
m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 4.5))
m.GridCover = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))
m.PhotovoltaicFeedin = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))
m.PhotovoltaicDirect = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(0, 16))


# objective
def minimize_cost(m):
    rule = sum(m.GridCover[t] * m.GridPrice[t] - m.PhotovoltaicFeedin[t] * m.FiT[t] for t in m.t)
    return rule
m.OBJ = pyo.Objective(rule=minimize_cost)


def calc_StorageFillLevel(m, t):
    if t == 1:
        return m.StorageFillLevel[t] == 5
    else:
#        m.StorageFillLevel[t] == m.StorageFillLevel[t - 1] + m.BatCharge[t] - m.BatDischarge[t]
        return m.StorageFillLevel[t] == m.StorageFillLevel[t - 1] + m.BatCharge[t] - m.BatDischarge[t]
m.calc_Storage = pyo.Constraint(m.t, rule=calc_StorageFillLevel)


def calc_BatteryCharge(m, t):
#    m.PhotovoltaicProfile[t] == m.BatCharge[t] + m.PhotovoltaicDirect[t] + m.PhotovoltaicFeedin[t]
    return m.PhotovoltaicProfile[t] == m.BatCharge[t] + m.PhotovoltaicDirect[t] + m.PhotovoltaicFeedin[t]

m.BatteryCharge = pyo.Constraint(m.t, rule=calc_BatteryCharge)



def calc_BatteryDischarge(m, t):
#    m.LoadProfile[t] == m.GridCover[t] + m.PhotovoltaicDirect[t] + m.BatDischarge[t]
    return m.LoadProfile[t] == m.GridCover[t] + m.PhotovoltaicDirect[t] + m.BatDischarge[t]

m.BatteryDischarge = pyo.Constraint(m.t, rule=calc_BatteryDischarge)

instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("glpk")
results = opt.solve(instance, tee=True)
print(results)
total_cost = instance.OBJ()

print(total_cost)

#price = [var2[i] for i in range(1, hours_of_simulation + 1)]
PhotovoltaicProfile = [PhotovoltaicProfile[i] for i in range (1, 9)]
StorageFillLevel = [instance.StorageFillLevel[t]() for t in m.t]

plt.figure()
plt.subplot(211)
plt.plot(PhotovoltaicProfile)
plt.title('PV')

plt.subplot(212)
plt.plot(StorageFillLevel)
plt.title('SOC')

plt.show()