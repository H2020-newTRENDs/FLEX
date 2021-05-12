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


#------------------------------------------------------------------------------
#fix starting values
Start_SFL = 0
Eff_Charge = 0.95
Eff_Discharge = 0.95
Power_ChargeMax = 4.5
Power_DischargeMax = 4.5
SelfDischarge = 3
Capacity = 15
hours_of_simulation = 96
SelfDischarge_hourly = Capacity/100*SelfDischarge/30/24


#------------------------------------------------------------------------------


#fixed profiles
load_profile = create_dict([1,1,1,1, 1,1,3,1, 1,1,2,3, 2,1,1,1, 1,2,3,2, 1,1,1,1]*int(hours_of_simulation/24))
pv_profile = create_dict([0,0,0,0, 0,0,1,1, 2,4,5,8, 8,7,4,3, 1,0,0,0 ,0,0,0,0,]*int(hours_of_simulation/24))
pv_price = create_dict([0.0792]*hours_of_simulation)
var2 = create_dict([0.30]*hours_of_simulation)
self_discharge = create_dict([SelfDischarge_hourly]*hours_of_simulation)


m = pyo.AbstractModel()
#------------------------------------------------------------------------------
#parameters
m.t = pyo.RangeSet(1,hours_of_simulation)
m.load_profile = pyo.Param(m.t, initialize= load_profile)
m.pv_profile = pyo.Param(m.t, initialize= pv_profile)
m.pv_price = pyo.Param(m.t, initialize= pv_price)
m.p = pyo.Param(m.t, initialize=var2)
m.self_discharge = pyo.Param(m.t, initialize=self_discharge)
#------------------------------------------------------------------------------
#variables

m.sfl = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.bat_charge = pyo.Var(m.t, within=pyo.NonNegativeReals)

m.bat_pos = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.bat_discharge = pyo.Var(m.t, within=pyo.NonNegativeReals)

m.bat_neg = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.grid_load = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.pv_grid = pyo.Var(m.t, within=pyo.NonNegativeReals)


#------------------------------------------------------------------------------
#objective
def minimize_cost(m):
    rule = sum(m.grid_load[t]*m.p[t] - m.pv_grid[t]*m.pv_price[t] for t in m.t)
    return rule
m.OBJ = pyo.Objective(rule=minimize_cost)

#------------------------------------------------------------------------------
#constraints
def MaxBatLoad(m,t):
    return m.sfl[t] <= Capacity
m.MaxBatLoad = pyo.Constraint(m.t, rule=MaxBatLoad)

def MinBatLoad(m,t):
    return m.sfl[t] >= 0
m.MinBatLoad = pyo.Constraint(m.t, rule=MinBatLoad)

def MaxDischarge(m,t):
    return m.bat_discharge[t] <= Power_DischargeMax
m.MaxDischarge = pyo.Constraint(m.t, rule=MaxDischarge)

def MaxCharge(m,t):
    return m.bat_charge [t] <= Power_ChargeMax
m.MaxCharge = pyo.Constraint(m.t, rule=MaxCharge)

def BatChargeandDischarge(m,t):
    if t == 1:
        return m.sfl[t] == Start_SFL
    else:
        m.sfl[t] = m.sfl[t - 1] + m.bat_pos[t] - m.bat_neg[t] - m.self_discharge[t]
        return m.sfl[t]
m.BatChargeandDischarge = pyo.Constraint(m.t, rule=BatChargeandDischarge)

def UseOfPV(m,t):
    return m.bat_charge[t] == m.pv_profile[t] - m.pv_grid[t]
m.UseOfPV = pyo.Constraint(m.t, rule=UseOfPV)


def LoadEnergy(m,t):
    return m.load_profile[t] == m.bat_discharge[t] + m.grid_load[t]
m.LoadEnergy = pyo.Constraint(m.t, rule=LoadEnergy)

def DischargeLoss(m,t):
    return m.bat_discharge[t] == m.bat_neg[t] - (m.bat_neg[t]*(1-Eff_Discharge))
m.DischargeLoss = pyo.Constraint(m.t, rule=DischargeLoss)

def ChargeLoss(m,t):
    return m.bat_pos[t] == m.bat_charge[t] - (m.bat_charge[t]*(1-Eff_Charge))
m.ChargeLoss = pyo.Constraint(m.t, rule=ChargeLoss)

instance = m.create_instance(report_timing=True)
opt = pyo.SolverFactory("glpk")
results = opt.solve(instance, tee=True)
print(results)

price = [var2[i] for i in range(1, hours_of_simulation+1)]
pv_profile = [pv_profile[i] for i in range(1, hours_of_simulation+1)]

sfl = [instance.sfl[t]() for t in m.t]
bat_charge = [instance.bat_charge[t]() for t in m.t]
bat_discharge = [instance.bat_discharge[t]() for t in m.t]
grid_load = [instance.grid_load[t]() for t in m.t]
pv_grid = [instance.pv_grid[t]() for t in m.t]

print(sfl)
print(bat_charge)
print(bat_discharge)
print(grid_load)
print(price)
print(pv_grid)

def show_results():
    cost = [list(price)[i] * grid_load[i] for i in range(hours_of_simulation)]
    total_cost = instance.Object()

    plt.style.use('seaborn-whitegrid')


    x_axis = np.arange(hours_of_simulation)
    plt.plot(x_axis, sfl, label= 'SFL', color='black', linestyle='solid')
#    plt.plot(x_axis, bat_charge, label ='Charge', color='green', linestyle='dashed', linewidth=0.5)
#    plt.plot(x_axis, bat_discharge, label='Discharge', color='red', linestyle='dotted', linewidth =0.5)
    plt.plot(x_axis, pv_profile, label='PV_Profile', color='orange', linestyle='solid',linewidth=0.5)
    plt.plot(x_axis, pv_grid, label='PV_grid', color='green', linestyle='solid',linewidth=0.5)
    plt.xlabel('hours of simulation')
    plt.ylabel('Power in kWh')
    plt.legend()
    plt.title('Total cost(€): ' +str(round(total_cost,2)))


    plt.show()



show_results()