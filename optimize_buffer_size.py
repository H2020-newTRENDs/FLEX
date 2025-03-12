import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pathlib import Path
from models.operation.model_base import OperationScenario, OperationModel
from pyomo.opt import TerminationCondition





def rewritten_5R1C_optimziation(
    model: OperationModel,

):
    """

    """

    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=np.arange(1, 8761))  # time

    m.T_outside = pyo.Param(m.t, initialize={t: model.T_outside[t-1] for t in m.t})
    m.Q_Solar = pyo.Param(m.t, initialize={t: model.Q_Solar[t-1] for t in m.t})
    m.CPWater = pyo.Param(initialize=model.CPWater)
    m.ElectricityPrice = pyo.Param(m.t, initialize={t: model.ElectricityPrice[t-1] for t in m.t})

    m.SpaceHeatingHourlyCOP = pyo.Param(m.t, initialize={t: model.SpaceHeatingHourlyCOP[t-1] for t in m.t})
    max_temp, min_temp = model.generate_target_indoor_temperature(temperature_offset=3)
    m.lower_room_temperature = pyo.Param(m.t, initialize={t: min_temp[t-1] for t in m.t})
    m.upper_room_temperature = pyo.Param(m.t, initialize={t: max_temp[t-1] for t in m.t})

    
    # building parameters
    m.Am = pyo.Param(initialize=model.Am)
    m.Atot = pyo.Param(initialize=model.Atot)
    m.Qi = pyo.Param(initialize=model.Qi)
    m.Htr_w = pyo.Param(initialize=model.Htr_w)
    m.Htr_em = pyo.Param(initialize=model.Htr_em)
    m.Htr_ms = pyo.Param(initialize=model.Htr_ms)
    m.Htr_is = pyo.Param(initialize=model.Htr_is)
    m.Hve = pyo.Param(initialize=model.Hve)
    m.PHI_ia = pyo.Param(initialize=model.PHI_ia)
    m.Cm = pyo.Param(initialize=model.Cm)
    m.BuildingMassTemperatureStartValue = pyo.Param(initialize=model.BuildingMassTemperatureStartValue)
    m.Htr_3 = model.Htr_3
    m.Htr_2 = model.Htr_2
    m.Htr_1 = model.Htr_1

    # set up variables
    m.T_Room = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.T_BuildingMass = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)

    def calc_supply_of_space_heating(m, t):
        return m.Q_RoomHeating[t] == m.E_Heating_HP_out[t] * m.SpaceHeatingHourlyCOP[t]
    m.calc_supply_of_space_heating_rule = pyo.Constraint( m.t, rule=calc_supply_of_space_heating )


    # def air_temperature(m, t):
    #     # equation correct, tested
    #     if t == 1:
    #         Tm_start = m.BuildingMassTemperatureStartValue
    #     else:
    #         Tm_start = m.T_BuildingMass[t - 1]

    #     PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
    #     PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
    #     u = (m.Htr_is+m.Htr_ms+m.Htr_w)
    #     r = (m.Htr_w*m.T_outside[t]+PHI_st) / u
    #     C_plus = m.Cm/3600 + 0.5*(m.Htr_ms + m.Htr_em)
    #     C_minus = m.Cm/3600 - 0.5*(m.Htr_ms + m.Htr_em)
    #     v = (m.Htr_ms*Tm_start*C_minus + m.Htr_ms*PHI_m + m.Htr_ms*m.Htr_em*m.T_outside[t]) / (C_plus*u)
    #     y = (1 - m.Htr_ms**2/(C_plus*u))
    #     w = (r + v) / y
    #     z = (1 - m.Htr_is**2 / ((m.Htr_is+m.Hve)*u*y))
    #     x = (m.Hve*m.T_outside[t] + m.Q_RoomHeating[t] + m.PHI_ia) / (m.Htr_is + m.Hve)
    #     T_room = x / z + m.Htr_is*w / ((m.Htr_is + m.Hve)*z)

    #     return pyo.inequality(m.lower_room_temperature[t], T_room, m.upper_room_temperature[t])
    # m.room_temperature_rule = pyo.Constraint(m.t, rule=air_temperature)

    # def thermal_mass_temperature_rc(m, t):
    #     # equation correct! tested
    #     if t == 1:
    #         Tm_start = m.BuildingMassTemperatureStartValue
    #     else:
    #         Tm_start = m.T_BuildingMass[t - 1]
    #     T_sup = m.T_outside[t]
    #     PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
    #     PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
    #     u = (m.Htr_is+m.Htr_ms+m.Htr_w) * (m.Htr_is+m.Hve)
    #     v = (m.Htr_is * (m.Hve * T_sup + m.Q_RoomHeating[t] + m.PHI_ia)) / (u-m.Htr_is**2)
    #     C_plus = m.Cm/3600 + 0.5*(m.Htr_ms + m.Htr_em)
    #     C_minus = m.Cm/3600 - 0.5*(m.Htr_ms + m.Htr_em)
    #     w = C_plus * (m.Htr_ms + m.Htr_is + m.Htr_w) * (1-m.Htr_is**2/u)
    #     T_mass = (Tm_start*C_minus + PHI_m + m.Htr_em * m.T_outside[t]) / (C_plus * (1-m.Htr_ms**2/w)) + (m.Htr_ms*(m.Htr_w*m.T_outside[t]+PHI_st)/(w-m.Htr_ms**2)) + (m.Htr_ms*v)/(C_plus*(1-m.Htr_ms**2/w))
    #     return m.T_BuildingMass[t] == T_mass
    # m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)



    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        # Equ. C.2
        PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
        # Equ. C.3
        PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
        # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
        T_sup = m.T_outside[t]
        # Equ. C.5
        PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * \
                    (PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 *
                    (((m.PHI_ia + m.Q_RoomHeating[t]) / m.Hve) + T_sup)) / m.Htr_2
        # Equ. C.4
        return m.T_BuildingMass[t] == \
                (Tm_start * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / \
                ((m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))
    m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)



    def room_temperature_rc(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        # Equ. C.3
        PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (
                0.5 * m.Qi + m.Q_Solar[t]
        )
        # Equ. C.9
        T_m = (m.T_BuildingMass[t] + Tm_start) / 2
        T_sup = m.T_outside[t]
        # Euq. C.10
        T_surface = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (T_sup + (m.PHI_ia + m.Q_RoomHeating[t]) / m.Hve)
                ) / (m.Htr_ms + m.Htr_w + m.Htr_1)
        # Equ. C.11
        T_air = (
                        m.Htr_is * T_surface
                        + m.Hve * T_sup
                        + m.PHI_ia
                        + m.Q_RoomHeating[t]
                ) / (m.Htr_is + m.Hve)
        return m.T_Room[t] == T_air

    m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)
    
    # def surface_temperature(m, t):
    #     if t == 1:
    #         Tm_start = m.BuildingMassTemperatureStartValue
    #     else:
    #         Tm_start = m.T_BuildingMass[t - 1]
    #     T_m = (m.T_BuildingMass[t] + Tm_start) / 2
    #     PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
    #     return m.T_surface[t] == (m.Htr_is*m.T_Room[t] + m.Htr_ms * T_m + m.Htr_w * m.T_outside[t] + PHI_st) / (m.Htr_is+m.Htr_ms+m.Htr_w)
    # m.surface_temperature_rule = pyo.Constraint(m.t, rule=surface_temperature)


    # def thermal_mass_temperature_rc(m, t):
    #     if t == 1:
    #         Tm_start = m.BuildingMassTemperatureStartValue
    #     else:
    #         Tm_start = m.T_BuildingMass[t - 1]
    #     T_m = (m.T_BuildingMass[t] + Tm_start) / 2
        
    #     PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
    #     PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])

    #     T_surface = (m.Htr_is*m.T_Room[t] + m.Htr_ms * T_m + m.Htr_w * m.T_outside[t] + PHI_st) / (m.Htr_is+m.Htr_ms+m.Htr_w)

    #     T_BuildingMass = (Tm_start * (m.Cm/3600 - 0.5*(m.Htr_ms + m.Htr_em)) + PHI_m + m.Htr_ms * T_surface + m.Htr_em * m.T_outside[t]) / \
    #                                     (m.Cm/3600 + 0.5*m.Htr_ms + 0.5 * m.Htr_em)
    #     return m.T_BuildingMass[t] == T_BuildingMass
    # m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)



    def set_lower_room_temp_bound(m, t):
        return m.T_Room[t] >= m.lower_room_temperature[t] 
    m.lower_room_temp_constraint = pyo.Constraint(m.t, rule=set_lower_room_temp_bound)

    def set_upper_room_temp_bound(m, t):
        return m.T_Room[t] <= m.upper_room_temperature[t] 
    m.upper_room_temp_constraint = pyo.Constraint(m.t, rule=set_upper_room_temp_bound)


    def minimize_cost(m):
        rule = sum(m.E_Heating_HP_out[t] * m.ElectricityPrice[t] for t in m.t)
        return rule
    m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory("gurobi")
    solver.options["FeasibilityTol"] = 1e-9  # Tighten feasibility tolerance
    solver.options["OptimalityTol"] = 1e-9  # Improve optimality precision
    solver.options["NumericFocus"] = 2  # Increase numerical stability
    solver.options["BarConvTol"] = 1e-10  # Improve convergence tolerance    
    results = solver.solve(m, tee=True)
    if results.solver.termination_condition == TerminationCondition.optimal:
        solve_status=True
        m.solutions.load_from(results)
        return m, solve_status

    else:
        solve_status=False
        print("model could not be solved!")


def find_6R2C_params(
    model: OperationModel,
    target_indoor_temp: np.array

):
    """

    """

    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=np.arange(1, 8761))  # time

    m.target_indoor_temperature = pyo.Param(m.t, initialize={t: target_indoor_temp[t-1] for t in m.t})
    # set up parameters
    # m.target_heating_demand = pyo.Param(m.t, initialize=)
    # m.target_room_temperature = pyo.Param(m.t, initialize=)
    m.T_outside = pyo.Param(m.t, initialize={t: model.T_outside[t-1] for t in m.t})
    m.Q_Solar = pyo.Param(m.t, initialize={t: model.Q_Solar[t-1] for t in m.t})
    m.CPWater = pyo.Param(initialize=model.CPWater)
    m.ElectricityPrice = pyo.Param(m.t, initialize={t: model.ElectricityPrice[t-1] for t in m.t})

    m.SpaceHeatingHourlyCOP = pyo.Param(m.t, initialize={t: model.SpaceHeatingHourlyCOP[t-1] for t in m.t})
    max_temp, min_temp = model.generate_target_indoor_temperature(temperature_offset=3)
    m.lower_room_temperature = pyo.Param(m.t, initialize={t: min_temp[t-1] for t in m.t})
    m.upper_room_temperature = pyo.Param(m.t, initialize={t: max_temp[t-1] for t in m.t})
    # m.set_temp = pyo.Param(m.t, initialize={t: set_temperature[t-1] for t in m.t})

    
    # building parameters
    m.Am = pyo.Param(initialize=model.Am)
    m.Atot = pyo.Param(initialize=model.Atot)
    m.Qi = pyo.Param(initialize=model.Qi)
    m.Htr_w = pyo.Param(initialize=model.Htr_w)
    m.Htr_em = pyo.Param(initialize=model.Htr_em)
    m.Htr_ms = pyo.Param(initialize=model.Htr_ms)
    m.Htr_is = pyo.Param(initialize=model.Htr_is)
    m.Hve = pyo.Param(initialize=model.Hve)
    # m.Htr_1 = pyo.Param(initialize=)
    # m.Htr_2 = pyo.Param(initialize=)
    # m.Htr_3 = pyo.Param(initialize=)
    m.PHI_ia = pyo.Param(initialize=model.PHI_ia)
    m.Cm = pyo.Param(initialize=model.Cm)
    m.BuildingMassTemperatureStartValue = pyo.Param(initialize=model.BuildingMassTemperatureStartValue)

    # set up variables
    m.T_surface = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.T_Room = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.T_BuildingMass = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)

    # # floor heating
    # Cf is provided in J/K, as it is water we take a multitude of liters per square meter heated are
    possible_Cfs = [model.CPWater * 3600 * model.scenario.building.Af * n for n in range(1, 100)]
    possible_Htr_fs = [x for x in range(1, 10_000, 1)]

    m.Cf = pyo.Var(within=pyo.NonNegativeReals, bounds=(min(possible_Cfs), max(possible_Cfs)))
    m.lambda_Cf = pyo.Var(range(len(possible_Cfs)), bounds=(0,1))
    m.Htr_f = pyo.Var(within=pyo.NonNegativeReals, bounds=(min(possible_Htr_fs), max(possible_Htr_fs)))
    m.lambda_Hf = pyo.Var(range(len(possible_Htr_fs)), bounds=(0,1))


    m.T_floor = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 45))
    m.FloorTemperatureStartValue = pyo.Param(initialize=25)  # 25°C
    

    # contrain Cf and Htr_f to make the problem piecewise linear:
    def contraint_for_Cf_sum(m):
        return m.Cf == sum(possible_Cfs[i] * m.lambda_Cf[i] for i in range(len(possible_Cfs)))
    m.Cf_sum_rule = pyo.Constraint(rule=contraint_for_Cf_sum)

    def contraint_for_Cf_lambda(m):
        return sum(m.lambda_Cf[i] for i in range(len(possible_Cfs))) == 1
    m.Cf_lambda_rule = pyo.Constraint(rule=contraint_for_Cf_lambda)

    def contraint_for_Htr_f_sum(m):
        return m.Htr_f == sum(possible_Htr_fs[i] * m.lambda_Hf[i] for i in range(len(possible_Htr_fs)))
    m.Htr_f_sum_rule = pyo.Constraint(rule=contraint_for_Htr_f_sum)

    def contraint_for_Htr_f_lambda(m):
        return sum(m.lambda_Hf[i] for i in range(len(possible_Htr_fs))) == 1
    m.Htr_f_lambda_rule = pyo.Constraint(rule=contraint_for_Htr_f_lambda)



    def calc_supply_of_space_heating(m, t):
        return m.Q_RoomHeating[t] == m.E_Heating_HP_out[t] * m.SpaceHeatingHourlyCOP[t]
    m.calc_supply_of_space_heating_rule = pyo.Constraint( m.t, rule=calc_supply_of_space_heating )


    def air_temperature(m, t):
        return m.T_Room[t] * (m.Htr_is + m.Hve + m.Htr_f) == (m.T_surface[t]*m.Htr_is + m.Hve*m.T_outside[t] + m.PHI_ia + m.T_floor[t]*m.Htr_f) 
                                
    m.room_temperature_rule = pyo.Constraint(m.t, rule=air_temperature)

    def floor_temperature(m, t):
        if t == 1:
            Tf_start = m.FloorTemperatureStartValue
        else:
            Tf_start = m.T_floor[t - 1]
        return m.T_floor[t] * (m.Cf/3600 + 0.5*m.Htr_f) == (m.Htr_f*m.T_Room[t] + m.Q_RoomHeating[t] + Tf_start * (m.Cf/3600 - 0.5*m.Htr_f)) 
                                
    m.floor_temperature_rule = pyo.Constraint(m.t, rule=floor_temperature)

    def surface_temperature(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        T_m = (m.T_BuildingMass[t] + Tm_start) / 2
        PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
        return m.T_surface[t] == (m.Htr_is*m.T_Room[t] + m.Htr_ms * T_m + m.Htr_w * m.T_outside[t] + PHI_st) / (m.Htr_is+m.Htr_ms+m.Htr_w)
    m.surface_temperature_rule = pyo.Constraint(m.t, rule=surface_temperature)


    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        
        PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
        return m.T_BuildingMass[t] == (Tm_start * (m.Cm/3600 - 0.5*(m.Htr_ms + m.Htr_em)) + PHI_m + m.Htr_ms * m.T_surface[t] + m.Htr_em * m.T_outside[t]) / \
                                        (m.Cm/3600 + 0.5*m.Htr_ms + 0.5 * m.Htr_em)
    m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)


    def set_lower_room_temp_bound(m, t):
        return m.T_Room[t] >= m.lower_room_temperature[t]
    m.lower_room_temp_constraint = pyo.Constraint(m.t, rule=set_lower_room_temp_bound)

    def set_upper_room_temp_bound(m, t):
        return m.T_Room[t] <= m.upper_room_temperature[t] 
    m.upper_room_temp_constraint = pyo.Constraint(m.t, rule=set_upper_room_temp_bound)


    def minimize_cost(m):
        rule = sum(m.E_Heating_HP_out[t] * m.ElectricityPrice[t] for t in m.t)
        return rule
    m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)

    # def minimize_heat_demand_error(m):
    #     rule = sum(m.target_indoor_temperature[t] - m.T_Room[t] for t in m.t)
    #     return rule
    # m.total_operation_cost_rule = pyo.Objective(rule=minimize_heat_demand_error, sense=pyo.minimize)


    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory("gurobi")    
    solver.options["NonConvex"] = 2  # Allow non-convex quadratic constraints

    results = solver.solve(m, tee=True)
    if results.solver.termination_condition == TerminationCondition.optimal:
        solve_status=True
        m.solutions.load_from(results)
        return m, solve_status

    else:
        solve_status=False
        print("model could not be solved!")


def calculate_6R2C_with_specific_params(
    model: OperationModel,
    Htr_f: float,
    Cf: float,
    target_indoor_temp,

):
    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=np.arange(1, 8761))  # time
    m.target_indoor_temperature = pyo.Param(m.t, initialize={t: target_indoor_temp[t-1] for t in m.t})


    # set up parameters
    m.T_outside = pyo.Param(m.t, initialize={t: model.T_outside[t-1] for t in m.t})
    m.Q_Solar = pyo.Param(m.t, initialize={t: model.Q_Solar[t-1] for t in m.t})
    m.CPWater = pyo.Param(initialize=model.CPWater)
    m.ElectricityPrice = pyo.Param(m.t, initialize={t: model.ElectricityPrice[t-1] for t in m.t})

    m.SpaceHeatingHourlyCOP = pyo.Param(m.t, initialize={t: model.SpaceHeatingHourlyCOP[t-1] for t in m.t})
    max_temp, min_temp = model.generate_target_indoor_temperature(temperature_offset=3)
    m.lower_room_temperature = pyo.Param(m.t, initialize={t: min_temp[t-1] for t in m.t})
    m.upper_room_temperature = pyo.Param(m.t, initialize={t: max_temp[t-1] for t in m.t})

    # building parameters
    m.Am = pyo.Param(initialize=model.Am)
    m.Atot = pyo.Param(initialize=model.Atot)
    m.Qi = pyo.Param(initialize=model.Qi)
    m.Htr_w = pyo.Param(initialize=model.Htr_w)
    m.Htr_em = pyo.Param(initialize=model.Htr_em)
    m.Htr_ms = pyo.Param(initialize=model.Htr_ms)
    m.Htr_is = pyo.Param(initialize=model.Htr_is)
    m.Hve = pyo.Param(initialize=model.Hve)
    m.PHI_ia = pyo.Param(initialize=model.PHI_ia)
    m.Cm = pyo.Param(initialize=model.Cm)
    m.BuildingMassTemperatureStartValue = pyo.Param(initialize=model.BuildingMassTemperatureStartValue)

    # set up variables
    m.T_surface = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.T_Room = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.T_BuildingMass = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.reference_Q_RoomHeating = pyo.Param(m.t, initialize={t: model.Q_RoomHeating[t-1] for t in m.t}) # just used to determine summer and winter time for contraint

    # # floor heating

    m.Cf = pyo.Param(initialize=Cf)
    m.Htr_f = pyo.Param(initialize=Htr_f)

    m.T_floor = pyo.Var(m.t, within=pyo.NonNegativeReals, bounds=(20, 45))
    m.FloorTemperatureStartValue = pyo.Param(initialize=25)  # 25°C
    

    def calc_supply_of_space_heating(m, t):
        return m.Q_RoomHeating[t] == m.E_Heating_HP_out[t] * m.SpaceHeatingHourlyCOP[t]
    m.calc_supply_of_space_heating_rule = pyo.Constraint( m.t, rule=calc_supply_of_space_heating )


    def air_temperature(m, t):
        return m.T_Room[t] * (m.Htr_is + m.Hve + m.Htr_f) == (m.T_surface[t]*m.Htr_is + m.Hve*m.T_outside[t] + m.PHI_ia + m.T_floor[t]*m.Htr_f)             
    m.room_temperature_rule = pyo.Constraint(m.t, rule=air_temperature)

    def floor_temperature(m, t):
        if t == 1:
            Tf_start = m.FloorTemperatureStartValue
        else:
            Tf_start = m.T_floor[t - 1]
        return m.T_floor[t] * (m.Cf/3600 + 0.5*m.Htr_f) == (m.Htr_f*m.T_Room[t] + m.Q_RoomHeating[t] + Tf_start * (m.Cf/3600 - 0.5*m.Htr_f))                
    m.floor_temperature_rule = pyo.Constraint(m.t, rule=floor_temperature)

    def surface_temperature(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        T_m = (m.T_BuildingMass[t] + Tm_start) / 2
        PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
        return m.T_surface[t] == (m.Htr_is*m.T_Room[t] + m.Htr_ms * T_m + m.Htr_w * m.T_outside[t] + PHI_st) / (m.Htr_is+m.Htr_ms+m.Htr_w)
    m.surface_temperature_rule = pyo.Constraint(m.t, rule=surface_temperature)


    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            Tm_start = m.BuildingMassTemperatureStartValue
        else:
            Tm_start = m.T_BuildingMass[t - 1]
        
        PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
        return m.T_BuildingMass[t] == (Tm_start * (m.Cm/3600 - 0.5*(m.Htr_ms + m.Htr_em)) + PHI_m + m.Htr_ms * m.T_surface[t] + m.Htr_em * m.T_outside[t]) / \
                                        (m.Cm/3600 + 0.5*m.Htr_ms + 0.5 * m.Htr_em)
    m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)


    def set_lower_room_temp_bound(m, t):
        if m.reference_Q_RoomHeating[t] > 0:
            return m.T_Room[t] >= m.target_indoor_temperature[t] - 0.3
        else:
            return m.T_Room[t] >= m.lower_room_temperature[t]
    m.lower_room_temp_constraint = pyo.Constraint(m.t, rule=set_lower_room_temp_bound)

    def set_upper_room_temp_bound(m, t):
        if m.reference_Q_RoomHeating[t] > 0:
            return m.T_Room[t] <= m.target_indoor_temperature[t] + 0.3
        else:
            return m.T_Room[t] <= m.upper_room_temperature[t] 
    m.upper_room_temp_constraint = pyo.Constraint(m.t, rule=set_upper_room_temp_bound)


    def minimize_cost(m):
        rule = sum(m.E_Heating_HP_out[t] * m.ElectricityPrice[t] for t in m.t)
        return rule
    m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory("gurobi")    
    results = solver.solve(m, tee=True)
    if results.solver.termination_condition == TerminationCondition.optimal:
        solve_status=True
        m.solutions.load_from(results)
        return m, solve_status

    else:
        solve_status=False
        print("model could not be solved!")
        return None, solve_status

if __name__ == "__main__":
    pass