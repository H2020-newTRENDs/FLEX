
from .abstract import AbstractOperationModel
from functools import wraps
import time
import pyomo.environ as pyo


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        print("function >>{}<< time for execution: {}".format(func.__name__, t_end - t_start))
        return result
    return wrapper


class OptOperationModel(AbstractOperationModel):

    @performance_counter
    def create_abstract_model(self):
        CPWater = 4200 / 3600
        m = pyo.AbstractModel()
        m.t = pyo.Set()
        # -------------
        # 1. Parameters
        # -------------

        # price
        m.electricity_price = pyo.Param(m.t, mutable=True)  # C/Wh
        # Feed in Tariff of Photovoltaic
        m.FiT = pyo.Param(m.t, mutable=True)  # C/Wh
        # solar gains:
        m.Q_Solar = pyo.Param(m.t, mutable=True)  # W
        # outside temperature
        m.T_outside = pyo.Param(m.t, mutable=True)  # °C
        # COP of heatpump
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t, mutable=True)
        # COP of cooling
        m.CoolingCOP = pyo.Param(m.t, mutable=True)
        # electricity load profile
        m.BaseLoadProfile = pyo.Param(m.t, mutable=True)
        # PV profile
        m.PhotovoltaicProfile = pyo.Param(m.t, mutable=True)
        # HotWater
        m.HotWaterProfile = pyo.Param(m.t, mutable=True)
        m.HotWaterHourlyCOP = pyo.Param(m.t, mutable=True)

        # Smart Technologies
        m.DayHour = pyo.Param(m.t, mutable=True)

        # building data:
        m.Am = pyo.Param(mutable=True)
        m.Atot = pyo.Param(mutable=True)
        m.Qi = pyo.Param(mutable=True)
        m.Htr_w = pyo.Param(mutable=True)
        m.Htr_em = pyo.Param(mutable=True)
        m.Htr_3 = pyo.Param(mutable=True)
        m.Htr_1 = pyo.Param(mutable=True)
        m.Htr_2 = pyo.Param(mutable=True)
        m.Hve = pyo.Param(mutable=True)
        m.Htr_ms = pyo.Param(mutable=True)
        m.Htr_is = pyo.Param(mutable=True)
        m.PHI_ia = pyo.Param(mutable=True)
        m.Cm = pyo.Param(mutable=True)
        m.BuildingMassTemperatureStartValue = pyo.Param(mutable=True)

        # Heating Tank data
        # Mass of water in tank
        m.M_WaterTank_heating = pyo.Param(mutable=True)
        # Surface of Tank in m2
        m.A_SurfaceTank_heating = pyo.Param(mutable=True)
        # insulation of tank, for calc of losses
        m.U_ValueTank_heating = pyo.Param(mutable=True)
        m.T_TankStart_heating = pyo.Param(mutable=True)
        # surrounding temp of tank
        m.T_TankSurrounding_heating = pyo.Param(mutable=True)

        # DHW Tank data
        # Mass of water in tank
        m.M_WaterTank_DHW = pyo.Param(mutable=True)
        # Surface of Tank in m2
        m.A_SurfaceTank_DHW = pyo.Param(mutable=True)
        # insulation of tank, for calc of losses
        m.U_ValueTank_DHW = pyo.Param(mutable=True)
        m.T_TankStart_DHW = pyo.Param(mutable=True)
        # surrounding temp of tank
        m.T_TankSurrounding_DHW = pyo.Param(mutable=True)

        # heat pump
        m.SpaceHeating_HeatPumpMaximalThermalPower = pyo.Param(mutable=True)

        # Battery data
        m.ChargeEfficiency = pyo.Param(mutable=True)
        m.DischargeEfficiency = pyo.Param(mutable=True)

        # ----------------------------
        # 2. Variables
        # ----------------------------
        # Variables SpaceHeating
        m.Q_HeatingTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_HeatingTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_HeatingTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_room_heating = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Variables DHW
        m.Q_DHWTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_DHWTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_DHWTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_DHW_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_DHWTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Variable space cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Temperatures (room and thermal mass)
        m.T_room = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Tm_t = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Grid variables
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Grid2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # PV variables
        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Electric Load and Electricity fed back to the grid
        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Feedin = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Battery variables
        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # -------------------------------------
        # 3. Constraints:
        # -------------------------------------

        # (1) grid balance
        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t]

        m.calc_UseOfGrid = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (2) PV balance
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] == m.PhotovoltaicProfile[t]

        m.calc_UseOfPV = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (3) Feed in
        def calc_SumOfFeedin(m, t):
            return m.Feedin[t] == m.PV2Grid[t]

        m.calc_SumOfFeedin = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (4) Sum of Loads
        def calc_SumOfLoads_with_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.Q_Heating_HP_out[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_RoomCooling[t] / m.CoolingCOP[t] \
                   + m.Q_DHW_HP_out[t] / m.HotWaterHourlyCOP[t]

        m.calc_SumOfLoads_with_cooling = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

        def calc_SumOfLoads_without_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.Q_Heating_HP_out[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_DHW_HP_out[t] / m.HotWaterHourlyCOP[t]

        m.calc_SumOfLoads_without_cooling = pyo.Constraint(m.t, rule=calc_SumOfLoads_without_cooling)

        # (5) load coverage
        def calc_SupplyOfLoads(m, t):
            return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

        m.calc_SupplyOfLoads = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # Battery:
        # (6) Battery charge
        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t]  # * Household.Battery.Grid2Battery

        m.calc_BatCharge = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (7) Battery discharge
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.BatDischarge[t] == m.Bat2Load[t]

        m.calc_BatDischarge = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8) Battery SOC
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.DischargeEfficiency))

        m.calc_BatSoC = pyo.Constraint(m.t, rule=calc_BatSoC)

        # Heating Tank:
        # (9) energy in the tank
        def tank_energy_heating(m, t):
            if t == 1:
                return m.E_HeatingTank[t] == CPWater * m.M_WaterTank_heating * (273.15 + m.T_TankStart_heating)
            else:
                return m.E_HeatingTank[t] == m.E_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] + \
                       m.Q_HeatingElement[t] - m.U_ValueTank_heating * m.A_SurfaceTank_heating * (
                               (m.E_HeatingTank[t] / (m.M_WaterTank_heating * CPWater)) -
                               (m.T_TankSurrounding_heating + 273.15))

        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

        # (10) room heating
        def room_heating(m, t):
            return m.Q_room_heating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t]

        m.room_heating_constraint = pyo.Constraint(m.t, rule=room_heating)

        # (11) supply for space heating
        def calc_supply_of_space_heating(m, t):
            return m.Q_HeatingTank_bypass[t] + m.Q_HeatingTank_in[t] == m.Q_Heating_HP_out[t]

        m.calc_use_of_HP_power_DHW_constraint = pyo.Constraint(m.t, rule=calc_supply_of_space_heating)

        # DHW:
        # (12) energy in DHW tank
        def tank_energy_DHW(m, t):
            if t == 1:
                return m.E_DHWTank[t] == CPWater * m.M_WaterTank_DHW * (273.15 + m.T_TankStart_DHW)
            else:
                return m.E_DHWTank[t] == m.E_DHWTank[t - 1] - m.Q_DHWTank_out[t] + m.Q_DHWTank_in[t] - \
                       m.U_ValueTank_DHW * m.A_SurfaceTank_DHW * (
                               m.E_DHWTank[t] / (m.M_WaterTank_DHW * CPWater) - (m.T_TankSurrounding_DHW + 273.15)
                       )

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (13) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.calc_SupplyOfDHW = pyo.Constraint(m.t, rule=calc_hot_water_profile)

        # (14) supply for the hot water
        def calc_supply_of_DHW(m, t):
            return m.Q_DHWTank_bypass[t] + m.Q_DHWTank_in[t] == m.Q_DHW_HP_out[t]

        m.bypass_DHW_constraint = pyo.Constraint(m.t, rule=calc_supply_of_DHW)

        # Heat Pump
        # (13) restrict heat pump power
        def constrain_heating_max_power(m, t):
            return m.Q_DHW_HP_out[t] + m.Q_Heating_HP_out[t] <= m.SpaceHeating_HeatPumpMaximalThermalPower

        m.max_HP_power_constraint = pyo.Constraint(m.t, rule=constrain_heating_max_power)

        # ----------------------------------------------------
        # 6. State transition: Building temperature (RC-Model)
        # ----------------------------------------------------

        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                # Equ. C.2
                PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * (
                            PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                            ((m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.BuildingMassTemperatureStartValue *
                                     ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / (
                               (m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))

            else:
                # Equ. C.2
                PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
                T_sup = m.T_outside[t]
                # Equ. C.5
                PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * (
                            PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                            ((m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
                # Equ. C.4
                return m.Tm_t[t] == (m.Tm_t[t - 1] * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / (
                        (m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.BuildingMassTemperatureStartValue) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                        T_sup + (m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / m.Hve)) / (
                              m.Htr_ms + m.Htr_w + m.Htr_1)
                # Equ. C.11
                T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / (
                        m.Htr_is + m.Hve)
                return m.T_room[t] == T_air
            else:
                # Equ. C.3
                PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
                # Equ. C.9
                T_m = (m.Tm_t[t] + m.Tm_t[t - 1]) / 2
                T_sup = m.T_outside[t]
                # Euq. C.10
                T_s = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 * (
                        T_sup + (m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / m.Hve)) / (
                              m.Htr_ms + m.Htr_w + m.Htr_1)
                # Equ. C.11
                T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / (
                        m.Htr_is + m.Hve)
                return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        # ------------
        # 7. Objective
        # ------------

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.electricity_price[t] - m.Feedin[t] * m.FiT[t] for t in m.t)
            return rule

        m.Objective = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)
        # return the household model
        return m

    @performance_counter
    def update_instance(self, total_input, instance):
        """
        Function takes the instance and updates its parameters as well as fixes various parameters to 0 if they are
        not used because there is no storage available for example. Solves the instance and returns the solved instance.
        """

        input_parameters = total_input["input_parameters"]
        household = total_input["Household"]
        Environment = total_input["Environment"]
        HeatingTargetTemperature = total_input["HeatingTargetTemperature"]
        CoolingTargetTemperature = total_input["CoolingTargetTemperature"]
        # print("Household ID: " + str(household["ID"]))
        # print("Environment ID: " + str(Environment["ID"]))
        CPWater = 4200 / 3600
        # update the instance
        for t in range(1, len(HeatingTargetTemperature) + 1):
            # time dependent parameters are updated:
            instance.electricity_price[t] = input_parameters[None]["ElectricityPrice"][t]
            instance.FiT[t] = input_parameters[None]["FiT"][t]
            instance.Q_Solar[t] = input_parameters[None]["Q_Solar"][t]
            instance.PhotovoltaicProfile[t] = input_parameters[None]["PhotovoltaicProfile"][t]
            instance.T_outside[t] = input_parameters[None]["T_outside"][t]
            instance.SpaceHeatingHourlyCOP[t] = input_parameters[None]["SpaceHeatingHourlyCOP"][t]
            instance.CoolingCOP[t] = input_parameters[None]["CoolingCOP"][t]
            instance.BaseLoadProfile[t] = input_parameters[None]["BaseLoadProfile"][t]
            instance.HotWaterProfile[t] = input_parameters[None]["HotWaterProfile"][t]
            instance.HotWaterHourlyCOP[t] = input_parameters[None]["HotWaterHourlyCOP"][t]
            instance.DayHour[t] = input_parameters[None]["DayHour"][t]

            # Boundaries:
            # Heating
            instance.Q_HeatingElement[t].setub(household["SpaceHeating_HeatingElementPower"])
            # room heating is handled in if cases
            # Temperatures for RC model
            instance.T_room[t].setlb(HeatingTargetTemperature[t - 1])

            instance.Tm_t[t].setub(100)  # so it wont be infeasible when no cooling
            # maximum Grid load
            instance.Grid[t].setub(household["Building_MaximalGridPower"])
            instance.Grid2Load[t].setub(household["Building_MaximalGridPower"])
            # maximum load of house and electricity fed back to the grid
            instance.Load[t].setub(household["Building_MaximalGridPower"])
            instance.Feedin[t].setub(household["Building_MaximalGridPower"])

        # special cases:
        # Room Cooling:
        if household["SpaceCooling_SpaceCoolingPower"] == 0:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.Q_RoomCooling[t].fix(0)
                instance.T_room[t].setub(100)
            instance.calc_SumOfLoads_without_cooling.activate()
            instance.calc_SumOfLoads_with_cooling.deactivate()

        else:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.Q_RoomCooling[t].fixed = False
                instance.Q_RoomCooling[t].setub(household["SpaceCooling_SpaceCoolingPower"])
                instance.T_room[t].setub(CoolingTargetTemperature[t - 1])
            instance.calc_SumOfLoads_without_cooling.deactivate()
            instance.calc_SumOfLoads_with_cooling.activate()

        # Thermal storage Heating
        if household["SpaceHeating_TankSize"] == 0:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.E_HeatingTank[t].fix(0)
                instance.Q_HeatingTank_out[t].fix(0)
                instance.Q_HeatingTank_in[t].fix(0)

                instance.Q_Heating_HP_out[t].setub(
                    household["SpaceHeating_HeatPumpMaximalThermalPower"] + household[
                        "SpaceHeating_HeatingElementPower"]
                )  # W

            instance.tank_energy_rule_heating.deactivate()
        else:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.E_HeatingTank[t].fixed = False
                instance.Q_HeatingTank_out[t].fixed = False
                instance.Q_HeatingTank_in[t].fixed = False
                instance.E_HeatingTank[t].setlb(
                    CPWater * float(household["SpaceHeating_TankSize"]) * (
                            273.15 + float(household["SpaceHeating_TankMinimalTemperature"])
                    )
                )
                instance.E_HeatingTank[t].setub(
                    CPWater * household["SpaceHeating_TankSize"] * (
                            273.15 + household["SpaceHeating_TankMaximalTemperature"])
                )
                instance.Q_Heating_HP_out[t].setub(
                    household["SpaceHeating_HeatPumpMaximalThermalPower"] + household[
                        "SpaceHeating_HeatingElementPower"]
                )  # W

            instance.tank_energy_rule_heating.activate()

        # Thermal storage DHW
        if household["DHW_TankSize"] == 0:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.E_DHWTank[t].fix(0)
                instance.Q_DHWTank_out[t].fix(0)
                instance.Q_DHWTank_in[t].fix(0)

                instance.Q_DHW_HP_out[t].setub(household["SpaceHeating_HeatPumpMaximalThermalPower"])  # W
            instance.tank_energy_rule_DHW.deactivate()
        else:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.E_DHWTank[t].fixed = False
                instance.Q_DHWTank_out[t].fixed = False
                instance.Q_DHWTank_in[t].fixed = False
                instance.E_DHWTank[t].setlb(
                    CPWater * float(household["DHW_TankSize"]) *
                    (273.15 + float(household["DHW_TankMinimalTemperature"]))
                )
                instance.E_DHWTank[t].setub(
                    CPWater * household["DHW_TankSize"] *
                    (273.15 + household["DHW_TankMaximalTemperature"])
                )
                instance.Q_DHW_HP_out[t].setub(household["SpaceHeating_HeatPumpMaximalThermalPower"])  # W

            instance.tank_energy_rule_DHW.activate()

        # Battery
        if household["Battery_Capacity"] == 0:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                # fix the parameters to 0
                instance.Grid2Bat[t].fix(0)
                instance.Bat2Load[t].fix(0)
                instance.BatSoC[t].fix(0)
                instance.BatCharge[t].fix(0)
                instance.BatDischarge[t].fix(0)
                instance.PV2Bat[t].fix(0)
            instance.calc_BatCharge.deactivate()
            instance.calc_BatDischarge.deactivate()
            instance.calc_BatSoC.deactivate()
        else:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.Grid2Bat[t].fixed = False
                instance.Bat2Load[t].fixed = False
                instance.BatSoC[t].fixed = False
                instance.BatCharge[t].fixed = False
                instance.BatDischarge[t].fixed = False
                instance.PV2Bat[t].fixed = False
                # set upper bounds
                instance.Grid2Bat[t].setub(household["Building_MaximalGridPower"])

                instance.Bat2Load[t].setub(household["Battery_MaxDischargePower"])
                instance.BatSoC[t].setub(household["Battery_Capacity"])
                instance.BatCharge[t].setub(household["Battery_MaxChargePower"])
                instance.BatDischarge[t].setub(household["Battery_MaxDischargePower"])
            instance.calc_BatCharge.activate()
            instance.calc_BatDischarge.activate()
            instance.calc_BatSoC.activate()

        # PV
        if household["PV_PVPower"] == 0:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                instance.PV2Load[t].fix(0)
                instance.PV2Bat[t].fix(0)
                instance.PV2Grid[t].fix(0)
            instance.calc_UseOfPV.deactivate()

        else:
            for t in range(1, len(HeatingTargetTemperature) + 1):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.PV2Load[t].fixed = False
                instance.PV2Bat[t].fixed = False
                instance.PV2Grid[t].fixed = False
                # set upper bounds
                instance.PV2Load[t].setub(household["Building_MaximalGridPower"])
                instance.PV2Bat[t].setub(household["Building_MaximalGridPower"])
                instance.PV2Grid[t].setub(household["Building_MaximalGridPower"])
            instance.calc_UseOfPV.activate()

        # update time independent parameters
        # building parameters:
        instance.Am = input_parameters[None]["Am"][None]
        instance.Atot = input_parameters[None]["Atot"][None]
        instance.Qi = input_parameters[None]["Qi"][None]
        instance.Htr_w = input_parameters[None]["Htr_w"][None]
        instance.Htr_em = input_parameters[None]["Htr_em"][None]
        instance.Htr_3 = input_parameters[None]["Htr_3"][None]
        instance.Htr_1 = input_parameters[None]["Htr_1"][None]
        instance.Htr_2 = input_parameters[None]["Htr_2"][None]
        instance.Hve = input_parameters[None]["Hve"][None]
        instance.Htr_ms = input_parameters[None]["Htr_ms"][None]
        instance.Htr_is = input_parameters[None]["Htr_is"][None]
        instance.PHI_ia = input_parameters[None]["PHI_ia"][None]
        instance.Cm = input_parameters[None]["Cm"][None]
        instance.BuildingMassTemperatureStartValue = input_parameters[None]["BuildingMassTemperatureStartValue"][None]
        # Battery parameters
        instance.ChargeEfficiency = input_parameters[None]["ChargeEfficiency"][None]
        instance.DischargeEfficiency = input_parameters[None]["DischargeEfficiency"][None]
        # Thermal storage heating parameters
        instance.T_TankStart_heating = input_parameters[None]["T_TankStart_heating"][None]
        instance.M_WaterTank_heating = input_parameters[None]["M_WaterTank_heating"][None]
        instance.U_ValueTank_heating = input_parameters[None]["U_ValueTank_heating"][None]
        instance.T_TankSurrounding_heating = input_parameters[None]["T_TankSurrounding_heating"][None]
        instance.A_SurfaceTank_heating = input_parameters[None]["A_SurfaceTank_heating"][None]
        # Thermal storage DHW parameters
        instance.T_TankStart_DHW = input_parameters[None]["T_TankStart_DHW"][None]
        instance.M_WaterTank_DHW = input_parameters[None]["M_WaterTank_DHW"][None]
        instance.U_ValueTank_DHW = input_parameters[None]["U_ValueTank_DHW"][None]
        instance.T_TankSurrounding_DHW = input_parameters[None]["T_TankSurrounding_DHW"][None]
        instance.A_SurfaceTank_DHW = input_parameters[None]["A_SurfaceTank_DHW"][None]
        # HP
        instance.SpaceHeating_HeatPumpMaximalThermalPower = \
        input_parameters[None]["SpaceHeating_HeatPumpMaximalThermalPower"][None]
        return instance





