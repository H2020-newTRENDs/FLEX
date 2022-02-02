from _Refactor.models.operation.abstract import AbstractOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector
from functools import wraps
import time
import pyomo.environ as pyo
import numpy as np


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

    def creat_Dict(self, value_list: list) -> dict:
        Dictionary = {}
        for index, value in enumerate(value_list, start=1):
            Dictionary[index] = value
        return Dictionary

    def create_pyomo_dict(self) -> dict:
        pyomo_dict = {
            None: {"t": {
                None: np.arange(1, 8761)},
                "ElectricityPrice": self.creat_Dict(self.scenario.electricityprice_class.electricity_price),  # C/Wh
                "FiT": self.creat_Dict(self.scenario.feedintariff_class.feed_in_tariff),  # C/Wh
                "Q_Solar": self.creat_Dict(self.calculate_solar_gains()),  # W
                "PhotovoltaicProfile": self.creat_Dict(self.scenario.pv_class.power),
                "T_outside": self.creat_Dict(self.scenario.region_class.temperature),  # °C
                "SpaceHeatingHourlyCOP": self.creat_Dict(
                    self.COP_HP(outside_temperature=self.scenario.region_class.temperature,
                                supply_temperature=self.scenario.boiler_class.heating_supply_temperature,
                                efficiency=self.scenario.boiler_class.carnot_efficiency_factor,
                                source=self.scenario.boiler_class.name)),
                "BaseLoadProfile": self.creat_Dict(self.scenario.electricitydemand_class.electricity_demand),  # W
                "HotWaterProfile": self.creat_Dict(self.scenario.hotwaterdemand_class.hot_water_demand),
                "HotWaterHourlyCOP": self.creat_Dict(
                    self.COP_HP(outside_temperature=self.scenario.region_class.temperature,
                                supply_temperature=self.scenario.boiler_class.hot_water_supply_temperature,
                                efficiency=self.scenario.boiler_class.carnot_efficiency_factor,
                                source=self.scenario.boiler_class.name)),
                "DayHour": self.creat_Dict(self.day_hour),
                "CoolingCOP": {None: self.scenario.airconditioner_class.efficiency},
                "ChargeEfficiency": {None: self.scenario.battery_class.charge_efficiency},
                "DischargeEfficiency": {None: self.scenario.battery_class.discharge_efficiency},
                "Am": {None: self.calculate_Am(Am_factor=self.scenario.building_class.Am_factor,
                                               Af=self.scenario.building_class.Af)},
                "Atot": {None: self.calculate_Atot(Af=self.scenario.building_class.Af)},
                "Qi": {None: self.calculate_Qi(specific_internal_gains=self.scenario.building_class.internal_gains,
                                               Af=self.scenario.building_class.Af)},
                "Htr_w": {None: self.scenario.building_class.Htr_w},
                "Htr_em": {None: self.calculate_Htr_em(Hop=self.scenario.building_class.Hop,
                                                       Am_factor=self.scenario.building_class.Am_factor,
                                                       Af=self.scenario.building_class.Af)},
                "Htr_3": {None: self.calculate_Htr_3(Hve=self.scenario.building_class.Hve,
                                                     Af=self.scenario.building_class.Af,
                                                     Htr_w=self.scenario.building_class.Htr_w,
                                                     Am_factor=self.scenario.building_class.Am_factor)},
                "Htr_1": {None: self.calculate_Htr_1(Hve=self.scenario.building_class.Hve,
                                                     Af=self.scenario.building_class.Af)},
                "Htr_2": {None: self.calculate_Htr_2(Hve=self.scenario.building_class.Hve,
                                                     Af=self.scenario.building_class.Af,
                                                     Htr_w=self.scenario.building_class.Htr_w)},
                "Hve": {None: self.scenario.building_class.Hve},
                "Htr_ms": {None: self.calculate_Htr_ms(Am_factor=self.scenario.building_class.Am_factor,
                                                       Af=self.scenario.building_class.Af)},
                "Htr_is": {None: self.calculate_Htr_is(Af=self.scenario.building_class.Af)},
                "PHI_ia": {
                    None: self.calculate_PHI_ia(specific_internal_gains=self.scenario.building_class.internal_gains,
                                                Af=self.scenario.building_class.Af)},
                "Cm": {None: self.calculate_Cm(CM_factor=self.scenario.building_class.CM_factor,
                                               Af=self.scenario.building_class.Af)},
                "BuildingMassTemperatureStartValue": {
                    None: self.scenario.building_class.building_mass_temperature_start},
                # space heating tank
                "T_TankStart_heating": {None: self.scenario.spaceheatingtank_class.temperature_start},
                # min temp is start temp
                "M_WaterTank_heating": {None: self.scenario.spaceheatingtank_class.size},
                "U_ValueTank_heating": {None: self.scenario.spaceheatingtank_class.loss},
                "T_TankSurrounding_heating": {None: self.scenario.spaceheatingtank_class.temperature_surrounding},
                "A_SurfaceTank_heating": {None: self.scenario.spaceheatingtank_class.surface_area},
                # DHW Tank
                "T_TankStart_DHW": {None: self.scenario.hotwatertank_class.temperature_start},
                # min temp is start temp
                "M_WaterTank_DHW": {None: self.scenario.hotwatertank_class.size},
                "U_ValueTank_DHW": {None: self.scenario.hotwatertank_class.loss},
                "T_TankSurrounding_DHW": {None: self.scenario.hotwatertank_class.temperature_surrounding},
                "A_SurfaceTank_DHW": {None: self.scenario.hotwatertank_class.surface_area},
                "SpaceHeating_HeatPumpMaximalThermalPower": {
                    None: self.scenario.boiler_class.thermal_power_max}
            }
        }
        return pyomo_dict

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
        m.CoolingCOP = pyo.Param(mutable=True)  # single value because it is not dependent on time
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
        m.Feed2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

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

        m.UseOfGrid_rule = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (2) PV balance
        def calc_UseOfPV(m, t):
            return m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] == m.PhotovoltaicProfile[t]

        m.UseOfPV_rule = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (3) Feed in
        def calc_SumOfFeedin(m, t):
            return m.Feed2Grid[t] == m.PV2Grid[t]

        m.SumOfFeedin_rule = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (4) Sum of Loads
        def calc_SumOfLoads_with_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.Q_Heating_HP_out[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_RoomCooling[t] / m.CoolingCOP \
                   + m.Q_DHW_HP_out[t] / m.HotWaterHourlyCOP[t]

        m.SumOfLoads_with_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

        def calc_SumOfLoads_without_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.Q_Heating_HP_out[t] / m.SpaceHeatingHourlyCOP[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_DHW_HP_out[t] / m.HotWaterHourlyCOP[t]

        m.SumOfLoads_without_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_without_cooling)

        # (5) load coverage
        def calc_SupplyOfLoads(m, t):
            return m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

        m.SupplyOfLoads_rule = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # Battery:
        # (6) Battery charge
        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t]  # * Household.Battery.Grid2Battery

        m.BatCharge_rule = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (7) Battery discharge
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.BatDischarge[t] == m.Bat2Load[t]

        m.BatDischarge_rule = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (8) Battery SOC
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.DischargeEfficiency))

        m.BatSoC_rule = pyo.Constraint(m.t, rule=calc_BatSoC)

        # Heating Tank:
        # (9) energy in the tank
        def tank_energy_heating(m, t):
            if t == 1:
                return m.E_HeatingTank[t] == CPWater * m.M_WaterTank_heating * (273.15 + m.T_TankStart_heating) - \
                       m.Q_HeatingTank_out[t]
            else:
                return m.E_HeatingTank[t] == m.E_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] + \
                       m.Q_HeatingElement[t] - m.U_ValueTank_heating * m.A_SurfaceTank_heating * (
                               (m.E_HeatingTank[t] / (m.M_WaterTank_heating * CPWater)) -
                               (m.T_TankSurrounding_heating + 273.15))

        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

        # (10) room heating
        def room_heating(m, t):
            return m.Q_room_heating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t]

        m.room_heating_rule = pyo.Constraint(m.t, rule=room_heating)

        # (11) supply for space heating
        def calc_supply_of_space_heating(m, t):
            return m.Q_HeatingTank_bypass[t] + m.Q_HeatingTank_in[t] == m.Q_Heating_HP_out[t]

        m.calc_use_of_HP_power_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_space_heating)

        # DHW:
        # (12) energy in DHW tank
        def tank_energy_DHW(m, t):
            if t == 1:
                return m.E_DHWTank[t] == CPWater * m.M_WaterTank_DHW * (273.15 + m.T_TankStart_DHW) - m.Q_DHWTank_out[t]
            else:
                return m.E_DHWTank[t] == m.E_DHWTank[t - 1] - m.Q_DHWTank_out[t] + m.Q_DHWTank_in[t] - \
                       m.U_ValueTank_DHW * m.A_SurfaceTank_DHW * (
                               m.E_DHWTank[t] / (m.M_WaterTank_DHW * CPWater) - (m.T_TankSurrounding_DHW + 273.15)
                       )

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (13) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.SupplyOfDHW_rule = pyo.Constraint(m.t, rule=calc_hot_water_profile)

        # (14) supply for the hot water
        def calc_supply_of_DHW(m, t):
            return m.Q_DHWTank_bypass[t] + m.Q_DHWTank_in[t] == m.Q_DHW_HP_out[t]

        m.bypass_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW)

        # Heat Pump
        # (13) restrict heat pump power
        def constrain_heating_max_power(m, t):
            return m.Q_DHW_HP_out[t] + m.Q_Heating_HP_out[t] <= m.SpaceHeating_HeatPumpMaximalThermalPower

        m.max_HP_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power)

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
            rule = sum(m.Grid[t] * m.electricity_price[t] - m.Feed2Grid[t] * m.FiT[t] for t in m.t)
            return rule

        m.Objective_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)
        # return the household model
        return m

    @performance_counter
    def update_instance(self, instance):
        """
        Function takes the instance and updates its parameters as well as fixes various parameters to 0 if they are
        not used because there is no storage available for example. Solves the instance and returns the solved instance.
        """
        CPWater = 4200 / 3600
        # update the instance
        solar_gains = self.calculate_solar_gains()

        space_heating_hourly_COP = self.COP_HP(outside_temperature=np.array(self.scenario.region_class.temperature),
                                               supply_temperature=self.scenario.boiler_class.heating_supply_temperature,
                                               efficiency=self.scenario.boiler_class.carnot_efficiency_factor,
                                               source=self.scenario.boiler_class.name)
        hot_water_hourly_COP = self.COP_HP(outside_temperature=np.array(self.scenario.region_class.temperature),
                                           supply_temperature=self.scenario.boiler_class.hot_water_supply_temperature,
                                           efficiency=self.scenario.boiler_class.carnot_efficiency_factor,
                                           source=self.scenario.boiler_class.name)
        for t in range(1, 8761):
            index = t - 1  # pyomo starts at index 1
            # time dependent parameters are updated:
            instance.electricity_price[t] = self.scenario.electricityprice_class.electricity_price[index]
            instance.FiT[t] = self.scenario.feedintariff_class.feed_in_tariff[index]
            instance.Q_Solar[t] = solar_gains[index]
            instance.PhotovoltaicProfile[t] = self.scenario.pv_class.power[index]
            instance.T_outside[t] = self.scenario.region_class.temperature[index]
            instance.SpaceHeatingHourlyCOP[t] = space_heating_hourly_COP[index]


            input_parameters = 1
            HeatingTargetTemperature = 1
            # instance.CoolingCOP[t] = self.household.airconditioner_class.efficiency  # is a single value, no index
            instance.BaseLoadProfile[t] = self.scenario.electricitydemand_class.electricity_demand[index]
            instance.HotWaterProfile[t] = self.scenario.hotwaterdemand_class.hot_water_demand[index]
            instance.HotWaterHourlyCOP[t] = hot_water_hourly_COP[index]
            instance.DayHour[t] = self.day_hour[index]

            # Boundaries:
            # Heating
            instance.Q_HeatingElement[t].setub(self.scenario.boiler_class.heating_element_power)
            # room heating is handled in if cases
            # Temperatures for RC model
            instance.T_room[t].setlb(self.scenario.behavior_class.indoor_set_temperature_min[index])

            instance.Tm_t[t].setub(100)  # so it wont be infeasible when no cooling
            # maximum Grid load
            instance.Grid[t].setub(self.scenario.building_class.grid_power_max)
            instance.Grid2Load[t].setub(self.scenario.building_class.grid_power_max)
            # maximum load of house and electricity fed back to the grid
            instance.Load[t].setub(self.scenario.building_class.grid_power_max)
            instance.Feed2Grid[t].setub(self.scenario.building_class.grid_power_max)

        # special cases:
        # Room Cooling:
        if self.scenario.airconditioner_class.power == 0:
            for t in range(1, 8761):
                instance.Q_RoomCooling[t].fix(0)
                instance.T_room[t].setub(100)
            instance.SumOfLoads_without_cooling_rule.activate()
            instance.SumOfLoads_with_cooling_rule.deactivate()

        else:
            for t in range(1, 8761):
                instance.Q_RoomCooling[t].fixed = False
                instance.Q_RoomCooling[t].setub(self.scenario.airconditioner_class.power)
                instance.T_room[t].setub(self.scenario.behavior_class.indoor_set_temperature_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.deactivate()
            instance.SumOfLoads_with_cooling_rule.activate()

        # Thermal storage Heating
        if self.scenario.spaceheatingtank_class.size == 0:
            for t in range(1, 8761):
                instance.E_HeatingTank[t].fix(0)
                instance.Q_HeatingTank_out[t].fix(0)
                instance.Q_HeatingTank_in[t].fix(0)

                instance.Q_Heating_HP_out[t].setub(self.scenario.boiler_class.thermal_power_max +
                                                   self.scenario.boiler_class.heating_element_power)

            instance.tank_energy_rule_heating.deactivate()
        else:
            for t in range(1, 8761):
                instance.E_HeatingTank[t].fixed = False
                instance.Q_HeatingTank_out[t].fixed = False
                instance.Q_HeatingTank_in[t].fixed = False


                instance.E_HeatingTank[t].setlb(
                    CPWater * self.scenario.spaceheatingtank_class.size *
                    (273.15 + self.scenario.spaceheatingtank_class.temperature_min)
                )

                instance.E_HeatingTank[t].setub(
                    CPWater * self.scenario.spaceheatingtank_class.size *
                    (273.15 + self.scenario.spaceheatingtank_class.temperature_max)
                )
                instance.Q_Heating_HP_out[t].setub(
                    self.scenario.boiler_class.thermal_power_max + self.scenario.boiler_class.heating_element_power
                )

            instance.tank_energy_rule_heating.activate()

        # Thermal storage DHW
        if self.scenario.hotwatertank_class.size == 0:
            for t in range(1, 8761):
                instance.E_DHWTank[t].fix(0)
                instance.Q_DHWTank_out[t].fix(0)
                instance.Q_DHWTank_in[t].fix(0)

                instance.Q_DHW_HP_out[t].setub(
                    self.scenario.boiler_class.thermal_power_max + self.scenario.boiler_class.heating_element_power
                )
            instance.tank_energy_rule_DHW.deactivate()
        else:
            for t in range(1, 8761):
                instance.E_DHWTank[t].fixed = False
                instance.Q_DHWTank_out[t].fixed = False
                instance.Q_DHWTank_in[t].fixed = False
                instance.E_DHWTank[t].setlb(
                    CPWater * self.scenario.hotwatertank_class.size *
                    (273.15 + self.scenario.hotwatertank_class.temperature_min)
                )
                instance.E_DHWTank[t].setub(
                    CPWater * self.scenario.hotwatertank_class.size *
                    (273.15 + self.scenario.hotwatertank_class.temperature_max)
                )
                instance.Q_DHW_HP_out[t].setub(
                    self.scenario.boiler_class.thermal_power_max + self.scenario.boiler_class.heating_element_power
                )
            instance.tank_energy_rule_DHW.activate()

        # Battery
        if self.scenario.battery_class.capacity == 0:
            for t in range(1, 8761):
                # fix the parameters to 0
                instance.Grid2Bat[t].fix(0)
                instance.Bat2Load[t].fix(0)
                instance.BatSoC[t].fix(0)
                instance.BatCharge[t].fix(0)
                instance.BatDischarge[t].fix(0)
                instance.PV2Bat[t].fix(0)
            instance.BatCharge_rule.deactivate()
            instance.BatDischarge_rule.deactivate()
            instance.BatSoC_rule.deactivate()
        else:
            for t in range(1, 8761):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.Grid2Bat[t].fixed = False
                instance.Bat2Load[t].fixed = False
                instance.BatSoC[t].fixed = False
                instance.BatCharge[t].fixed = False
                instance.BatDischarge[t].fixed = False
                instance.PV2Bat[t].fixed = False
                # set upper bounds
                instance.Grid2Bat[t].setub(self.scenario.battery_class.charge_power_max)

                instance.Bat2Load[t].setub(self.scenario.battery_class.discharge_power_max)
                instance.BatSoC[t].setub(self.scenario.battery_class.capacity)
                instance.BatCharge[t].setub(self.scenario.battery_class.charge_power_max)
                instance.BatDischarge[t].setub(self.scenario.battery_class.discharge_power_max)
            instance.BatCharge_rule.activate()
            instance.BatDischarge_rule.activate()
            instance.BatSoC_rule.activate()

        # PV
        if self.scenario.pv_class.peak_power == 0:
            for t in range(1, 8761):
                instance.PV2Load[t].fix(0)
                instance.PV2Bat[t].fix(0)
                instance.PV2Grid[t].fix(0)
            instance.UseOfPV_rule.deactivate()

        else:
            for t in range(1, 8761):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.PV2Load[t].fixed = False
                instance.PV2Bat[t].fixed = False
                instance.PV2Grid[t].fixed = False
                # set upper bounds
                instance.PV2Load[t].setub(self.scenario.building_class.grid_power_max)
                instance.PV2Bat[t].setub(self.scenario.building_class.grid_power_max)
                instance.PV2Grid[t].setub(self.scenario.building_class.grid_power_max)
            instance.UseOfPV_rule.activate()

        # update time independent parameters
        # building parameters:
        instance.Am = self.scenario.building_class.Am_factor
        instance.Hve = self.scenario.building_class.Hve
        instance.Htr_w = self.scenario.building_class.Htr_w
        # parameters that have to be calculated:
        instance.Atot = self.calculate_Atot(Af=self.scenario.building_class.Af)
        instance.Qi = self.calculate_Qi(specific_internal_gains=self.scenario.building_class.internal_gains,
                                        Af=self.scenario.building_class.Af)
        instance.Htr_em = self.calculate_Htr_em(Hop=self.scenario.building_class.Hop,
                                                Am_factor=self.scenario.building_class.Am_factor,
                                                Af=self.scenario.building_class.Af)
        instance.Htr_3 = self.calculate_Htr_3(Hve=self.scenario.building_class.Hve,
                                              Af=self.scenario.building_class.Af,
                                              Htr_w=self.scenario.building_class.Htr_w,
                                              Am_factor=self.scenario.building_class.Am_factor)
        instance.Htr_1 = self.calculate_Htr_1(Hve=self.scenario.building_class.Hve,
                                              Af=self.scenario.building_class.Af)
        instance.Htr_2 = self.calculate_Htr_2(Hve=self.scenario.building_class.Hve,
                                              Af=self.scenario.building_class.Af,
                                              Htr_w=self.scenario.building_class.Htr_w)
        instance.Htr_ms = self.calculate_Htr_ms(Am_factor=self.scenario.building_class.Am_factor,
                                                Af=self.scenario.building_class.Af)
        instance.Htr_is = self.calculate_Htr_is(Af=self.scenario.building_class.Af)
        instance.PHI_ia = self.calculate_PHI_ia(specific_internal_gains=self.scenario.building_class.internal_gains,
                                                Af=self.scenario.building_class.Af)
        instance.Cm = self.calculate_Cm(CM_factor=self.scenario.building_class.CM_factor,
                                        Af=self.scenario.building_class.Af)

        instance.BuildingMassTemperatureStartValue = self.scenario.building_class.building_mass_temperature_start
        # Battery parameters
        instance.ChargeEfficiency = self.scenario.battery_class.charge_efficiency
        instance.DischargeEfficiency = self.scenario.battery_class.discharge_efficiency
        # Thermal storage heating parameters
        instance.T_TankStart_heating = self.scenario.spaceheatingtank_class.temperature_start
        instance.M_WaterTank_heating = self.scenario.spaceheatingtank_class.size
        instance.U_ValueTank_heating = self.scenario.spaceheatingtank_class.loss
        instance.T_TankSurrounding_heating = self.scenario.spaceheatingtank_class.temperature_surrounding
        instance.A_SurfaceTank_heating = self.scenario.spaceheatingtank_class.surface_area
        # Thermal storage DHW parameters
        instance.T_TankStart_DHW = self.scenario.hotwatertank_class.temperature_start
        instance.M_WaterTank_DHW = self.scenario.hotwatertank_class.size
        instance.U_ValueTank_DHW = self.scenario.hotwatertank_class.loss
        instance.T_TankSurrounding_DHW = self.scenario.hotwatertank_class.temperature_surrounding
        instance.A_SurfaceTank_DHW = self.scenario.hotwatertank_class.surface_area
        # HP
        instance.SpaceHeating_HeatPumpMaximalThermalPower = self.scenario.boiler_class.thermal_power_max + \
                                                            self.scenario.boiler_class.heating_element_power
        # Cooling
        instance.CoolingCOP = self.scenario.airconditioner_class.efficiency  # is a single value, no index

        return instance

    def solve_optimization(self, instance2solve):
        Opt = pyo.SolverFactory("gurobi")
        result = Opt.solve(instance2solve, tee=False)
        return instance2solve


if __name__ == "__main__":
    from _Refactor.core.household.abstract_scenario import AbstractScenario

    scenario = AbstractScenario(scenario_id=0)
    model_class = OptOperationModel(scenario)

    abstract_model = model_class.create_abstract_model()
    pyomo_instance = abstract_model.create_instance(data=model_class.create_pyomo_dict())
    updated_instance = model_class.update_instance(pyomo_instance)
    # solve model
    solved_instance = model_class.solve_optimization(updated_instance)
    # datacollector
    hourly_results_df = OptimizationDataCollector().collect_optimization_results_hourly(solved_instance)
    yearly_results_df = OptimizationDataCollector().collect_optimization_results_yearly(solved_instance)

    pass

