from typing import Iterable
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import logging
from models.operation.constants import OperationResultVar
from models.operation.model_base import OperationModel


class OptInstance:
    def __init__(self, instance_length: int):
        self.instance_length = instance_length
    
    # @performance_counter
    def create_instance(self):
        model = self.setup_model()
        instance = model.create_instance()
        return instance

    def setup_model(self):
        m = pyo.AbstractModel()
        self.setup_sets(m)
        self.setup_params(m)
        self.setup_variables(m)
        self.setup_constraint_space_heating_tank(m)
        self.setup_constraint_space_heating_room(m)
        self.setup_constraint_thermal_mass_temperature(m)
        self.setup_constraint_room_temperature(m)
        self.setup_constraint_heating_element(m)
        self.setup_constraint_hot_water(m)
        self.setup_constraint_heat_pump(m)
        self.setup_constraint_boiler(m)
        self.setup_constraint_space_cooling(m)
        self.setup_constraint_pv(m)
        self.setup_constraint_battery(m)
        self.setup_constraint_ev(m)
        self.setup_constraint_electricity_demand(m)
        self.setup_constraint_electricity_supply(m)
        self.setup_objective(m)
        return m

    def setup_sets(self, m):
        m.t = pyo.Set(initialize=np.arange(1, self.instance_length + 1))

    @staticmethod
    def setup_params(m):
        # time dependent parameters:
        # external parameters:
        # outside temperature
        m.T_outside = pyo.Param(m.t, mutable=True)
        # solar gains:
        m.Q_Solar = pyo.Param(m.t, mutable=True)
        m.ElectricityPrice = pyo.Param(m.t, mutable=True)
        m.price_quantile_for_charging_thermal_mass = pyo.Param(mutable=True)  # not time depend, has to be calculated outside
        m.FuelPrice = pyo.Param(m.t, mutable=True)
        # Feed in Tariff of Photovoltaic
        m.FiT = pyo.Param(m.t, mutable=True)

        # heating
        # Boiler
        m.Boiler_COP = pyo.Param(mutable=True, within=pyo.Any)
        m.Boiler_MaximalThermalPower = pyo.Param(mutable=True, within=pyo.Any)
        # COP of heat pump
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t, mutable=True)
        # COP of heatpump for charging buffer storage
        m.SpaceHeatingHourlyCOP_tank = pyo.Param(m.t, mutable=True)
        # hot water
        m.HotWaterProfile = pyo.Param(m.t, mutable=True)
        # COP for hot water
        m.HotWaterHourlyCOP = pyo.Param(m.t, mutable=True)
        # COP for hot water when charging DHW storage
        m.HotWaterHourlyCOP_tank = pyo.Param(m.t, mutable=True)

        # time independent parameters:
        m.SpaceHeating_MaxBoilerPower = pyo.Param(mutable=True)
        m.CPWater = pyo.Param(mutable=True)

        # electricity load profile
        m.BaseLoadProfile = pyo.Param(m.t, mutable=True)
        # PV
        m.PhotovoltaicProfile = pyo.Param(m.t, mutable=True)
        # Electric vehicle
        m.EVDemandProfile = pyo.Param(m.t, mutable=True)

        # COP of cooling
        m.CoolingHourlyCOP = pyo.Param(m.t, mutable=True)

        # heating element
        m.HeatingElement_efficiency = pyo.Param(mutable=True)

        # heating tank
        m.U_LossTank_heating = pyo.Param(mutable=True)
        m.A_SurfaceTank_heating = pyo.Param(mutable=True)
        m.M_WaterTank_heating = pyo.Param(mutable=True)
        m.T_TankSurrounding_heating = pyo.Param(mutable=True)
        m.Q_TankEnergyMin_heating = pyo.Param(mutable=True)
        m.space_heating_tank_temperature_max = pyo.Param(mutable=True)

        # DHW tank
        m.U_LossTank_DHW = pyo.Param(mutable=True)
        m.A_SurfaceTank_DHW = pyo.Param(mutable=True)
        m.M_WaterTank_DHW = pyo.Param(mutable=True)
        m.T_TankSurrounding_DHW = pyo.Param(mutable=True)
        m.Q_TankEnergyMin_DHW = pyo.Param(mutable=True)
        

        # battery
        m.BatteryChargeEfficiency = pyo.Param(mutable=True)
        m.BatteryDischargeEfficiency = pyo.Param(mutable=True)

        # electric vehicle
        m.EVChargeEfficiency = pyo.Param(mutable=True)
        m.EVDischargeEfficiency = pyo.Param(mutable=True)
        m.EVCapacity = pyo.Param(mutable=True)

        # building parameters
        m.Am = pyo.Param(mutable=True)
        m.Atot = pyo.Param(mutable=True)
        m.Qi = pyo.Param(mutable=True)
        m.Htr_w = pyo.Param(mutable=True)
        m.Htr_em = pyo.Param(mutable=True)
        m.Htr_ms = pyo.Param(mutable=True)
        m.Htr_is = pyo.Param(mutable=True)
        m.Hve = pyo.Param(mutable=True)
        m.Htr_1 = pyo.Param(mutable=True)
        m.Htr_2 = pyo.Param(mutable=True)
        m.Htr_3 = pyo.Param(mutable=True)
        m.PHI_ia = pyo.Param(mutable=True)
        m.Cm = pyo.Param(mutable=True)

        # start values
        m.BuildingMassTemperatureStartValue = pyo.Param(mutable=True)
        m.BatterySOCStartValue = pyo.Param(mutable=True)
        m.T_TankStart_DHW = pyo.Param(mutable=True)
        m.dhw_tank_loss_factor = pyo.Param(mutable=True)
        m.T_TankStart_heating = pyo.Param(mutable=True)
        m.heating_tank_loss_factor = pyo.Param(mutable=True)

        # from the reference model the thermal mass temperature to identify the "value" of the preheated building mass at the end of the optimization
        m.reference_BuildingMassTemperature = pyo.Param(m.t, mutable=True)
        m.reference_Q_RoomHeating_binary = pyo.Param(m.t, mutable=True)
        m.thermal_mass_loss_factor = pyo.Param(m.t, mutable=True)
        
    @staticmethod
    def setup_variables(m):
        # space heating
        m.Q_HeatingTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_HeatingTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_Heating_Boiler_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Fuel = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # heating element
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingElement_DHW = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingElement_heat = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Temperatures (room and thermal mass)
        m.T_Room = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.T_BuildingMass = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # space cooling
        m.Q_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_RoomCooling = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # hot water
        m.Q_DHWTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_DHWTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_DHWTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_DHW_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_DHWTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_DHW_Boiler_out = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # PV
        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # battery
        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # EV
        m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # grid
        m.Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Grid2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Grid2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Electric Load and Electricity fed back to the grid
        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Feed2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

    @staticmethod
    def setup_constraint_space_heating_tank(m):
        def tank_energy_heating(m, t):
            if t == 1:
                return m.Q_HeatingTank[t] == \
                       m.CPWater * m.M_WaterTank_heating * (273.15 + m.T_TankStart_heating) - \
                       m.Q_HeatingTank_out[t]
            else:
                return m.Q_HeatingTank[t] == \
                       m.Q_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] - \
                       m.U_LossTank_heating * m.A_SurfaceTank_heating * \
                       ((m.Q_HeatingTank[t] / (m.M_WaterTank_heating * m.CPWater)) -
                        (m.T_TankSurrounding_heating + 273.15))
        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

    @staticmethod
    def setup_constraint_space_heating_room(m):
        def room_heating(m, t):
            return m.Q_RoomHeating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t]
        m.room_heating_rule = pyo.Constraint(m.t, rule=room_heating)

    @staticmethod
    def setup_constraint_thermal_mass_temperature(m):
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
                        (((m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
            # Equ. C.4
            return m.T_BuildingMass[t] == \
                   (Tm_start * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / \
                   ((m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))
        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

    @staticmethod
    def setup_constraint_room_temperature(m):

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
            T_s = (
                          m.Htr_ms * T_m
                          + PHI_st
                          + m.Htr_w * m.T_outside[t]
                          + m.Htr_1
                          * (
                                  T_sup
                                  + (m.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / m.Hve
                          )
                  ) / (m.Htr_ms + m.Htr_w + m.Htr_1)
            # Equ. C.11
            T_air = (
                            m.Htr_is * T_s
                            + m.Hve * T_sup
                            + m.PHI_ia
                            + m.Q_RoomHeating[t]
                            - m.Q_RoomCooling[t]
                    ) / (m.Htr_is + m.Hve)
            return m.T_Room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

    @staticmethod
    def setup_constraint_hot_water(m):

        def tank_energy_DHW(m, t):
            if t == 1:
                return (
                        m.Q_DHWTank[t]
                        == m.CPWater
                        * m.M_WaterTank_DHW
                        * (273.15 + m.T_TankStart_DHW)
                        - m.Q_DHWTank_out[t]
                )
            else:
                return m.Q_DHWTank[t] == m.Q_DHWTank[t - 1] - m.Q_DHWTank_out[
                    t
                ] + m.Q_DHWTank_in[t] - m.U_LossTank_DHW * m.A_SurfaceTank_DHW * (
                               m.Q_DHWTank[t] / (m.M_WaterTank_DHW * m.CPWater)
                               - (m.T_TankSurrounding_DHW + 273.15)
                       )

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (14) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.SupplyOfDHW_rule = pyo.Constraint(m.t, rule=calc_hot_water_profile)

    @staticmethod
    def setup_constraint_heat_pump(m):
        def calc_supply_of_space_heating_HP(m, t):
            return (
                    m.Q_HeatingTank_bypass[t] * m.SpaceHeatingHourlyCOP_tank[t] +
                    m.Q_HeatingTank_in[t] * m.SpaceHeatingHourlyCOP[t] ==
                    m.E_Heating_HP_out[t] * m.SpaceHeatingHourlyCOP_tank[t] * m.SpaceHeatingHourlyCOP[t] +
                    m.Q_HeatingElement_heat[t]
            )

        m.calc_use_of_HP_power_DHW_rule = pyo.Constraint(
            m.t, rule=calc_supply_of_space_heating_HP
        )

        def calc_supply_of_DHW_HP(m, t):
            return (
                    m.Q_DHWTank_bypass[t] * m.HotWaterHourlyCOP_tank[t] +
                    m.Q_DHWTank_in[t] * m.HotWaterHourlyCOP[t] ==
                    m.E_DHW_HP_out[t] * m.HotWaterHourlyCOP_tank[t] * m.HotWaterHourlyCOP[t] +
                    m.Q_HeatingElement_DHW[t]
            )

        m.bypass_DHW_HP_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW_HP)

        def constrain_heating_max_power_HP(m, t):
            return (
                    m.E_DHW_HP_out[t] + m.E_Heating_HP_out[t]
                    <= m.SpaceHeating_MaxBoilerPower
            )

        m.max_HP_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power_HP)

    @staticmethod
    def setup_constraint_boiler(m):
        def calc_supply_of_space_heating_fuel_boiler(m, t):
            return (m.Q_HeatingTank_bypass[t] + m.Q_HeatingTank_in[t] ==
                    m.Q_Heating_Boiler_out[t] + m.Q_HeatingElement_heat[t])

        m.calc_use_of_fuel_boiler_power_DHW_rule = pyo.Constraint(
            m.t, rule=calc_supply_of_space_heating_fuel_boiler
        )

        def calc_supply_of_DHW_fuel_boiler(m, t):
            return (
                    m.Q_DHWTank_bypass[t] + m.Q_DHWTank_in[t] ==
                    m.Q_DHW_Boiler_out[t] + m.Q_HeatingElement_DHW[t]
            )

        m.bypass_DHW_fuel_boiler_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW_fuel_boiler)

        def constrain_heating_max_power_fuel_boiler(m, t):
            return (
                    m.Q_DHW_Boiler_out[t] + m.Q_Heating_Boiler_out[t]
                    <= m.Boiler_MaximalThermalPower
            )

        m.max_fuel_boiler_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power_fuel_boiler)

        def calc_boiler_conversion(m, t):
            return (
                    m.Q_DHW_Boiler_out[t] + m.Q_Heating_Boiler_out[t] == m.Fuel[t] * m.Boiler_COP
            )

        m.boiler_conversion_rule = pyo.Constraint(m.t, rule=calc_boiler_conversion)

    @staticmethod
    def setup_constraint_heating_element(m):
        def calc_heating_element(m, t):
            return m.Q_HeatingElement_DHW[t] + m.Q_HeatingElement_heat[t] == m.Q_HeatingElement[t]
        m.heating_element_rule = pyo.Constraint(m.t, rule=calc_heating_element)

    @staticmethod
    def setup_constraint_space_cooling(m):
        def calc_E_RoomCooling_with_cooling(m, t):
            return m.E_RoomCooling[t] == m.Q_RoomCooling[t] / m.CoolingHourlyCOP[t]
        m.E_RoomCooling_with_cooling_rule = pyo.Constraint(m.t, rule=calc_E_RoomCooling_with_cooling)

    @staticmethod
    def setup_constraint_pv(m):

        def calc_UseOfPV(m, t):
            return m.PhotovoltaicProfile[t] == \
                   m.PV2EV[t] + m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t]
        m.UseOfPV_rule = pyo.Constraint(m.t, rule=calc_UseOfPV)

        def calc_SumOfFeedin(m, t):
            return m.Feed2Grid[t] == m.PV2Grid[t]
        m.SumOfFeedin_rule = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

    @staticmethod
    def setup_constraint_battery(m):

        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t] + m.EV2Bat[t]
        m.BatCharge_rule = pyo.Constraint(m.t, rule=calc_BatCharge)

        def calc_BatDischarge(m, t):
            return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t]
        m.BatDischarge_rule = pyo.Constraint(m.t, rule=calc_BatDischarge)

        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == m.BatterySOCStartValue + m.BatCharge[t] * m.BatteryChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.BatteryDischargeEfficiency))
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.BatteryChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.BatteryDischargeEfficiency))
        m.BatSoC_rule = pyo.Constraint(m.t, rule=calc_BatSoC)

    @staticmethod
    def setup_constraint_ev(m):

        def calc_EVCharge(m, t):
            return m.EVCharge[t] == m.PV2EV[t] + m.Grid2EV[t] + m.Bat2EV[t]
        m.EVCharge_rule = pyo.Constraint(m.t, rule=calc_EVCharge)

        def calc_EVDischarge(m, t):
            return m.EVDischarge[t] == m.EVDemandProfile[t] + m.EV2Load[t] + m.EV2Bat[t]
        m.EVDischarge_rule = pyo.Constraint(m.t, rule=calc_EVDischarge)

        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == m.EVCapacity + \
                                     m.EVCharge[t] * m.EVChargeEfficiency - \
                                     m.EVDischarge[t] / m.EVDischargeEfficiency
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + \
                                     m.EVCharge[t] * m.EVChargeEfficiency - \
                                     m.EVDischarge[t] / m.EVDischargeEfficiency
        m.EVSoC_rule = pyo.Constraint(m.t, rule=calc_EVSoC)

    @staticmethod
    def setup_constraint_electricity_demand(m):

        def calc_SumOfLoads_with_cooling(m, t):
            return m.Load[t] == \
                   m.BaseLoadProfile[t] + \
                   m.E_Heating_HP_out[t] + \
                   m.Q_HeatingElement[t] / m.HeatingElement_efficiency + \
                   m.E_RoomCooling[t] + \
                   m.E_DHW_HP_out[t]
        m.SumOfLoads_with_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

    @staticmethod
    def setup_constraint_electricity_supply(m):

        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t] + m.Grid2EV[t]
        m.UseOfGrid_rule = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        def calc_SupplyOfLoads(m, t):
            return m.Load[t] == m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + m.EV2Load[t]
        m.SupplyOfLoads_rule = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)
    

    
    @staticmethod
    def setup_objective(m):
        def minimize_cost(m):
            nr_timesteps = max(m.t)
            average_price = sum(m.ElectricityPrice[t] for t in m.t) / nr_timesteps
            average_outisde_temperature = sum(m.T_outside[t] for t in m.t) / nr_timesteps
            average_heating_COP = sum(m.SpaceHeatingHourlyCOP[t] for t in m.t) / nr_timesteps
            solar_gains = 0
            average_dhw_tank_COP = sum(m.HotWaterHourlyCOP_tank[t] for t in m.t) / nr_timesteps
            average_heating_tank_COP = sum(m.SpaceHeatingHourlyCOP_tank[t] for t in m.t) / nr_timesteps

            # calculate the value of a pre-heated thermal mass:
            # Equ. C.2
            PHI_m = m.Am / m.Atot * (0.5 * m.Qi + solar_gains)
            # Equ. C.3
            PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + solar_gains)
            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = average_outisde_temperature

                    
            # difference between the building mass temperature of the same timestep in the reference model:
            PHI_mtot = (m.T_BuildingMass[nr_timesteps] - m.reference_BuildingMassTemperature[nr_timesteps]) * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) 
            
            # Equ. C.5
            Q_RoomHeating = (((PHI_mtot - PHI_m - m.Htr_em * average_outisde_temperature) *  m.Htr_2 / m.Htr_3 - PHI_st - m.Htr_w * average_outisde_temperature) / m.Htr_1 - T_sup) * m.Hve - m.PHI_ia  #- m.Q_RoomCooling[t] 
            # Equ. C.4
            # equivalent electricity of the heat pump when the thermal mass is hotter than in the reference model at the end of the time horizon
            # including binary variable to restrain the model from pre-heating in summer where there is no value in pre heating
            thermal_mass_value = Q_RoomHeating / average_heating_COP * (1-m.thermal_mass_loss_factor[nr_timesteps]) * m.reference_Q_RoomHeating_binary[nr_timesteps]

            # monetize battery 
            battery_value =  m.BatSoC[nr_timesteps] * m.BatteryDischargeEfficiency
            
            # monetize DHW tank
            # DHW_tank_value = (m.Q_DHWTank[nr_timesteps] - m.Q_TankEnergyMin_DHW) / average_dhw_tank_COP * (1-m.dhw_tank_loss_factor)
            
            # monetize heating tank
            # heating_tank_value =  (m.Q_HeatingTank[nr_timesteps] - m.Q_TankEnergyMin_heating) / average_heating_tank_COP  * (1-m.heating_tank_loss_factor) * m.reference_Q_RoomHeating_binary[nr_timesteps]
                
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] + m.Fuel[t] * m.FuelPrice[t] - m.Feed2Grid[t] * m.FiT[t] for t in m.t) - thermal_mass_value * m.price_quantile_for_charging_thermal_mass   # DHW_tank_value  heating_tank_value monetizing battery storage with average electricity price and discharge efficiency
                                               
            return rule
        m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)

    @staticmethod
    def create_dict(values) -> dict:
        dictionary = {}
        for index, value in enumerate(values, start=1):
            dictionary[index] = value
        return dictionary


class OptOperationModel(OperationModel):
    def get_indices(self):
        indices = [(0, 24)]  # Start with the initial 24 elements
        indices += [(i, i + 24) for i in np.arange(12, 8760, 24)]
        return indices
    
    def update_initial_values(self, model, last_timestep_values):
        for var_name, value in last_timestep_values.items():
            try:
                getattr(model, var_name)[1].value = value
            except KeyError:
                getattr(model, var_name).value = value
        return model
    
    def update_large_model(self, large_model, timestep_values, nr_iteration: int):
        indices = self.get_indices()[nr_iteration]
        for var_name, value in timestep_values.items():
            # getattr(large_model, var_name).extract_values().value() = value
            large_var = getattr(large_model, var_name)            
            if isinstance(value, Iterable) and len(value) != 1:
                for t in range(len(value)):
                    large_var[indices[0]+1+t].value = value[t]
                else:
                    large_var.value = value
        return large_model
    
    def extract_timestep_values(self, model, timestep: int):
        last_timestep_values = {}
        for var in model.component_objects(pyo.Var, active=True):
            var_name = var.getname(fully_qualified=True)
            last_timestep_values[var_name] = pyo.value(var[timestep])
            if var_name == "T_BuildingMass":
                last_timestep_values["BuildingMassTemperatureStartValue"] = pyo.value(var[timestep])
            elif var_name == "BatSoC":
                last_timestep_values["BatterySOCStartValue"] = pyo.value(var[timestep])
            elif var_name == "Q_HeatingTank" and model.M_WaterTank_heating.value != 0:
                last_timestep_values["T_TankStart_heating"] = pyo.value(var[timestep]) / (model.M_WaterTank_heating.value * model.CPWater.value) - 273.15
            elif var_name == "Q_DHWTank" and model.M_WaterTank_DHW.value != 0:
                last_timestep_values["T_TankStart_DHW"] = pyo.value(var[timestep]) / (model.M_WaterTank_DHW.value * model.CPWater.value) - 273.15
            elif var_name == "BatSoC":
                last_timestep_values["BatterySOCStartValue"] = pyo.value(var[timestep])             

        return last_timestep_values
    
    def extract_first_x_values(self, model, nr_of_values: int):
        first_x_values = {}
        for var in model.component_objects(pyo.Var, active=True):
            var_name = var.getname(fully_qualified=True)
            first_x_values[var_name] = [pyo.value(var[t]) for t in range(1, nr_of_values+1)]

        for param in model.component_objects(pyo.Param, active=True):
            param_name = param.getname(fully_qualified=True) 
            param_values = []
            for t in sorted(param)[:nr_of_values]:
                if param[t].is_constant():
                    param_values.append(pyo.value(param[t]))
                else:
                    if param[t].value is not None:
                        param_values.append(pyo.value(param[t]))
            first_x_values[param_name] = param_values

        
        return first_x_values

    # @performance_counter
    def solve(self, full_instance, rolling_horizon: bool):
        logger = logging.getLogger(f"{self.project_name}")
        logger.info("starting solving Opt model.")
        solved = True

        if rolling_horizon:
            sub_instances = self.return_splitted_scenarios()
            instance = OptInstance(instance_length=36).create_instance()
            instance_12 = OptInstance(instance_length=12).create_instance()
            instance_24 = OptInstance(instance_length=24).create_instance()
            for i, sub_instance in enumerate(sub_instances):
                if i == 0:
                    opt_instance = OptConfig(model=sub_instance, instance_length=24).config_instance(instance_24)
                    results = pyo.SolverFactory("gurobi").solve(opt_instance, tee=False)
                    if results.solver.termination_condition != TerminationCondition.optimal:
                        solved = False
                        break
                    opt_instance.solutions.load_from(results)
                    # get timestep_values of 12 o'clock and use them as initial values for the next run:
                    last_timestep_values = self.extract_timestep_values(model=opt_instance, timestep=12)

                    # update the large model with the results:
                    timestep_values = self.extract_first_x_values(model=opt_instance, nr_of_values=24)
                    full_instance = self.update_large_model(large_model=full_instance, timestep_values=timestep_values, nr_iteration=i)

                elif i == len(sub_instances)-1:  # here we need to initialize a new instance with 12 values as size
                    opt_instance = OptConfig(model=sub_instance, instance_length=12).config_instance(instance_12)
                    opt_instance = self.update_initial_values(model=opt_instance, last_timestep_values=last_timestep_values)
                    results = pyo.SolverFactory("gurobi").solve(opt_instance, tee=False)
                    if results.solver.termination_condition != TerminationCondition.optimal:
                        solved = False
                        break
                    opt_instance.solutions.load_from(results)

                    # load the results into the full 8760 model:
                    timestep_values = self.extract_first_x_values(model=opt_instance, nr_of_values=12)
                    full_instance = self.update_large_model(large_model=full_instance, timestep_values=timestep_values, nr_iteration=i)

                else:
                    opt_instance = OptConfig(model=sub_instance, instance_length=36).config_instance(instance)
                    opt_instance = self.update_initial_values(model=opt_instance, last_timestep_values=last_timestep_values)
                    results = pyo.SolverFactory("gurobi").solve(opt_instance, tee=False)
                    if results.solver.termination_condition != TerminationCondition.optimal:
                        solved = False
                        break
                    opt_instance.solutions.load_from(results)
                    last_timestep_values = self.extract_timestep_values(model=opt_instance, timestep=24)

                    # update the large model with the results:
                    timestep_values = self.extract_first_x_values(model=opt_instance, nr_of_values=24)
                    full_instance = self.update_large_model(large_model=full_instance, timestep_values=timestep_values, nr_iteration=i)
                print(f"calculated day {i}")

        else:
            opt_instance = OptConfig(model=self, instance_length=8760).config_instance(full_instance)
            results = pyo.SolverFactory("gurobi").solve(opt_instance, tee=False)
            if results.solver.termination_condition != TerminationCondition.optimal:
                        return None, False
            else:
                opt_instance.solutions.load_from(results)
                full_instance = opt_instance
        return full_instance, solved


class OptConfig:

    def __init__(self, model: 'OperationModel', instance_length: int):
        self.model = model
        self.instance_length = instance_length

    # @performance_counter
    def config_instance(self, instance):
        self.config_room_temperature(instance)
        self.config_vehicle(instance)
        self.config_static_params(instance)
        self.config_external_params(instance)
        self.config_prices(instance)
        # self.config_grid(instance)
        self.config_space_heating(instance)
        self.config_space_heating_tank(instance)
        self.config_hot_water_tank(instance)
        self.config_battery(instance)
        self.config_space_cooling_technology(instance)
        self.config_pv(instance)
        self.config_heating_element(instance)
        return instance

    def config_static_params(self, instance):
        instance.CPWater = self.model.CPWater
        # building parameters:
        instance.Am = self.model.Am_factor
        instance.Hve = self.model.Hve
        instance.Htr_w = self.model.Htr_w
        # parameters that have to be calculated:
        instance.Atot = self.model.Atot
        instance.Qi = self.model.Qi
        instance.Htr_em = self.model.Htr_em
        instance.Htr_3 = self.model.Htr_3
        instance.Htr_1 = self.model.Htr_1
        instance.Htr_2 = self.model.Htr_2
        instance.Htr_ms = self.model.Htr_ms
        instance.Htr_is = self.model.Htr_is
        instance.PHI_ia = self.model.PHI_ia
        instance.Cm = self.model.Cm
        instance.BuildingMassTemperatureStartValue = self.model.BuildingMassTemperatureStartValue

        # Battery parameters
        instance.BatteryChargeEfficiency = self.model.battery_charge_efficiency
        instance.BatteryDischargeEfficiency = self.model.battery_discharge_efficiency
        instance.BatterySOCStartValue = self.model.BatterySOCStartValue

        # EV parameters
        instance.EVChargeEfficiency = self.model.EVChargeEfficiency
        instance.EVDischargeEfficiency = self.model.EVDischargeEfficiency
        instance.EVCapacity = self.model.EVCapacity

        # Thermal storage heating parameters
        instance.T_TankStart_heating = self.model.T_TankStart_heating
        instance.M_WaterTank_heating = self.model.M_WaterTank_heating
        instance.U_LossTank_heating = self.model.U_LossTank_heating
        instance.T_TankSurrounding_heating = self.model.T_TankSurrounding_heating
        instance.A_SurfaceTank_heating = self.model.A_SurfaceTank_heating
        instance.Q_TankEnergyMin_heating = self.model.Q_TankEnergyMin_heating
        instance.space_heating_tank_temperature_max = self.model.space_heating_tank_temperature_max
        instance.heating_tank_loss_factor = self.model.heating_tank_loss_factor

        # Thermal storage DHW parameters
        instance.T_TankStart_DHW = self.model.T_TankStart_DHW
        instance.M_WaterTank_DHW = self.model.M_WaterTank_DHW
        instance.U_LossTank_DHW = self.model.U_LossTank_DHW
        instance.T_TankSurrounding_DHW = self.model.T_TankSurrounding_DHW
        instance.A_SurfaceTank_DHW = self.model.A_SurfaceTank_DHW
        instance.Q_TankEnergyMin_DHW = self.model.Q_TankEnergyMin_DHW
        instance.dhw_tank_loss_factor = self.model.dhw_tank_loss_factor

        # HP
        instance.SpaceHeating_MaxBoilerPower = self.model.SpaceHeating_MaxBoilerPower

        # Boiler
        instance.Boiler_COP = self.model.fuel_boiler_efficiency
        instance.Boiler_MaximalThermalPower = self.model.SpaceHeating_MaxBoilerPower

    def config_external_params(self, instance):
        for t in range(1, self.instance_length + 1):
            instance.Q_Solar[t] = self.model.Q_Solar[t-1]
            instance.T_outside[t] = self.model.T_outside[t-1]
            instance.HotWaterProfile[t] = self.model.HotWaterProfile[t-1]
            instance.BaseLoadProfile[t] = self.model.BaseLoadProfile[t-1]
            instance.PhotovoltaicProfile[t] = self.model.PhotovoltaicProfile[t-1]
            instance.thermal_mass_loss_factor[t] = self.model.thermal_mass_loss_factor[t-1]


        if self.model.boiler_type in ["Air_HP", "Ground_HP", "Electric"]:
            for t in range(1, self.instance_length + 1):
                # unfix heat pump parameters:
                instance.E_Heating_HP_out[t].fixed = False
                instance.E_DHW_HP_out[t].fixed = False

                # update heat pump parameters
                instance.SpaceHeatingHourlyCOP[t] = self.model.SpaceHeatingHourlyCOP[t - 1]
                instance.SpaceHeatingHourlyCOP_tank[t] = self.model.SpaceHeatingHourlyCOP_tank[t - 1]
                instance.HotWaterHourlyCOP[t] = self.model.HotWaterHourlyCOP[t - 1]
                instance.HotWaterHourlyCOP_tank[t] = self.model.HotWaterHourlyCOP_tank[t - 1]

                # set boiler specifics to zero
                instance.Fuel[t].fix(0)
                instance.Q_DHW_Boiler_out[t].fix(0)
                instance.Q_Heating_Boiler_out[t].fix(0)

            # deactivate boiler functions
            instance.boiler_conversion_rule.deactivate()
            instance.max_fuel_boiler_power_rule.deactivate()
            instance.bypass_DHW_fuel_boiler_rule.deactivate()
            instance.calc_use_of_fuel_boiler_power_DHW_rule.deactivate()
            # activate heat pump functions
            instance.max_HP_power_rule.activate()
            instance.bypass_DHW_HP_rule.activate()
            instance.calc_use_of_HP_power_DHW_rule.activate()

        else:  # fuel based boiler instead of heat pump
            for t in range(1, self.instance_length + 1):
                # unfix boiler parameters:
                instance.Fuel[t].fixed = False
                instance.Q_DHW_Boiler_out[t].fixed = False
                instance.Q_Heating_Boiler_out[t].fixed = False

                # fix heat pump parameters
                instance.E_Heating_HP_out[t].fix(0)
                instance.E_DHW_HP_out[t].fix(0)

                instance.HotWaterHourlyCOP_tank[t] = 0
                instance.HotWaterHourlyCOP[t] = 0
                instance.SpaceHeatingHourlyCOP[t] = 0
                instance.SpaceHeatingHourlyCOP_tank[t] = 0

            # activate boiler functions
            instance.boiler_conversion_rule.activate()
            instance.max_fuel_boiler_power_rule.activate()
            instance.bypass_DHW_fuel_boiler_rule.activate()
            instance.calc_use_of_fuel_boiler_power_DHW_rule.activate()
            # deactivate heat pump functions
            instance.max_HP_power_rule.deactivate()
            instance.bypass_DHW_HP_rule.deactivate()
            instance.calc_use_of_HP_power_DHW_rule.deactivate()

    def config_heating_element(self, instance):
        if self.model.HeatingElement_efficiency > 0:
            instance.HeatingElement_efficiency = self.model.HeatingElement_efficiency
        else:
            instance.HeatingElement_efficiency = 1  # to avoid that the model cant run
        if self.model.HeatingElement_power == 0:
            for t in range(1, self.instance_length + 1):
                instance.Q_HeatingElement[t].fix(0)
                instance.Q_HeatingElement_heat[t].fix(0)
                instance.Q_HeatingElement_DHW[t].fix(0)
            instance.heating_element_rule.deactivate()
        else:
            for t in range(1, self.instance_length + 1):
                instance.Q_HeatingElement[t].setub(self.model.HeatingElement_power)
                instance.Q_HeatingElement_heat[t].setub(self.model.HeatingElement_power)
                instance.Q_HeatingElement_DHW[t].setub(self.model.HeatingElement_power)
            instance.heating_element_rule.activate()

    def config_prices(self, instance):
        sorted_price = sorted(self.model.ElectricityPrice)
        lower_quantile = sorted_price[:int(self.instance_length*self.model.price_quantile_for_charging_thermal_mass)]
        instance.price_quantile_for_charging_thermal_mass = sum(lower_quantile) / len(lower_quantile)

        for t in range(1, self.instance_length + 1):
            instance.ElectricityPrice[t] = self.model.ElectricityPrice[t-1]
            instance.FiT[t] = self.model.FiT[t-1]
            if self.model.boiler_type not in ['Air_HP', 'Ground_HP', "Electric"]:
                instance.FuelPrice[t] = self.model.fuel_price.__dict__[self.model.boiler_type][t - 1]
            else:
                instance.FuelPrice[t] = 0

    def config_space_heating(self, instance):
        for t in range(1, self.instance_length + 1):
            instance.T_BuildingMass[t].setub(100)
        if self.model.boiler_type in ["Air_HP", "Ground_HP", "Electric"]:
            for t in range(1, self.instance_length + 1):
                instance.E_Heating_HP_out[t].setub(self.model.SpaceHeating_MaxBoilerPower)
        else:
            for t in range(1, self.instance_length + 1):
                instance.Q_Heating_Boiler_out[t].setub(self.model.SpaceHeating_MaxBoilerPower)
                instance.Q_DHW_Boiler_out[t].setub(self.model.SpaceHeating_MaxBoilerPower)

    def config_space_heating_tank(self, instance):
        if self.model.M_WaterTank_heating == 0:
            for t in range(1, self.instance_length + 1):
                instance.Q_HeatingTank[t].setlb(0)
                instance.Q_HeatingTank[t].setub(0)
                instance.Q_HeatingTank_out[t].fix(0)
                instance.Q_HeatingTank_in[t].fix(0)
                instance.Q_HeatingTank[t].fix(0)

            instance.tank_energy_rule_heating.deactivate()
        else:
            for t in range(1, self.instance_length + 1):
                instance.Q_HeatingTank_out[t].fixed = False
                instance.Q_HeatingTank_in[t].fixed = False
                instance.Q_HeatingTank[t].fixed = False
                instance.Q_HeatingTank[t].setlb(self.model.Q_TankEnergyMin_heating)
                instance.Q_HeatingTank[t].setub(self.model.Q_TankEnergyMax_heating)
            instance.tank_energy_rule_heating.activate()

    def config_hot_water_tank(self, instance):
        if self.model.M_WaterTank_DHW == 0:
            for t in range(1, self.instance_length + 1):
                instance.Q_DHWTank_out[t].fix(0)
                instance.Q_DHWTank_in[t].fix(0)
                instance.Q_DHWTank[t].setlb(0)
                instance.Q_DHWTank[t].setub(0)
                instance.Q_DHWTank[t].fix(0)

                instance.E_DHW_HP_out[t].setub(self.model.SpaceHeating_MaxBoilerPower)
            instance.tank_energy_rule_DHW.deactivate()
        else:
            for t in range(1, self.instance_length + 1):
                instance.Q_DHWTank_out[t].fixed = False
                instance.Q_DHWTank_in[t].fixed = False
                instance.Q_DHWTank[t].fixed = False
                instance.Q_DHWTank[t].setlb(self.model.Q_TankEnergyMin_DHW)
                instance.Q_DHWTank[t].setub(self.model.Q_TankEnergyMax_DHW)
                instance.E_DHW_HP_out[t].setub(self.model.SpaceHeating_MaxBoilerPower)
            instance.tank_energy_rule_DHW.activate()

    def config_battery(self, instance):
        if self.model.battery_capacity == 0:
            for t in range(1, self.instance_length + 1):
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
            # check if pv exists:
            if self.model.pv_size > 0:
                for t in range(1, self.instance_length + 1):
                    instance.PV2Bat[t].fixed = False
                    instance.PV2Bat[t].setub(self.model.battery_charge_power_max)
            else:
                for t in range(1, self.instance_length + 1):
                    instance.PV2Bat[t].fix(0)

            for t in range(1, self.instance_length + 1):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.Grid2Bat[t].fixed = False
                instance.Bat2Load[t].fixed = False
                instance.BatSoC[t].fixed = False
                instance.BatCharge[t].fixed = False
                instance.BatDischarge[t].fixed = False
                # set upper bounds
                instance.Grid2Bat[t].setub(self.model.battery_charge_power_max)
                instance.Bat2Load[t].setub(self.model.battery_discharge_power_max)
                instance.BatSoC[t].setub(self.model.battery_capacity)
                instance.BatCharge[t].setub(self.model.battery_charge_power_max)
                instance.BatDischarge[t].setub(self.model.battery_discharge_power_max)
            instance.BatCharge_rule.activate()
            instance.BatDischarge_rule.activate()
            instance.BatSoC_rule.activate()

    def config_room_temperature(self, instance):
        # in winter only 3°C increase to keep comfort level and in summer maximum reduction of 3°C
        for t in range(1, self.instance_length + 1):
            instance.reference_BuildingMassTemperature[t] = self.model.T_BuildingMass[t-1]
            instance.reference_Q_RoomHeating_binary[t] = self.model.reference_Q_RoomHeating_binary[t-1]
            instance.T_Room[t].setub(self.model.max_target_temperature[t - 1])
            instance.T_Room[t].setlb(self.model.min_target_temperature[t - 1])

    def config_space_cooling_technology(self, instance):
        if self.model.cooling_power == 0:
            for t in range(1, self.instance_length + 1):
                instance.Q_RoomCooling[t].fix(0)
                instance.E_RoomCooling[t].fix(0)
                instance.CoolingHourlyCOP[t] = 0

            instance.E_RoomCooling_with_cooling_rule.deactivate()
        else:
            for t in range(1, self.instance_length + 1):
                instance.Q_RoomCooling[t].fixed = False
                instance.E_RoomCooling[t].fixed = False
                instance.Q_RoomCooling[t].setub(self.model.cooling_power)
                instance.CoolingHourlyCOP[t] = self.model.CoolingHourlyCOP[t - 1]

            instance.E_RoomCooling_with_cooling_rule.activate()

    def config_vehicle(self, instance):
        max_discharge_ev = self.model.create_upper_bound_ev_discharge()
        if self.model.EVCapacity == 0:  # no EV is implemented
            for t in range(1, self.instance_length + 1):
                instance.Grid2EV[t].fix(0)
                instance.Bat2EV[t].fix(0)
                instance.PV2EV[t].fix(0)
                instance.EVSoC[t].fix(0)
                instance.EVCharge[t].fix(0)
                instance.EVDischarge[t].fix(0)
                instance.EV2Load[t].fix(0)
                instance.EV2Bat[t].fix(0)
                instance.EVDemandProfile[t] = 0
            instance.EVCharge_rule.deactivate()
            instance.EVDischarge_rule.deactivate()
            instance.EVSoC_rule.deactivate()
        else:
            # if there is a PV installed:
            if self.model.pv_size > 0:
                for t in range(1, self.instance_length + 1):
                    instance.PV2EV[t].fixed = False
            else:  # if there is no PV, EV can not be charged by it
                for t in range(1, self.instance_length + 1):
                    instance.PV2EV[t].fix(0)

            for t in range(1, self.instance_length + 1):
                # unfix variables
                instance.Grid2EV[t].fixed = False
                instance.Bat2EV[t].fixed = False
                instance.EVSoC[t].fixed = False
                instance.EVCharge[t].fixed = False
                instance.EVDischarge[t].fixed = False
                instance.EV2Load[t].fixed = False
                instance.EV2Bat[t].fixed = False

            for t in range(1, self.instance_length + 1):
                # set upper bounds
                instance.EVSoC[t].setub(self.model.EVCapacity)
                instance.EVCharge[t].setub(self.model.vehicle_charge_power_max)
                instance.EVDischarge[t].setub(max_discharge_ev[t - 1])
                instance.EVDemandProfile[t] = self.model.EVDemandProfile[t-1]
                # fix variables when EV is not at home:
                if self.model.EVAtHomeProfile[t-1] == 0:
                    instance.Grid2EV[t].fix(0)
                    instance.Bat2EV[t].fix(0)
                    instance.PV2EV[t].fix(0)
                    instance.EVCharge[t].fix(0)
                    instance.EV2Load[t].fix(0)
                    instance.EV2Bat[t].fix(0)

            # in case there is a stationary battery available:
            if self.model.battery_capacity > 0:
                if self.model.EVOptionV2B == 0:
                    for t in range(1, self.instance_length + 1):
                        instance.EV2Bat[t].fix(0)
                        instance.EV2Load[t].fix(0)

            # in case there is no stationary battery available:
            else:
                for t in range(1, self.instance_length + 1):
                    instance.EV2Bat[t].fix(0)
                    instance.Bat2EV[t].fix(0)
                    if self.model.EVOptionV2B == 0:
                        instance.EV2Load[t].fix(0)

            instance.EVCharge_rule.activate()
            instance.EVDischarge_rule.activate()
            instance.EVSoC_rule.activate()

    def config_pv(self, instance):
        if self.model.pv_size == 0:
            for t in range(1, self.instance_length + 1):
                instance.PV2Load[t].fix(0)
                instance.PV2Bat[t].fix(0)
                instance.PV2Grid[t].fix(0)
            instance.UseOfPV_rule.deactivate()
        else:
            for t in range(1, self.instance_length + 1):
                instance.PV2Load[t].fixed = False
                instance.PV2Grid[t].fixed = False
            instance.UseOfPV_rule.activate()


if __name__=="__main__":
    pass