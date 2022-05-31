import copy

import pyomo.environ as pyo
import numpy as np

from models.operation.model_base import OperationModel
from models.operation.enums import ResultEnum
from basics.kit import performance_counter, get_logger

logger = get_logger(__name__)


class OptOperationModel(OperationModel):

    @performance_counter
    def setup_abstract_model(self):
        m = pyo.AbstractModel()
        m.t = pyo.Set()

        # -------------
        # 1. Parameters
        # -------------

        # region
        m.T_outside = pyo.Param(m.t, mutable=True)  # °C
        m.Q_Solar = pyo.Param(m.t, mutable=True)  # W
        # space heating
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t)
        m.SpaceHeatingHourlyCOP_tank = pyo.Param(m.t)
        # hot water
        m.HotWaterHourlyCOP = pyo.Param(m.t)
        m.HotWaterHourlyCOP_tank = pyo.Param(m.t)
        # space cooling
        m.CoolingHourlyCOP = pyo.Param(m.t)
        # PV
        m.PhotovoltaicProfile = pyo.Param(m.t)
        # battery
        # EV
        m.EVDemandProfile = pyo.Param(m.t)
        m.EVAtHomeProfile = pyo.Param(m.t)
        # energy price
        m.ElectricityPrice = pyo.Param(m.t)
        m.FiT = pyo.Param(m.t)
        # behavior
        m.HotWaterProfile = pyo.Param(m.t, mutable=True)
        m.BaseLoadProfile = pyo.Param(m.t)

        # ----------------------------
        # 2. Variables
        # ----------------------------

        # space heating
        m.Q_HeatingTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_HeatingTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_RoomHeating = pyo.Var(m.t, within=pyo.NonNegativeReals)
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
        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Feed2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # -------------------------------------
        # 3. Constraints:
        # -------------------------------------

        # (1) grid balance
        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t] + m.Grid2EV[t] * m.EVAtHomeProfile[t]

        m.UseOfGrid_rule = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        # (2) PV balance
        def calc_UseOfPV(m, t):
            return m.PV2EV[t] * m.EVAtHomeProfile[t] + m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t] == \
                   m.PhotovoltaicProfile[t]

        m.UseOfPV_rule = pyo.Constraint(m.t, rule=calc_UseOfPV)

        # (3) Feed in
        def calc_SumOfFeedin(m, t):
            return m.Feed2Grid[t] == m.PV2Grid[t]

        m.SumOfFeedin_rule = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

        # (4) Sum of Loads
        def calc_SumOfLoads_with_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.E_Heating_HP_out[t] \
                   + m.Q_HeatingElement[t] \
                   + m.Q_RoomCooling[t] / self.CoolingCOP \
                   + m.E_DHW_HP_out[t]

        m.SumOfLoads_with_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

        def calc_SumOfLoads_without_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] \
                   + m.E_Heating_HP_out[t] \
                   + m.Q_HeatingElement[t] \
                   + m.E_DHW_HP_out[t]

        m.SumOfLoads_without_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_without_cooling)

        # (5) load coverage
        def calc_SupplyOfLoads(m, t):
            return m.EV2Load[t] * m.EVAtHomeProfile[t] * self.EVOptionV2B + \
                   m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

        m.SupplyOfLoads_rule = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # Battery:
        # (6a) Battery charge
        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t] + m.EV2Bat[t] * m.EVAtHomeProfile[t] * self.EVOptionV2B

        m.BatCharge_rule = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (6b) EV charge
        def calc_EVCharge(m, t):
            return m.EVCharge[t] == m.PV2EV[t] * m.EVAtHomeProfile[t] + m.Grid2EV[t] * m.EVAtHomeProfile[t] \
                   + m.Bat2EV[t] * m.EVAtHomeProfile[t]

        m.EVCharge_rule = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (7a) Battery discharge
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.EVAtHomeProfile[t]

        m.BatDischarge_rule = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (7b) EV discharge
        def calc_EVDischarge(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.EVDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.EVDischarge[t] == (m.EV2Load[t] + m.EV2Bat[t]) * m.EVAtHomeProfile[t] * self.EVOptionV2B \
                       + m.EVDemandProfile[t]

        m.EVDischarge_rule = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (8) Battery SOC
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * self.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - self.DischargeEfficiency))

        m.BatSoC_rule = pyo.Constraint(m.t, rule=calc_BatSoC)

        # (8a) EV SOC
        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0  # start of simulation, EV is empty
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + m.EVCharge[t] * self.EVChargeEfficiency * m.EVAtHomeProfile[t] - \
                       m.EVDischarge[t] * (1 + (1 - self.EVDischargeEfficiency))

        m.EVSoC_rule = pyo.Constraint(m.t, rule=calc_EVSoC)

        # Space heating and cooling:
        # (9) energy in the space heating tank
        def tank_energy_heating(m, t):
            if t == 1:
                return m.Q_HeatingTank[t] == self.CPWater * self.M_WaterTank_heating * \
                       (273.15 + self.T_TankStart_heating) - m.Q_HeatingTank_out[t]
            else:
                return m.Q_HeatingTank[t] == m.Q_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] \
                       - self.U_LossTank_heating * self.A_SurfaceTank_heating * (
                               (m.Q_HeatingTank[t] / (self.M_WaterTank_heating * self.CPWater)) -
                               (self.T_TankSurrounding_heating + 273.15))

        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

        # (10) space heating
        def room_heating(m, t):
            return m.Q_RoomHeating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t] + m.Q_HeatingElement[t]

        m.room_heating_rule = pyo.Constraint(m.t, rule=room_heating)

        # (11) supply for space heating
        def calc_supply_of_space_heating(m, t):
            return m.Q_HeatingTank_bypass[t] / m.SpaceHeatingHourlyCOP[t] + \
                   m.Q_HeatingTank_in[t] / m.SpaceHeatingHourlyCOP_tank[t] == m.E_Heating_HP_out[t]

        m.calc_use_of_HP_power_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_space_heating)

        # (12) space cooling
        def calc_E_RoomCooling_with_cooling(m, t):
            return m.E_RoomCooling[t] == m.Q_RoomCooling[t] / self.CoolingCOP

        m.E_RoomCooling_with_cooling_rule = pyo.Constraint(m.t, rule=calc_E_RoomCooling_with_cooling)

        # how water
        # (13) energy in DHW tank
        def tank_energy_DHW(m, t):
            if t == 1:
                return m.Q_DHWTank[t] == self.CPWater * self.M_WaterTank_DHW * \
                       (273.15 + self.T_TankStart_DHW) - m.Q_DHWTank_out[t]
            else:
                return m.Q_DHWTank[t] == m.Q_DHWTank[t - 1] - m.Q_DHWTank_out[t] + m.Q_DHWTank_in[t] - \
                       self.U_LossTank_DHW * self.A_SurfaceTank_DHW * \
                       (m.Q_DHWTank[t] / (self.M_WaterTank_DHW * self.CPWater) - (self.T_TankSurrounding_DHW + 273.15))

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (14) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.SupplyOfDHW_rule = pyo.Constraint(m.t, rule=calc_hot_water_profile)

        # (15) supply for the hot water
        def calc_supply_of_DHW(m, t):
            return m.Q_DHWTank_bypass[t] / m.HotWaterHourlyCOP[t] + \
                   m.Q_DHWTank_in[t] / m.HotWaterHourlyCOP_tank[t] == m.E_DHW_HP_out[t]

        m.bypass_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW)

        # Heat Pump
        # (16) restrict heat pump power
        def constrain_heating_max_power(m, t):
            return m.E_DHW_HP_out[t] + m.E_Heating_HP_out[t] <= self.SpaceHeating_HeatPumpMaximalElectricPower

        m.max_HP_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power)

        # ----------------------------------------------------
        # 6. State transition: Building temperature (RC-Model)
        # ----------------------------------------------------

        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                Tm_start = self.BuildingMassTemperatureStartValue
            else:
                Tm_start = m.T_BuildingMass[t - 1]
            # Equ. C.2
            PHI_m = self.Am / self.Atot * (0.5 * self.Qi + m.Q_Solar[t])
            # Equ. C.3
            PHI_st = (1 - self.Am / self.Atot - self.Htr_w / 9.1 / self.Atot) * (0.5 * self.Qi + m.Q_Solar[t])
            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = m.T_outside[t]
            # Equ. C.5
            PHI_mtot = PHI_m + self.Htr_em * m.T_outside[t] + self.Htr_3 * \
                       (PHI_st + self.Htr_w * m.T_outside[t] +
                        self.Htr_1 * (((self.PHI_ia + m.Q_RoomHeating[t] -
                                     m.Q_RoomCooling[t]) / self.Hve) + T_sup)) / self.Htr_2
            # Equ. C.4
            return m.T_BuildingMass[t] == (Tm_start * ((self.Cm / 3600) - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot) \
                   / ((self.Cm / 3600) + 0.5 * (self.Htr_3 + self.Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                Tm_start = self.BuildingMassTemperatureStartValue
            else:
                Tm_start = m.T_BuildingMass[t - 1]
            # Equ. C.3
            PHI_st = (1 - self.Am / self.Atot - self.Htr_w / 9.1 / self.Atot) * (0.5 * self.Qi + m.Q_Solar[t])
            # Equ. C.9
            T_m = (m.T_BuildingMass[t] + Tm_start) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (self.Htr_ms * T_m + PHI_st + self.Htr_w * m.T_outside[t] + self.Htr_1 *
                   (T_sup + (self.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / self.Hve)) / \
                  (self.Htr_ms + self.Htr_w + self.Htr_1)
            # Equ. C.11
            T_air = (self.Htr_is * T_s + self.Hve * T_sup + self.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / \
                    (self.Htr_is + self.Hve)
            return m.T_Room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        # ------------
        # 7. Objective
        # ------------

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feed2Grid[t] * m.FiT[t] for t in m.t)
            return rule

        m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)
        return m

    @staticmethod
    def create_dict(values) -> dict:
        dictionary = {}
        for index, value in enumerate(values, start=1):
            dictionary[index] = value
        return dictionary

    @property
    def params_dict(self) -> dict:
        pyomo_dict = {
            None: {
                # time
                "t": {None: self.Hour},
                # region
                "T_outside": self.create_dict(self.T_outside),  # °C
                "Q_Solar": self.create_dict(self.Q_Solar),  # W
                # space heating
                "SpaceHeatingHourlyCOP": self.create_dict(self.SpaceHeatingHourlyCOP),
                "SpaceHeatingHourlyCOP_tank": self.create_dict(self.SpaceHeatingHourlyCOP_tank),
                # hot water
                "HotWaterHourlyCOP": self.create_dict(self.HotWaterHourlyCOP),
                "HotWaterHourlyCOP_tank": self.create_dict(self.HotWaterHourlyCOP_tank),
                # space cooling
                "CoolingHourlyCOP": self.create_dict(self.CoolingHourlyCOP),
                # PV
                "PhotovoltaicProfile": self.create_dict(self.PhotovoltaicProfile),
                # battery
                # EV
                "EVDemandProfile": self.create_dict(self.EVDemandProfile),  # Wh
                "EVAtHomeProfile": self.create_dict(self.EVAtHomeProfile),  # 0 or 1
                # energy price
                "ElectricityPrice": self.create_dict(self.ElectricityPrice),  # C/Wh
                "FiT": self.create_dict(self.FiT),  # C/Wh
                # behavior
                "HotWaterProfile": self.create_dict(self.HotWaterProfile),
                "BaseLoadProfile": self.create_dict(self.BaseLoadProfile),  # Wh
            }
        }
        return pyomo_dict

    @performance_counter
    def config_instance(self, instance):

        # -------------
        # 1. Boundaries
        # -------------
        for t in range(1, 8761):
            instance.Q_HeatingElement[t].setub(self.scenario.boiler.heating_element_power)
            instance.T_Room[t].setlb(self.scenario.behavior.target_temperature_array_min[t - 1])
            instance.T_BuildingMass[t].setub(100)
            instance.Grid[t].setub(self.scenario.building.grid_power_max)
            instance.Grid2Load[t].setub(self.scenario.building.grid_power_max)
            instance.Load[t].setub(self.scenario.building.grid_power_max)
            instance.Feed2Grid[t].setub(self.scenario.building.grid_power_max)

        # ---------------
        # 2. Technologies
        # ---------------
        # Room Cooling:
        if self.scenario.space_cooling_technology.power == 0:
            no_cooling_target_temperature_max = self.generate_maximum_target_indoor_temperature_no_cooling(
                temperature_max=23  # TODO take this temp to config
            )
            for t in range(1, 8761):
                instance.Q_RoomCooling[t].fix(0)
                instance.E_RoomCooling[t].fix(0)
                instance.T_Room[t].setub(no_cooling_target_temperature_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.activate()
            instance.SumOfLoads_with_cooling_rule.deactivate()
            instance.E_RoomCooling_with_cooling_rule.deactivate()

        else:
            for t in range(1, 8761):
                instance.Q_RoomCooling[t].fixed = False
                instance.E_RoomCooling[t].fixed = False
                instance.Q_RoomCooling[t].setub(self.scenario.space_cooling_technology.power)
                instance.T_Room[t].setub(self.scenario.behavior.target_temperature_array_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.deactivate()
            instance.SumOfLoads_with_cooling_rule.activate()
            instance.E_RoomCooling_with_cooling_rule.activate()

        # Thermal storage Heating
        if self.scenario.space_heating_tank.size == 0:
            for t in range(1, 8761):
                instance.Q_HeatingTank[t].fix(0)
                instance.Q_HeatingTank_out[t].fix(0)
                instance.Q_HeatingTank_in[t].fix(0)
                instance.E_Heating_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

            instance.tank_energy_rule_heating.deactivate()
        else:
            for t in range(1, 8761):
                instance.Q_HeatingTank[t].fixed = False
                instance.Q_HeatingTank_out[t].fixed = False
                instance.Q_HeatingTank_in[t].fixed = False

                instance.Q_HeatingTank[t].setlb(
                    self.CPWater * self.scenario.space_heating_tank.size *
                    (273.15 + self.scenario.space_heating_tank.temperature_min)
                )

                instance.Q_HeatingTank[t].setub(
                    self.CPWater * self.scenario.space_heating_tank.size *
                    (273.15 + self.scenario.space_heating_tank.temperature_max)
                )
                instance.E_Heating_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

            instance.tank_energy_rule_heating.activate()

        # Thermal storage DHW
        if self.scenario.hot_water_tank.size == 0:
            for t in range(1, 8761):
                instance.Q_DHWTank[t].fix(0)
                instance.Q_DHWTank_out[t].fix(0)
                instance.Q_DHWTank_in[t].fix(0)

                instance.E_DHW_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)
            instance.tank_energy_rule_DHW.deactivate()
        else:
            for t in range(1, 8761):
                instance.Q_DHWTank[t].fixed = False
                instance.Q_DHWTank_out[t].fixed = False
                instance.Q_DHWTank_in[t].fixed = False
                instance.Q_DHWTank[t].setlb(
                    self.CPWater * self.scenario.hot_water_tank.size *
                    (273.15 + self.scenario.hot_water_tank.temperature_min)
                )
                instance.Q_DHWTank[t].setub(
                    self.CPWater * self.scenario.hot_water_tank.size *
                    (273.15 + self.scenario.hot_water_tank.temperature_max)
                )
                instance.E_DHW_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)
            instance.tank_energy_rule_DHW.activate()

        # Battery
        if self.scenario.battery.capacity == 0:
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
                instance.Grid2Bat[t].setub(self.scenario.battery.charge_power_max)
                instance.Bat2Load[t].setub(self.scenario.battery.discharge_power_max)
                instance.BatSoC[t].setub(self.scenario.battery.capacity)
                instance.BatCharge[t].setub(self.scenario.battery.charge_power_max)
                instance.BatDischarge[t].setub(self.scenario.battery.discharge_power_max)
            instance.BatCharge_rule.activate()
            instance.BatDischarge_rule.activate()
            instance.BatSoC_rule.activate()

        # EV
        # no EV is implemented
        if self.scenario.vehicle.capacity == 0:
            for t in range(1, 8761):
                # fix the parameters to 0
                instance.Grid2EV[t].fix(0)
                instance.Bat2EV[t].fix(0)
                instance.PV2EV[t].fix(0)
                instance.EVSoC[t].fix(0)
                instance.EVCharge[t].fix(0)
                instance.EVDischarge[t].fix(0)
                instance.EV2Load[t].fix(0)
                instance.EV2Bat[t].fix(0)

            instance.EVCharge_rule.deactivate()
            instance.EVDischarge_rule.deactivate()
            instance.EVSoC_rule.deactivate()
        else:
            # in case there is a stationary battery available:
            if self.scenario.battery.capacity > 0:
                max_discharge_EV = self.create_upper_bound_EV_discharge()
                for t in range(1, 8761):
                    # variables have to be unfixed in case they were fixed in a previous run
                    instance.Grid2EV[t].fixed = False
                    instance.PV2EV[t].fixed = False
                    instance.EVSoC[t].fixed = False
                    instance.EVCharge[t].fixed = False
                    instance.EVDischarge[t].fixed = False
                    instance.Bat2EV[t].fixed = False
                    if self.EVOptionV2B == 0:
                        instance.EV2Bat[t].fix(0)
                        instance.EV2Load[t].fix(0)
                    else:
                        instance.EV2Bat[t].fixed = False
                        instance.EV2Load[t].fixed = False

                    # set upper bounds
                    instance.Grid2EV[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EV2Load[t].setub(self.scenario.vehicle.discharge_power_max)
                    instance.PV2EV[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EVSoC[t].setub(self.scenario.vehicle.capacity)
                    instance.EVCharge[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EVDischarge[t].setub(max_discharge_EV[t - 1])
                instance.EVCharge_rule.activate()
                instance.EVDischarge_rule.activate()
                instance.EVSoC_rule.activate()

            # in case there is no stationary battery available:
            if self.scenario.battery.capacity == 0:
                max_discharge_EV = self.create_upper_bound_EV_discharge()
                for t in range(1, 8761):
                    # variables have to be unfixed in case they were fixed in a previous run
                    instance.Grid2EV[t].fixed = False
                    instance.PV2EV[t].fixed = False
                    instance.EVSoC[t].fixed = False
                    instance.EVCharge[t].fixed = False
                    instance.EVDischarge[t].fixed = False
                    instance.EV2Bat[t].fix(0)
                    instance.Bat2EV[t].fix(0)
                    if self.EVOptionV2B == 0:
                        instance.EV2Load[t].fix(0)
                    else:
                        instance.EV2Load[t].fixed = False

                    # set upper bounds
                    instance.Grid2EV[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EV2Load[t].setub(self.scenario.vehicle.discharge_power_max)
                    instance.PV2EV[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EVSoC[t].setub(self.scenario.vehicle.capacity)
                    instance.EVCharge[t].setub(self.scenario.vehicle.charge_power_max)
                    instance.EVDischarge[t].setub(max_discharge_EV[t - 1])
                instance.EVCharge_rule.activate()
                instance.EVDischarge_rule.activate()
                instance.EVSoC_rule.activate()

        # PV
        if self.scenario.pv.size == 0:
            for t in range(1, 8761):
                instance.PV2Load[t].fix(0)
                instance.PV2Bat[t].fix(0)
                instance.PV2Grid[t].fix(0)
                instance.PV2EV[t].fix(0)
            instance.UseOfPV_rule.deactivate()

        else:
            for t in range(1, 8761):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.PV2Load[t].fixed = False
                instance.PV2Bat[t].fixed = False
                instance.PV2Grid[t].fixed = False
                instance.PV2EV[t].fixed = False
                # set upper bounds
                instance.PV2Load[t].setub(self.scenario.building.grid_power_max)
                instance.PV2Bat[t].setub(self.scenario.building.grid_power_max)
                instance.PV2EV[t].setub(self.scenario.building.grid_power_max)

            instance.UseOfPV_rule.activate()

        return instance

    @performance_counter
    def setup_instance(self, model):
        instance = self.config_instance(model.create_instance(data=self.params_dict))
        return instance

    @performance_counter
    def solve_instance(self, instance2solve):
        logger.info("Solving optimization...")
        pyo.SolverFactory("gurobi").solve(instance2solve, tee=False)
        return instance2solve

    def run_fuel_boiler_opt(self):
        outside_temperature, q_solar, hot_water_demand = self.fuel_boiler_save_scenario()
        heating_cost, cooling_cost, heating_demand, cooling_demand, room_temperature, building_mass_temperature = \
            self.fuel_boiler_heating_cooling()
        hot_water_cost = self.fuel_boiler_hot_water()
        solved_instance = self.run_heatpump_opt()
        solved_instance.__dict__['T_outside'].store_values(self.create_dict(outside_temperature))
        solved_instance.__dict__['Q_Solar'].store_values(self.create_dict(q_solar))
        solved_instance.__dict__['Q_RoomHeating'].set_values(self.create_dict(heating_demand))
        solved_instance.__dict__['Q_RoomCooling'].set_values(self.create_dict(cooling_demand))
        solved_instance.__dict__['E_RoomCooling'].set_values(self.create_dict(cooling_demand / self.CoolingCOP))
        solved_instance.__dict__['T_Room'].set_values(self.create_dict(room_temperature))
        solved_instance.__dict__['T_BuildingMass'].set_values(self.create_dict(building_mass_temperature))
        solved_instance.__dict__['HotWaterProfile'].store_values(self.create_dict(hot_water_demand))
        solved_instance.__dict__['total_operation_cost_rule'] += (heating_cost + cooling_cost + hot_water_cost)
        return solved_instance

    def fuel_boiler_save_scenario(self):
        outside_temperature = copy.deepcopy(self.T_outside)
        q_solar = copy.deepcopy(self.Q_Solar)
        hot_water_demand = copy.deepcopy(self.HotWaterProfile)
        return outside_temperature, q_solar, hot_water_demand

    def fuel_boiler_heating_cooling(self):
        fuel_type: str = self.scenario.boiler.type
        fuel_price = self.scenario.energy_price.__dict__[fuel_type]
        electricity_price = self.scenario.energy_price.electricity
        heating_demand, cooling_demand, room_temperature, building_mass_temperature = \
            self.calculate_heating_and_cooling_demand()
        heating_cost = (fuel_price * heating_demand).sum()
        cooling_cost = (electricity_price * cooling_demand / self.CoolingCOP).sum()
        self.T_outside = 24 * np.ones(8760, )
        self.Q_Solar = np.zeros(8760, )
        return heating_cost, cooling_cost, heating_demand, cooling_demand, room_temperature, building_mass_temperature

    def fuel_boiler_hot_water(self):
        fuel_type: str = self.scenario.boiler.type
        fuel_price = self.scenario.energy_price.__dict__[fuel_type]
        hot_water_cost = (fuel_price * self.HotWaterProfile).sum()
        self.HotWaterProfile = np.zeros(8760, )
        return hot_water_cost

    def run_heatpump_opt(self):
        abstract_model = self.setup_abstract_model()
        instance = self.setup_instance(abstract_model)
        solved_instance = self.solve_instance(instance)
        logger.info(f'OptCost: {round(solved_instance.total_operation_cost_rule(), 2)}')
        return solved_instance

    def run(self):
        if self.scenario.boiler.type in ["Air_HP", "Ground_HP"]:
            solved_instance = self.run_heatpump_opt()
        else:
            solved_instance = self.run_fuel_boiler_opt()
        return solved_instance

