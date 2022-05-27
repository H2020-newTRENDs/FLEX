from models.operation.abstract import OperationModel
import numpy as np
import pyomo.environ as pyo
from basics.kit import performance_counter, get_logger

logger = get_logger(__name__)


class OptOperationModel(OperationModel):

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
                "t": {None: np.arange(1, 8761)},
                "DayHour": self.create_dict(self.DayHour),

                # region
                "T_outside": self.create_dict(self.scenario.region.temperature),  # °C
                "Q_Solar": self.create_dict(self.calculate_solar_gains()),  # W

                # space heating
                "SpaceHeatingHourlyCOP": self.create_dict(self.SpaceHeatingHourlyCOP),
                "SpaceHeatingHourlyCOP_tank": self.create_dict(self.SpaceHeatingHourlyCOP_tank),
                "T_TankStart_heating": {None: self.scenario.space_heating_tank.temperature_start},
                "M_WaterTank_heating": {None: self.scenario.space_heating_tank.size},
                "U_LossTank_heating": {None: self.scenario.space_heating_tank.loss},
                "T_TankSurrounding_heating": {None: self.scenario.space_heating_tank.temperature_surrounding},
                "A_SurfaceTank_heating": {None: self.scenario.space_heating_tank.surface_area},
                "SpaceHeating_HeatPumpMaximalElectricPower": {None: self.SpaceHeating_HeatPumpMaximalElectricPower},

                # hot water
                "HotWaterHourlyCOP": self.create_dict(self.HotWaterHourlyCOP),
                "HotWaterHourlyCOP_tank": self.create_dict(self.HotWaterHourlyCOP_tank),
                "T_TankStart_DHW": {None: self.scenario.hot_water_tank.temperature_start},
                "M_WaterTank_DHW": {None: self.scenario.hot_water_tank.size},
                "U_LossTank_DHW": {None: self.scenario.hot_water_tank.loss},
                "T_TankSurrounding_DHW": {None: self.scenario.hot_water_tank.temperature_surrounding},
                "A_SurfaceTank_DHW": {None: self.scenario.hot_water_tank.surface_area},

                # space cooling
                "CoolingCOP": {None: self.scenario.space_cooling_technology.efficiency},

                # PV
                "PhotovoltaicProfile": self.create_dict(self.scenario.pv.generation),

                # battery
                "ChargeEfficiency": {None: self.scenario.battery.charge_efficiency},
                "DischargeEfficiency": {None: self.scenario.battery.discharge_efficiency},

                # EV
                "EVDemandProfile": self.create_dict(self.scenario.behavior.vehicle_demand),  # Wh
                "EVAtHomeProfile": self.create_dict(self.scenario.behavior.vehicle_at_home),  # 0 or 1
                "EVChargeEfficiency": {None: self.scenario.vehicle.charge_efficiency},
                "EVDischargeEfficiency": {None: self.scenario.vehicle.discharge_efficiency},

                # energy price
                "ElectricityPrice": self.create_dict(self.scenario.energy_price.electricity),  # C/Wh
                "FiT": self.create_dict(self.scenario.energy_price.electricity_feed_in),  # C/Wh

                # behavior
                "HotWaterProfile": self.create_dict(self.scenario.behavior.hot_water_demand),
                "BaseLoadProfile": self.create_dict(self.scenario.behavior.appliance_electricity_demand),  # Wh

                # RC model parameters
                "Am": {None: self.Am},
                "Cm": {None: self.Cm},
                "Atot": {None: self.Atot},
                "Qi": {None: self.Qi},
                "Htr_w": {None: self.scenario.building.Htr_w},
                "Htr_em": {None: self.Htr_em},
                "Htr_1": {None: self.Htr_1},
                "Htr_2": {None: self.Htr_2},
                "Htr_3": {None: self.Htr_3},
                "Hve": {None: self.scenario.building.Hve},
                "Htr_ms": {None: self.Htr_ms},
                "Htr_is": {None: self.Htr_is},
                "PHI_ia": {None: self.PHI_ia},
                "BuildingMassTemperatureStartValue": {None: self.thermal_mass_start_temperature},
            }
        }
        return pyomo_dict

    @performance_counter
    def create_abstract_model(self):
        m = pyo.AbstractModel()
        m.t = pyo.Set()

        # -------------
        # 1. Parameters
        # -------------

        # time
        m.DayHour = pyo.Param(m.t)

        # region
        m.T_outside = pyo.Param(m.t)  # °C
        m.Q_Solar = pyo.Param(m.t)  # W

        # space heating
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t)
        m.SpaceHeatingHourlyCOP_tank = pyo.Param(m.t)
        m.T_TankStart_heating = pyo.Param()
        m.M_WaterTank_heating = pyo.Param()
        m.U_LossTank_heating = pyo.Param()
        m.T_TankSurrounding_heating = pyo.Param()
        m.A_SurfaceTank_heating = pyo.Param()
        m.SpaceHeating_HeatPumpMaximalElectricPower = pyo.Param()

        # hot water
        m.HotWaterHourlyCOP = pyo.Param(m.t)
        m.HotWaterHourlyCOP_tank = pyo.Param(m.t)
        m.T_TankStart_DHW = pyo.Param()
        m.M_WaterTank_DHW = pyo.Param()
        m.U_LossTank_DHW = pyo.Param()
        m.T_TankSurrounding_DHW = pyo.Param()
        m.A_SurfaceTank_DHW = pyo.Param()

        # space cooling
        m.CoolingCOP = pyo.Param()

        # PV
        m.PhotovoltaicProfile = pyo.Param(m.t)

        # battery
        m.ChargeEfficiency = pyo.Param()
        m.DischargeEfficiency = pyo.Param()

        # EV
        m.EVDemandProfile = pyo.Param(m.t)
        m.EVAtHomeProfile = pyo.Param(m.t)
        m.EVChargeEfficiency = pyo.Param()
        m.EVDischargeEfficiency = pyo.Param()

        # energy price
        m.ElectricityPrice = pyo.Param(m.t)
        m.FiT = pyo.Param(m.t)

        # behavior
        m.HotWaterProfile = pyo.Param(m.t)
        m.BaseLoadProfile = pyo.Param(m.t)

        # RC model parameters
        m.Am = pyo.Param()
        m.Cm = pyo.Param()
        m.Atot = pyo.Param()
        m.Qi = pyo.Param()
        m.Htr_w = pyo.Param()
        m.Htr_em = pyo.Param()
        m.Htr_1 = pyo.Param()
        m.Htr_2 = pyo.Param()
        m.Htr_3 = pyo.Param()
        m.Hve = pyo.Param()
        m.Htr_ms = pyo.Param()
        m.Htr_is = pyo.Param()
        m.PHI_ia = pyo.Param()
        m.BuildingMassTemperatureStartValue = pyo.Param()

        # ----------------------------
        # 2. Variables
        # ----------------------------
        # Variables SpaceHeating
        m.Q_HeatingTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingElement = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_HeatingTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_HeatingTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_HeatingTank_bypass = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_Heating_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Q_room_heating = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Variables DHW
        m.Q_DHWTank_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_DHWTank = pyo.Var(m.t, within=pyo.NonNegativeReals)  # energy in the tank
        m.Q_DHWTank_in = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.E_DHW_HP_out = pyo.Var(m.t, within=pyo.NonNegativeReals)
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
        m.Grid2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # PV variables
        m.PV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.PV2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Electric Load and Electricity fed back to the grid
        m.Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Feed2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Battery variables
        m.BatSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.BatDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Bat2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.Bat2EV = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # EV variables
        m.EVSoC = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EVCharge = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EVDischarge = pyo.Var(m.t, within=pyo.NonNegativeReals)

        m.EV2Bat = pyo.Var(m.t, within=pyo.NonNegativeReals)
        # m.EV2Grid = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.EV2Load = pyo.Var(m.t, within=pyo.NonNegativeReals)

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
                   + m.Q_RoomCooling[t] / m.CoolingCOP \
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
            return m.EV2Load[t] * m.EVAtHomeProfile[t] + m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] == m.Load[t]

        m.SupplyOfLoads_rule = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

        # Battery:
        # (6) Battery charge
        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t] + m.EV2Bat[t] * m.EVAtHomeProfile[t]

        m.BatCharge_rule = pyo.Constraint(m.t, rule=calc_BatCharge)

        # (6a) EV charge
        def calc_EVCharge(m, t):
            return m.EVCharge[t] == m.PV2EV[t] * m.EVAtHomeProfile[t] + m.Grid2EV[t] * m.EVAtHomeProfile[t] \
                   + m.Bat2EV[t] * m.EVAtHomeProfile[t]

        m.EVCharge_rule = pyo.Constraint(m.t, rule=calc_EVCharge)

        # (7) Battery discharge
        def calc_BatDischarge(m, t):
            if m.t[t] == 1:
                return m.BatDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.BatDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.EVAtHomeProfile[t]

        m.BatDischarge_rule = pyo.Constraint(m.t, rule=calc_BatDischarge)

        # (7a) EV discharge with bidirectional charge --> V2B
        def calc_EVDischarge(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.EVDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.EVDischarge[t] == m.EV2Load[t] * m.EVAtHomeProfile[t] + m.EV2Bat[t] * m.EVAtHomeProfile[t] \
                       + m.EVDemandProfile[t]

        m.EVDischarge_rule = pyo.Constraint(m.t, rule=calc_EVDischarge)

        # (7b) EV discharge without bidirectional charge --> no V2B
        def calc_EVDischarge_noV2B(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.EVDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.EVDischarge[t] == m.EVDemandProfile[t]

        m.EVDischarge_noV2B_rule = pyo.Constraint(m.t, rule=calc_EVDischarge_noV2B)

        # (8) Battery SOC
        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == 0  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * m.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - m.DischargeEfficiency))

        m.BatSoC_rule = pyo.Constraint(m.t, rule=calc_BatSoC)

        # (8a) EV SOC
        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0  # start of simulation, EV is empty
            else:
                return m.EVSoC[t] == m.EVSoC[t - 1] + m.EVCharge[t] * m.EVChargeEfficiency * m.EVAtHomeProfile[t] - \
                       m.EVDischarge[t] * (1 + (1 - m.EVDischargeEfficiency))

        m.EVSoC_rule = pyo.Constraint(m.t, rule=calc_EVSoC)

        # Heating Tank:
        # (9) energy in the tank
        def tank_energy_heating(m, t):
            if t == 1:
                return m.E_HeatingTank[t] == self.cp_water * m.M_WaterTank_heating * (273.15 + m.T_TankStart_heating) - \
                       m.Q_HeatingTank_out[t]
            else:
                return m.E_HeatingTank[t] == m.E_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] \
                       - m.U_LossTank_heating * m.A_SurfaceTank_heating * (
                               (m.E_HeatingTank[t] / (m.M_WaterTank_heating * self.cp_water)) -
                               (m.T_TankSurrounding_heating + 273.15))

        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

        # (10) room heating
        def room_heating(m, t):
            return m.Q_room_heating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t] + m.Q_HeatingElement[t]

        m.room_heating_rule = pyo.Constraint(m.t, rule=room_heating)

        # (11) supply for space heating
        def calc_supply_of_space_heating(m, t):
            return m.Q_HeatingTank_bypass[t] / m.SpaceHeatingHourlyCOP[t] + \
                   m.Q_HeatingTank_in[t] / m.SpaceHeatingHourlyCOP_tank[t] == m.E_Heating_HP_out[t]

        m.calc_use_of_HP_power_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_space_heating)

        # DHW:
        # (12) energy in DHW tank
        def tank_energy_DHW(m, t):
            if t == 1:
                return m.E_DHWTank[t] == self.cp_water * m.M_WaterTank_DHW * (273.15 + m.T_TankStart_DHW) - m.Q_DHWTank_out[t]
            else:
                return m.E_DHWTank[t] == m.E_DHWTank[t - 1] - m.Q_DHWTank_out[t] + m.Q_DHWTank_in[t] - \
                       m.U_LossTank_DHW * m.A_SurfaceTank_DHW * (
                               m.E_DHWTank[t] / (m.M_WaterTank_DHW * self.cp_water) - (m.T_TankSurrounding_DHW + 273.15)
                       )

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (13) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.SupplyOfDHW_rule = pyo.Constraint(m.t, rule=calc_hot_water_profile)

        # (14) supply for the hot water
        def calc_supply_of_DHW(m, t):
            return m.Q_DHWTank_bypass[t] / m.HotWaterHourlyCOP[t] + \
                   m.Q_DHWTank_in[t] / m.HotWaterHourlyCOP_tank[t] == m.E_DHW_HP_out[t]

        m.bypass_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW)

        # Heat Pump
        # (13) restrict heat pump power
        def constrain_heating_max_power(m, t):
            return m.E_DHW_HP_out[t] + m.E_Heating_HP_out[t] <= m.SpaceHeating_HeatPumpMaximalElectricPower

        m.max_HP_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power)

        # ----------------------------------------------------
        # 6. State transition: Building temperature (RC-Model)
        # ----------------------------------------------------

        def thermal_mass_temperature_rc(m, t):
            if t == 1:
                Tm_start = m.BuildingMassTemperatureStartValue
            else:
                Tm_start = m.Tm_t[t - 1]
            # Equ. C.2
            PHI_m = m.Am / m.Atot * (0.5 * m.Qi + m.Q_Solar[t])
            # Equ. C.3
            PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = m.T_outside[t]
            # Equ. C.5
            PHI_mtot = PHI_m + m.Htr_em * m.T_outside[t] + m.Htr_3 * \
                       (PHI_st + m.Htr_w * m.T_outside[t] +
                        m.Htr_1 * (((m.PHI_ia + m.Q_room_heating[t] -
                                     m.Q_RoomCooling[t]) / m.Hve) + T_sup)) / m.Htr_2
            # Equ. C.4
            return m.Tm_t[t] == (Tm_start * ((m.Cm / 3600) - 0.5 * (m.Htr_3 + m.Htr_em)) + PHI_mtot) / (
                           (m.Cm / 3600) + 0.5 * (m.Htr_3 + m.Htr_em))

        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

        def room_temperature_rc(m, t):
            if t == 1:
                Tm_start = m.BuildingMassTemperatureStartValue
            else:
                Tm_start = m.Tm_t[t - 1]
            # Equ. C.3
            PHI_st = (1 - m.Am / m.Atot - m.Htr_w / 9.1 / m.Atot) * (0.5 * m.Qi + m.Q_Solar[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + Tm_start) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (m.Htr_ms * T_m + PHI_st + m.Htr_w * m.T_outside[t] + m.Htr_1 *
                   (T_sup + (m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / m.Hve)) / \
                  (m.Htr_ms + m.Htr_w + m.Htr_1)
            # Equ. C.11
            T_air = (m.Htr_is * T_s + m.Hve * T_sup + m.PHI_ia + m.Q_room_heating[t] - m.Q_RoomCooling[t]) / \
                    (m.Htr_is + m.Hve)
            return m.T_room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

        # ------------
        # 7. Objective
        # ------------

        def minimize_cost(m):
            rule = sum(m.Grid[t] * m.ElectricityPrice[t] - m.Feed2Grid[t] * m.FiT[t] for t in m.t)
            return rule

        m.total_operation_cost_rule = pyo.Objective(rule=minimize_cost, sense=pyo.minimize)
        return m

    @performance_counter
    def config_instance(self, instance):

        # -------------
        # 1. Boundaries
        # -------------
        for t in range(1, 8761):
            instance.Q_HeatingElement[t].setub(self.scenario.boiler.heating_element_power)
            instance.T_room[t].setlb(self.scenario.behavior.target_temperature_array_min[t - 1])
            instance.Tm_t[t].setub(100)
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
                instance.T_room[t].setub(no_cooling_target_temperature_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.activate()
            instance.SumOfLoads_with_cooling_rule.deactivate()

        else:
            for t in range(1, 8761):
                instance.Q_RoomCooling[t].fixed = False
                instance.Q_RoomCooling[t].setub(self.scenario.space_cooling_technology.power)
                instance.T_room[t].setub(self.scenario.behavior.target_temperature_array_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.deactivate()
            instance.SumOfLoads_with_cooling_rule.activate()

        # Thermal storage Heating
        if self.scenario.space_heating_tank.size == 0:
            for t in range(1, 8761):
                instance.E_HeatingTank[t].fix(0)
                instance.Q_HeatingTank_out[t].fix(0)
                instance.Q_HeatingTank_in[t].fix(0)

                instance.E_Heating_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

            instance.tank_energy_rule_heating.deactivate()
        else:
            for t in range(1, 8761):
                instance.E_HeatingTank[t].fixed = False
                instance.Q_HeatingTank_out[t].fixed = False
                instance.Q_HeatingTank_in[t].fixed = False

                instance.E_HeatingTank[t].setlb(
                    self.cp_water * self.scenario.space_heating_tank.size *
                    (273.15 + self.scenario.space_heating_tank.temperature_min)
                )

                instance.E_HeatingTank[t].setub(
                    self.cp_water * self.scenario.space_heating_tank.size *
                    (273.15 + self.scenario.space_heating_tank.temperature_max)
                )
                instance.E_Heating_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

            instance.tank_energy_rule_heating.activate()

        # Thermal storage DHW
        if self.scenario.hot_water_tank.size == 0:
            for t in range(1, 8761):
                instance.E_DHWTank[t].fix(0)
                instance.Q_DHWTank_out[t].fix(0)
                instance.Q_DHWTank_in[t].fix(0)

                instance.E_DHW_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)
            instance.tank_energy_rule_DHW.deactivate()
        else:
            for t in range(1, 8761):
                instance.E_DHWTank[t].fixed = False
                instance.Q_DHWTank_out[t].fixed = False
                instance.Q_DHWTank_in[t].fixed = False
                instance.E_DHWTank[t].setlb(
                    self.cp_water * self.scenario.hot_water_tank.size *
                    (273.15 + self.scenario.hot_water_tank.temperature_min)
                )
                instance.E_DHWTank[t].setub(
                    self.cp_water * self.scenario.hot_water_tank.size *
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
            instance.EVDischarge_noV2B_rule.deactivate()

        # EV is implemented but can not provide electricity to the household
        elif self.scenario.vehicle.charge_bidirectional == 0 and self.scenario.vehicle.capacity > 0:
            # check if building has battery:
            if self.scenario.battery.capacity == 0:  # no battery to EV possible
                for t in range(1, 8760):
                    instance.Bat2EV[t].fix(0)
            else:  # battery to EV is possible
                for t in range(1, 8760):
                    instance.Bat2EV[t].fixed = False
            for t in range(1, 8761):
                # variables have to be unfixed in case they were fixed in a previous run
                instance.Grid2EV[t].fixed = False
                instance.PV2EV[t].fixed = False
                instance.EVSoC[t].fixed = False
                instance.EVCharge[t].fixed = False
                instance.EVDischarge[t].fixed = False
                instance.EV2Load[t].fix(0)
                instance.EV2Bat[t].fix(0)
                # set upper bounds
                instance.Grid2EV[t].setub(self.scenario.vehicle.charge_power_max)
                instance.PV2EV[t].setub(self.scenario.vehicle.charge_power_max)
                instance.EVSoC[t].setub(self.scenario.vehicle.capacity)
                instance.EVCharge[t].setub(self.scenario.vehicle.charge_power_max)
                # vehicle does not get discharged at home -> no upper bound

            instance.EVCharge_rule.activate()
            instance.EVDischarge_rule.deactivate()
            instance.EVSoC_rule.activate()
            instance.EVDischarge_noV2B_rule.activate()

        # EV can charge battery and supply electricity to house
        else:
            # in case there is a stationary battery available:
            if self.scenario.battery.capacity > 0:
                max_discharge_EV = self.create_upper_bound_EV_discharge()
                for t in range(1, 8761):
                    # variables have to be unfixed in case they were fixed in a previous run
                    instance.Grid2EV[t].fixed = False
                    instance.EV2Load[t].fixed = False
                    instance.PV2EV[t].fixed = False
                    instance.EVSoC[t].fixed = False
                    instance.EVCharge[t].fixed = False
                    instance.EVDischarge[t].fixed = False
                    instance.EV2Bat[t].fixed = False
                    instance.Bat2EV[t].fixed = False

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
                instance.EVDischarge_noV2B_rule.deactivate()
            # in case there is no stationary battery available:
            if self.scenario.battery.capacity == 0:
                max_discharge_EV = self.create_upper_bound_EV_discharge()
                for t in range(1, 8761):
                    # variables have to be unfixed in case they were fixed in a previous run
                    instance.Grid2EV[t].fixed = False
                    instance.EV2Load[t].fixed = False
                    instance.PV2EV[t].fixed = False
                    instance.EVSoC[t].fixed = False
                    instance.EVCharge[t].fixed = False
                    instance.EVDischarge[t].fixed = False
                    instance.EV2Bat[t].fix(0)
                    instance.Bat2EV[t].fix(0)

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
                instance.EVDischarge_noV2B_rule.deactivate()

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
    def solve_optimization(self, instance2solve):
        logger.info("Solving optimization...")
        pyo.SolverFactory("gurobi").solve(instance2solve, tee=False)
        return instance2solve

    def run(self):
        model = self.create_abstract_model()
        instance = self.config_instance(model.create_instance(data=self.params_dict))
        solved_instance = self.solve_optimization(instance)
        logger.info(f'Optimal operation cost: {round(solved_instance.total_operation_cost_rule(), 2)}')
        return solved_instance
