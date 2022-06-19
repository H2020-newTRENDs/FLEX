import pyomo.environ as pyo
from models.operation.model_base import OperationModel
from basics.kit import performance_counter, get_logger

logger = get_logger(__name__)


class OptOperationModel(OperationModel):

    def setup_model(self):
        m = pyo.AbstractModel()
        self.setup_sets(m)
        self.setup_params(m)
        self.setup_variables(m)
        self.setup_constraint_space_heating_tank(m)
        self.setup_constraint_space_heating_room(m)
        self.setup_constraint_thermal_mass_temperature(m)
        self.setup_constraint_room_temperature(m)
        self.setup_constraint_hot_water(m)
        self.setup_constraint_heat_pump(m)
        self.setup_constraint_space_cooling(m)
        self.setup_constraint_pv(m)
        self.setup_constraint_battery(m)
        self.setup_constraint_ev(m)
        self.setup_constraint_electricity_demand(m)
        self.setup_constraint_electricity_supply(m)
        self.setup_objective(m)
        return m

    def setup_sets(self, m):
        m.t = pyo.Set(initialize=self.Hour)

    def setup_params(self, m):
        # region
        m.T_outside = pyo.Param(m.t, initialize=self.create_dict(self.T_outside), mutable=True)  # °C
        m.Q_Solar = pyo.Param(m.t, initialize=self.create_dict(self.Q_Solar), mutable=True)  # W
        # space heating
        m.SpaceHeatingHourlyCOP = pyo.Param(m.t, initialize=self.create_dict(self.SpaceHeatingHourlyCOP))
        m.SpaceHeatingHourlyCOP_tank = pyo.Param(m.t, initialize=self.create_dict(self.SpaceHeatingHourlyCOP_tank))
        # hot water
        m.HotWaterHourlyCOP = pyo.Param(m.t, initialize=self.create_dict(self.HotWaterHourlyCOP))
        m.HotWaterHourlyCOP_tank = pyo.Param(m.t, initialize=self.create_dict(self.HotWaterHourlyCOP_tank))
        # space cooling
        m.CoolingHourlyCOP = pyo.Param(m.t, initialize=self.create_dict(self.CoolingHourlyCOP))
        # PV
        m.PhotovoltaicProfile = pyo.Param(m.t, initialize=self.create_dict(self.PhotovoltaicProfile))
        # battery
        # EV
        m.EVDemandProfile = pyo.Param(m.t, initialize=self.create_dict(self.EVDemandProfile))
        m.EVAtHomeProfile = pyo.Param(m.t, initialize=self.create_dict(self.EVAtHomeProfile))
        # energy price
        m.ElectricityPrice = pyo.Param(m.t, initialize=self.create_dict(self.ElectricityPrice))
        m.FiT = pyo.Param(m.t, initialize=self.create_dict(self.FiT))
        # behavior
        m.HotWaterProfile = pyo.Param(m.t, initialize=self.create_dict(self.HotWaterProfile), mutable=True)
        m.BaseLoadProfile = pyo.Param(m.t, initialize=self.create_dict(self.BaseLoadProfile))

    def setup_variables(self, m):
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

    def setup_constraint_space_heating_tank(self, m):
        def tank_energy_heating(m, t):
            if t == 1:
                return m.Q_HeatingTank[t] == \
                       self.CPWater * self.M_WaterTank_heating * (273.15 + self.T_TankStart_heating) - \
                       m.Q_HeatingTank_out[t]
            else:
                return m.Q_HeatingTank[t] == \
                       m.Q_HeatingTank[t - 1] - m.Q_HeatingTank_out[t] + m.Q_HeatingTank_in[t] - \
                       self.U_LossTank_heating * self.A_SurfaceTank_heating * \
                       ((m.Q_HeatingTank[t] / (self.M_WaterTank_heating * self.CPWater)) -
                        (self.T_TankSurrounding_heating + 273.15))
        m.tank_energy_rule_heating = pyo.Constraint(m.t, rule=tank_energy_heating)

    def setup_constraint_space_heating_room(self, m):
        def room_heating(m, t):
            return m.Q_RoomHeating[t] == m.Q_HeatingTank_out[t] + m.Q_HeatingTank_bypass[t] + m.Q_HeatingElement[t]
        m.room_heating_rule = pyo.Constraint(m.t, rule=room_heating)

    def setup_constraint_thermal_mass_temperature(self, m):
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
                       (PHI_st + self.Htr_w * m.T_outside[t] + self.Htr_1 *
                        (((self.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / self.Hve) + T_sup)) / self.Htr_2
            # Equ. C.4
            return m.T_BuildingMass[t] == \
                   (Tm_start * ((self.Cm / 3600) - 0.5 * (self.Htr_3 + self.Htr_em)) + PHI_mtot) / \
                   ((self.Cm / 3600) + 0.5 * (self.Htr_3 + self.Htr_em))
        m.thermal_mass_temperature_rule = pyo.Constraint(m.t, rule=thermal_mass_temperature_rc)

    def setup_constraint_room_temperature(self, m):

        def room_temperature_rc(m, t):
            if t == 1:
                Tm_start = self.BuildingMassTemperatureStartValue
            else:
                Tm_start = m.T_BuildingMass[t - 1]
            # Equ. C.3
            PHI_st = (1 - self.Am / self.Atot - self.Htr_w / 9.1 / self.Atot) * (
                    0.5 * self.Qi + m.Q_Solar[t]
            )
            # Equ. C.9
            T_m = (m.T_BuildingMass[t] + Tm_start) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (
                          self.Htr_ms * T_m
                          + PHI_st
                          + self.Htr_w * m.T_outside[t]
                          + self.Htr_1
                          * (
                                  T_sup
                                  + (self.PHI_ia + m.Q_RoomHeating[t] - m.Q_RoomCooling[t]) / self.Hve
                          )
                  ) / (self.Htr_ms + self.Htr_w + self.Htr_1)
            # Equ. C.11
            T_air = (
                            self.Htr_is * T_s
                            + self.Hve * T_sup
                            + self.PHI_ia
                            + m.Q_RoomHeating[t]
                            - m.Q_RoomCooling[t]
                    ) / (self.Htr_is + self.Hve)
            return m.T_Room[t] == T_air

        m.room_temperature_rule = pyo.Constraint(m.t, rule=room_temperature_rc)

    def setup_constraint_hot_water(self, m):

        def tank_energy_DHW(m, t):
            if t == 1:
                return (
                        m.Q_DHWTank[t]
                        == self.CPWater
                        * self.M_WaterTank_DHW
                        * (273.15 + self.T_TankStart_DHW)
                        - m.Q_DHWTank_out[t]
                )
            else:
                return m.Q_DHWTank[t] == m.Q_DHWTank[t - 1] - m.Q_DHWTank_out[
                    t
                ] + m.Q_DHWTank_in[t] - self.U_LossTank_DHW * self.A_SurfaceTank_DHW * (
                               m.Q_DHWTank[t] / (self.M_WaterTank_DHW * self.CPWater)
                               - (self.T_TankSurrounding_DHW + 273.15)
                       )

        m.tank_energy_rule_DHW = pyo.Constraint(m.t, rule=tank_energy_DHW)

        # (14) DHW profile coverage
        def calc_hot_water_profile(m, t):
            return m.HotWaterProfile[t] == m.Q_DHWTank_out[t] + m.Q_DHWTank_bypass[t]

        m.SupplyOfDHW_rule = pyo.Constraint(m.t, rule=calc_hot_water_profile)

    def setup_constraint_heat_pump(self, m):
        def calc_supply_of_space_heating(m, t):
            return (
                    m.Q_HeatingTank_bypass[t] / m.SpaceHeatingHourlyCOP[t]
                    + m.Q_HeatingTank_in[t] / m.SpaceHeatingHourlyCOP_tank[t]
                    == m.E_Heating_HP_out[t]
            )

        m.calc_use_of_HP_power_DHW_rule = pyo.Constraint(
            m.t, rule=calc_supply_of_space_heating
        )

        def calc_supply_of_DHW(m, t):
            return (
                    m.Q_DHWTank_bypass[t] / m.HotWaterHourlyCOP[t]
                    + m.Q_DHWTank_in[t] / m.HotWaterHourlyCOP_tank[t]
                    == m.E_DHW_HP_out[t]
            )

        m.bypass_DHW_rule = pyo.Constraint(m.t, rule=calc_supply_of_DHW)

        def constrain_heating_max_power(m, t):
            return (
                    m.E_DHW_HP_out[t] + m.E_Heating_HP_out[t]
                    <= self.SpaceHeating_HeatPumpMaximalElectricPower
            )

        m.max_HP_power_rule = pyo.Constraint(m.t, rule=constrain_heating_max_power)

    def setup_constraint_space_cooling(self, m):
        def calc_E_RoomCooling_with_cooling(m, t):
            return m.E_RoomCooling[t] == m.Q_RoomCooling[t] / self.CoolingCOP
        m.E_RoomCooling_with_cooling_rule = pyo.Constraint(m.t, rule=calc_E_RoomCooling_with_cooling)

    def setup_constraint_pv(self, m):

        def calc_UseOfPV(m, t):
            return m.PhotovoltaicProfile[t] == \
                   m.PV2EV[t] * m.EVAtHomeProfile[t] + m.PV2Load[t] + m.PV2Bat[t] + m.PV2Grid[t]
        m.UseOfPV_rule = pyo.Constraint(m.t, rule=calc_UseOfPV)

        def calc_SumOfFeedin(m, t):
            return m.Feed2Grid[t] == m.PV2Grid[t]
        m.SumOfFeedin_rule = pyo.Constraint(m.t, rule=calc_SumOfFeedin)

    def setup_constraint_battery(self, m):

        def calc_BatCharge(m, t):
            return m.BatCharge[t] == m.PV2Bat[t] + m.Grid2Bat[t] + m.EV2Bat[t] * m.EVAtHomeProfile[t] * self.EVOptionV2B
        m.BatCharge_rule = pyo.Constraint(m.t, rule=calc_BatCharge)

        def calc_BatDischarge(m, t):
            return m.BatDischarge[t] == m.Bat2Load[t] + m.Bat2EV[t] * m.EVAtHomeProfile[t]
        m.BatDischarge_rule = pyo.Constraint(m.t, rule=calc_BatDischarge)

        def calc_BatSoC(m, t):
            if t == 1:
                return m.BatSoC[t] == self.scenario.battery.capacity  # start of simulation, battery is empty
            else:
                return m.BatSoC[t] == m.BatSoC[t - 1] + m.BatCharge[t] * self.ChargeEfficiency - \
                       m.BatDischarge[t] * (1 + (1 - self.DischargeEfficiency))
        m.BatSoC_rule = pyo.Constraint(m.t, rule=calc_BatSoC)

    def setup_constraint_ev(self, m):

        def calc_EVCharge(m, t):
            return m.EVCharge[t] == \
                   m.PV2EV[t] * m.EVAtHomeProfile[t] + \
                   m.Grid2EV[t] * m.EVAtHomeProfile[t] + m.Bat2EV[t] * m.EVAtHomeProfile[t]
        m.EVCharge_rule = pyo.Constraint(m.t, rule=calc_EVCharge)

        def calc_EVDischarge(m, t):
            if m.t[t] == 1:
                return m.EVDischarge[t] == 0  # start of simulation, battery is empty
            elif m.t[t] == m.t[-1]:
                return m.EVDischarge[t] == 0  # at the end of simulation Battery will be empty, so no discharge
            else:
                return m.EVDischarge[t] == \
                       m.EVDemandProfile[t] + \
                       (m.EV2Load[t] + m.EV2Bat[t]) * m.EVAtHomeProfile[t] * self.EVOptionV2B
        m.EVDischarge_rule = pyo.Constraint(m.t, rule=calc_EVDischarge)

        def calc_EVSoC(m, t):
            if t == 1:
                return m.EVSoC[t] == 0  # start of simulation, EV is empty
            else:
                return m.EVSoC[t] == \
                       m.EVSoC[t - 1] + m.EVCharge[t] * self.EVChargeEfficiency * m.EVAtHomeProfile[t] - \
                       m.EVDischarge[t] * (1 + (1 - self.EVDischargeEfficiency))
        m.EVSoC_rule = pyo.Constraint(m.t, rule=calc_EVSoC)

    def setup_constraint_electricity_demand(self, m):

        def calc_SumOfLoads_with_cooling(m, t):
            return m.Load[t] == \
                   m.BaseLoadProfile[t] + m.E_Heating_HP_out[t] + m.Q_HeatingElement[t] + \
                   m.Q_RoomCooling[t] / self.CoolingCOP + m.E_DHW_HP_out[t]
        m.SumOfLoads_with_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_with_cooling)

        def calc_SumOfLoads_without_cooling(m, t):
            return m.Load[t] == m.BaseLoadProfile[t] + m.E_Heating_HP_out[t] + m.Q_HeatingElement[t] + m.E_DHW_HP_out[t]
        m.SumOfLoads_without_cooling_rule = pyo.Constraint(m.t, rule=calc_SumOfLoads_without_cooling)

    def setup_constraint_electricity_supply(self, m):

        def calc_UseOfGrid(m, t):
            return m.Grid[t] == m.Grid2Load[t] + m.Grid2Bat[t] + m.Grid2EV[t] * m.EVAtHomeProfile[t]
        m.UseOfGrid_rule = pyo.Constraint(m.t, rule=calc_UseOfGrid)

        def calc_SupplyOfLoads(m, t):
            return m.Load[t] == \
                   m.Grid2Load[t] + m.PV2Load[t] + m.Bat2Load[t] + \
                   m.EV2Load[t] * m.EVAtHomeProfile[t] * self.EVOptionV2B
        m.SupplyOfLoads_rule = pyo.Constraint(m.t, rule=calc_SupplyOfLoads)

    def setup_objective(self, m):
        def minimize_cost(m):
            rule = sum(
                m.Grid[t] * m.ElectricityPrice[t] - m.Feed2Grid[t] * m.FiT[t]
                for t in m.t
            )
            return rule

        m.total_operation_cost_rule = pyo.Objective(
            rule=minimize_cost, sense=pyo.minimize
        )

    @staticmethod
    def create_dict(values) -> dict:
        dictionary = {}
        for index, value in enumerate(values, start=1):
            dictionary[index] = value
        return dictionary

    def config_instance(self, instance):
        self.config_grid(instance)
        self.config_space_heating(instance)
        self.config_hot_water_tank(instance)
        self.config_battery(instance)
        self.config_space_cooling_technology(instance)
        self.config_vehicle(instance)
        self.config_pv(instance)
        return instance

    def config_grid(self, instance):
        for t in range(1, 8761):
            instance.Grid[t].setub(self.scenario.building.grid_power_max)
            instance.Feed2Grid[t].setub(self.scenario.building.grid_power_max)

    def config_space_heating(self, instance):
        for t in range(1, 8761):
            instance.Q_HeatingElement[t].setub(self.scenario.boiler.heating_element_power)
            instance.T_Room[t].setlb(self.scenario.behavior.target_temperature_array_min[t - 1])
            instance.T_BuildingMass[t].setub(100)
            instance.Q_HeatingTank[t].setlb(self.Q_TankEnergyMin_heating)
            instance.Q_HeatingTank[t].setub(self.Q_TankEnergyMax_heating)
            instance.E_Heating_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

    def config_hot_water_tank(self, instance):
        for t in range(1, 8761):
            instance.Q_DHWTank[t].setlb(self.Q_TankEnergyMin_DHW)
            instance.Q_DHWTank[t].setub(self.Q_TankEnergyMax_DHW)
            instance.E_DHW_HP_out[t].setub(self.SpaceHeating_HeatPumpMaximalElectricPower)

    def config_battery(self, instance):
        for t in range(1, 8761):
            instance.Grid2Bat[t].setub(self.scenario.battery.charge_power_max)
            instance.Bat2Load[t].setub(self.scenario.battery.discharge_power_max)
            instance.BatSoC[t].setub(self.scenario.battery.capacity)
            instance.BatCharge[t].setub(self.scenario.battery.charge_power_max)
            instance.BatDischarge[t].setub(self.scenario.battery.discharge_power_max)

    def config_space_cooling_technology(self, instance):
        if self.scenario.space_cooling_technology.power == 0:
            no_cooling_target_temperature_max = (
                self.generate_maximum_target_indoor_temperature_no_cooling(temperature_max=23)
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
                instance.Q_RoomCooling[t].setub(self.scenario.space_cooling_technology.power)
                instance.T_Room[t].setub(self.scenario.behavior.target_temperature_array_max[t - 1])
            instance.SumOfLoads_without_cooling_rule.deactivate()
            instance.SumOfLoads_with_cooling_rule.activate()
            instance.E_RoomCooling_with_cooling_rule.activate()

    def config_vehicle(self, instance):
        if self.scenario.vehicle.capacity == 0:
            for t in range(1, 8761):
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
            max_discharge_EV = self.create_upper_bound_EV_discharge()
            for t in range(1, 8761):
                if self.EVOptionV2B == 0:
                    instance.EV2Bat[t].fix(0)
                    instance.EV2Load[t].fix(0)
                instance.Grid2EV[t].setub(self.scenario.vehicle.charge_power_max)
                instance.EV2Load[t].setub(self.scenario.vehicle.discharge_power_max)
                instance.PV2EV[t].setub(self.scenario.vehicle.charge_power_max)
                instance.EVSoC[t].setub(self.scenario.vehicle.capacity)
                instance.EVCharge[t].setub(self.scenario.vehicle.charge_power_max)
                instance.EVDischarge[t].setub(max_discharge_EV[t - 1])
            instance.EVCharge_rule.activate()
            instance.EVDischarge_rule.activate()
            instance.EVSoC_rule.activate()

    def config_pv(self, instance):
        if self.scenario.pv.size == 0:
            for t in range(1, 8761):
                instance.PV2Load[t].fix(0)
                instance.PV2Bat[t].fix(0)
                instance.PV2Grid[t].fix(0)
                instance.PV2EV[t].fix(0)
            instance.UseOfPV_rule.deactivate()
        else:
            for t in range(1, 8761):
                instance.PV2Load[t].setub(self.scenario.building.grid_power_max)
                instance.PV2Bat[t].setub(self.scenario.building.grid_power_max)
                instance.PV2EV[t].setub(self.scenario.building.grid_power_max)
            instance.UseOfPV_rule.activate()

    @performance_counter
    def setup_instance(self):
        model = self.setup_model()
        instance = model.create_instance()
        instance = self.config_instance(instance)
        return instance

    @performance_counter
    def solve_instance(self, instance):
        pyo.SolverFactory("gurobi").solve(instance, tee=False)
        return instance

    def run_fuel_boiler_opt(self):
        (
            outside_temperature,
            q_solar,
            hot_water_demand,
        ) = self.fuel_boiler_save_scenario()
        (
            heating_cost,
            cooling_cost,
            heating_demand,
            cooling_demand,
            room_temperature,
            building_mass_temperature,
        ) = self.fuel_boiler_heating_cooling()
        hot_water_cost = self.fuel_boiler_hot_water()
        self.fuel_boiler_remove_heating_cooling_hot_water_demand()
        solved_instance = self.run_heatpump_opt()
        solved_instance.__dict__["T_outside"].store_values(self.create_dict(outside_temperature))
        solved_instance.__dict__["Q_Solar"].store_values(self.create_dict(q_solar))
        solved_instance.__dict__["Q_RoomHeating"].set_values(self.create_dict(heating_demand))
        solved_instance.__dict__["Q_RoomCooling"].set_values(self.create_dict(cooling_demand))
        solved_instance.__dict__["E_RoomCooling"].set_values(self.create_dict(cooling_demand / self.CoolingCOP))
        solved_instance.__dict__["T_Room"].set_values(self.create_dict(room_temperature))
        solved_instance.__dict__["T_BuildingMass"].set_values(self.create_dict(building_mass_temperature))
        solved_instance.__dict__["HotWaterProfile"].store_values(self.create_dict(hot_water_demand))
        solved_instance.__dict__["total_operation_cost_rule"] += (heating_cost + cooling_cost + hot_water_cost)
        return solved_instance

    def run_heatpump_opt(self):
        instance = self.setup_instance()
        solved_instance = self.solve_instance(instance)
        return solved_instance

    def run(self):
        if self.scenario.boiler.type in ["Air_HP", "Ground_HP"]:
            solved_instance = self.run_heatpump_opt()
        else:
            solved_instance = self.run_fuel_boiler_opt()
        logger.info(f"OptCost: {round(solved_instance.total_operation_cost_rule(), 2)}")
        return solved_instance
