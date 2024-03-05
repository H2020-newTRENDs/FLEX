import numpy as np
import copy
import logging
from models.operation.model_base import OperationModel


class RefOperationModel(OperationModel):

    def solve(self):
        logger = logging.getLogger(f"{self.scenario.config.project_name}")
        logger.info(f"starting solving Ref model.")
        if self.scenario.boiler.type in ["Air_HP", "Ground_HP", "Electric"]:
            model_ref = self.run_heatpump_ref()
        else:
            model_ref = self.run_fuel_boiler_ref()
        logger.info(f"RefCost: {round(self.TotalCost.sum(), 2)}")
        return model_ref

    """
    Heat Pump REF
    """
    def run_heatpump_ref(self):
        """
        Assumption for the Reference scenario: the produced PV power is always used for the immediate electric demand,
        if there is a surplus of PV power, it will be used to charge the Battery,
        if the Battery is full or not available, the PV power will be used to heat the HotWaterTank,
        if the HotWaterTank is not available or full, the PV power will sold to the Grid.
        The surplus of PV energy is never used to Preheat or Precool the building.
        """
        self.calc_space_heating_demand()
        self.calc_space_cooling_demand()
        self.calc_hot_water_demand()
        grid_demand, pv_surplus = self.calc_load()
        grid_demand, pv_surplus = self.calc_battery_energy(grid_demand, pv_surplus)
        grid_demand, pv_surplus = self.calculate_ev_energy(grid_demand, pv_surplus)
        grid_demand, pv_surplus = self.calc_hot_water_tank_energy(grid_demand, pv_surplus)
        self.calc_grid(grid_demand, pv_surplus)
        self.set_boiler_parameters_to_zero()
        return self

    def calc_space_heating_demand(self):
        hp_max = (self.SpaceHeating_MaxBoilerPower * self.SpaceHeatingHourlyCOP)
        self.Q_HeatingElement_heat = np.where(self.Q_RoomHeating - hp_max < 0, 0, self.Q_RoomHeating - hp_max)
        self.Q_HeatingTank_bypass = self.Q_RoomHeating - self.Q_HeatingElement_heat
        self.E_Heating_HP_out = self.Q_HeatingTank_bypass / self.SpaceHeatingHourlyCOP
        self.Q_HeatingTank = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_in = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_out = np.zeros(self.Q_HeatingTank_bypass.shape)

    def calc_space_cooling_demand(self):
        self.E_RoomCooling = self.Q_RoomCooling / self.CoolingCOP

    def calc_hot_water_demand(self):
        self.Q_DHWTank_bypass = copy.deepcopy(self.HotWaterProfile)
        self.E_DHW_HP_out = self.Q_DHWTank_bypass / self.HotWaterHourlyCOP

    def calc_load(self):
        self.Load = (
            self.BaseLoadProfile
            + self.E_Heating_HP_out
            + self.Q_HeatingElement_heat
            + self.E_RoomCooling
            + self.E_DHW_HP_out
        )
        grid_demand = np.where(
            self.Load - self.PhotovoltaicProfile < 0,
            0,
            self.Load - self.PhotovoltaicProfile,
        )
        pv_surplus = np.where(
            self.PhotovoltaicProfile - self.Load < 0,
            0,
            self.PhotovoltaicProfile - self.Load,
        )
        self.PV2Load = self.PhotovoltaicProfile - pv_surplus
        return grid_demand, pv_surplus

    def check_hp_max_power(self, HP_power):
        """returns True if maximum power is exceeded"""
        # check if the maximal capacity of the heat pump is exceeded by charging the storage:
        return HP_power > self.SpaceHeating_MaxBoilerPower

    def calc_battery_energy(self, grid_demand: np.array, pv_surplus: np.array):

        self.BatSoC = np.zeros(pv_surplus.shape)
        self.Bat2Load = np.zeros(pv_surplus.shape)
        self.PV2Bat = np.zeros(pv_surplus.shape)
        self.Grid2Bat = np.zeros(pv_surplus.shape)
        self.BatCharge = self.PV2Bat
        self.BatDischarge = self.Bat2Load

        if self.scenario.battery.capacity > 0:

            # setup parameters
            capacity = self.scenario.battery.capacity  # kWh
            max_charge_power = self.scenario.battery.charge_power_max  # kW
            max_discharge_power = self.scenario.battery.discharge_power_max  # kW
            charge_efficiency = self.scenario.battery.charge_efficiency
            discharge_efficiency = self.scenario.battery.discharge_efficiency

            # return values of this function
            grid_demand_after_battery = np.copy(grid_demand)
            pv_surplus_after_battery = np.copy(pv_surplus)

            for i in range(0, len(pv_surplus)):

                if i == 0:
                    bat_soc_start = self.scenario.battery.capacity
                else:
                    bat_soc_start = self.BatSoC[i - 1]

                if pv_surplus[i] > 0:
                    if bat_soc_start < capacity:
                        bat_gap = capacity - bat_soc_start
                        if pv_surplus[i] >= bat_gap:
                            charge_amount = min(bat_gap, max_charge_power)
                        else:
                            charge_amount = min(pv_surplus[i], max_charge_power)
                        self.BatSoC[i] = (bat_soc_start + charge_amount * charge_efficiency)
                        self.PV2Bat[i] = charge_amount
                        pv_surplus_after_battery[i] -= charge_amount
                    else:
                        self.BatSoC[i] = capacity
                        self.PV2Bat[i] = 0
                else:
                    if bat_soc_start > 0:
                        if bat_soc_start * discharge_efficiency <= grid_demand[i]:
                            discharge_amount = min(bat_soc_start, max_discharge_power)
                        else:
                            discharge_amount = min(grid_demand[i], max_discharge_power)
                        grid_demand_after_battery[i] = grid_demand[i] - discharge_amount * discharge_efficiency
                        self.BatSoC[i] = bat_soc_start - discharge_amount
                        self.Bat2Load[i] = discharge_amount * discharge_efficiency
                    else:
                        grid_demand_after_battery[i] = grid_demand[i]
                        self.BatSoC[i] = 0
                        self.Bat2Load[i] = 0

        else:
            grid_demand_after_battery = grid_demand
            pv_surplus_after_battery = pv_surplus

        return grid_demand_after_battery, pv_surplus_after_battery

    def calculate_ev_energy(self, grid_demand, pv_surplus):

        self.EVDemandProfile = np.zeros(pv_surplus.shape)
        self.EVAtHomeProfile = np.zeros(pv_surplus.shape)

        self.EVSoC = np.zeros(pv_surplus.shape)
        self.EVCharge = np.zeros(pv_surplus.shape)
        self.EVDischarge = np.zeros(pv_surplus.shape)
        self.EV2Bat = np.zeros(pv_surplus.shape)
        self.EV2Load = np.zeros(pv_surplus.shape)
        self.Grid2EV = np.zeros(pv_surplus.shape)
        self.PV2EV = np.zeros(pv_surplus.shape)
        self.Bat2EV = np.zeros(pv_surplus.shape)

        if self.scenario.vehicle.capacity > 0:
            self.EVAtHomeProfile = np.array(self.scenario.behavior.vehicle_at_home, dtype=int)

            capacity = self.scenario.vehicle.capacity  # Wh
            max_charge_power = self.scenario.vehicle.charge_power_max  # W
            charge_efficiency = self.scenario.vehicle.charge_efficiency  # %
            discharge_efficiency = self.scenario.vehicle.discharge_efficiency  # %
            self.EVDemandProfile = self.scenario.behavior.vehicle_demand
            self.EVDischarge = self.EVDemandProfile / discharge_efficiency

            grid_demand_after_ev = np.copy(grid_demand)
            pv_surplus_after_ev = np.copy(pv_surplus)

            for i in range(0, len(pv_surplus)):

                if i == 0:
                    ev_soc_start = self.scenario.vehicle.capacity
                else:
                    ev_soc_start = self.EVSoC[i - 1]

                if self.EVAtHomeProfile[i] == 1 and ev_soc_start <= capacity:

                    charge_necessary = capacity - ev_soc_start
                    if charge_necessary <= max_charge_power:
                        charge_amount = charge_necessary
                    else:
                        charge_amount = max_charge_power

                    if pv_surplus[i] > 0:
                        if pv_surplus[i] * charge_efficiency <= charge_amount:
                            pv_surplus_after_ev[i] -= pv_surplus[i]
                            grid_demand_after_ev[i] += (charge_amount / charge_efficiency - pv_surplus[i])
                            self.Grid2EV[i] = charge_amount - pv_surplus[i] * charge_efficiency
                            self.PV2EV[i] = pv_surplus[i] * charge_efficiency
                        else:
                            pv_surplus_after_ev[i] -= charge_amount / charge_efficiency
                            self.PV2EV[i] = charge_amount / charge_efficiency
                    else:
                        grid_demand_after_ev[i] += charge_amount / charge_efficiency
                        self.Grid2EV[i] = charge_amount

                    self.EVCharge[i] = charge_amount
                    self.EVSoC[i] = ev_soc_start + charge_amount - self.EVDischarge[i]

                else:
                    self.EVSoC[i] = ev_soc_start - self.EVDischarge[i]

        else:
            grid_demand_after_ev = grid_demand
            pv_surplus_after_ev = pv_surplus

        return grid_demand_after_ev, pv_surplus_after_ev

    def calc_hot_water_tank_energy(self, grid_demand: np.array, pv_surplus: np.array):
        """
        calculates the usage and the energy in the domestic hot water tank.
        The tank is charged by the heat pump with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized, the energy for DHW will always be solemnly provided by the DHW tank.
        Therefore, the heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.

        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """
        logger = logging.getLogger(f"{self.scenario.config.project_name}")
        self.Q_DHWTank = np.ones(pv_surplus.shape) * self.scenario.hot_water_tank.temperature_min
        self.Q_DHWTank_out = np.zeros(pv_surplus.shape)
        self.Q_DHWTank_in = np.zeros(pv_surplus.shape)
        self.Q_HeatingElement_DHW = np.zeros(pv_surplus.shape)

        if self.scenario.hot_water_tank.size > 0:

            # setup parameters
            hot_water_demand = self.HotWaterProfile
            temperature_min = self.scenario.hot_water_tank.temperature_min
            temperature_max = self.scenario.hot_water_tank.temperature_max
            size = self.scenario.hot_water_tank.size
            surface_area = self.A_SurfaceTank_DHW
            loss = self.U_LossTank_DHW
            surrounding_temperature = self.T_TankSurrounding_DHW
            cop = self.HotWaterHourlyCOP
            cop_tank = self.HotWaterHourlyCOP_tank
            tank_capacity = size * self.CPWater

            # return values of this function
            grid_demand_after_hot_water_tank = np.copy(grid_demand)
            pv_surplus_after_hot_water_tank = np.copy(pv_surplus)
            tank_temperature = np.zeros(pv_surplus.shape)
            tank_loss = np.zeros(pv_surplus.shape)

            for i in range(0, len(pv_surplus)):

                if i == 0:
                    temp_start = self.T_TankStart_DHW
                    tank_temperature[i] = temp_start
                else:
                    temp_start = tank_temperature[i - 1]
                tank_loss[i] = ((temp_start - surrounding_temperature) * surface_area * loss)  # W

                if pv_surplus[i] > 0:
                    # check if the temperature in the tank is lower than minimum temperature
                    if temp_start < temperature_min:
                        # charge the tank at least to minimum required temperature
                        tank_in_necessary = (
                            temperature_min - temp_start
                        ) * tank_capacity
                        tank_in_space = (temperature_max - temp_start) * tank_capacity
                        # if there is pv surplus is not enough to charge the minimum required
                        if pv_surplus[i] * cop_tank[i] < tank_in_necessary:
                            q_tank_in = tank_in_necessary
                            pv_surplus_after_hot_water_tank[i] -= pv_surplus[i]
                            grid_demand_after_hot_water_tank[i] += (
                                tank_in_necessary / cop_tank[i] - pv_surplus[i]
                            )
                        elif ( # if pv surplus is large enough to meet minimum requirement but not too large for tank:
                            tank_in_necessary < pv_surplus[i] * cop_tank[i] < tank_in_space
                        ):
                            q_tank_in = pv_surplus[i] * cop_tank[i]
                            pv_surplus_after_hot_water_tank[i] -= pv_surplus[i]
                        else:
                            q_tank_in = tank_in_space
                            pv_surplus_after_hot_water_tank[i] -= (
                                q_tank_in / cop_tank[i]
                            )
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        tank_temperature[i] = (
                            tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity
                        )

                    elif temperature_min <= temp_start < temperature_max:
                        tank_in_space = (temperature_max - temp_start) * tank_capacity
                        if pv_surplus[i] * cop_tank[i] > tank_in_space:
                            q_tank_in = tank_in_space
                        else:
                            q_tank_in = pv_surplus[i] * cop_tank[i]
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        pv_surplus_after_hot_water_tank[i] -= q_tank_in / cop_tank[i]
                        tank_temperature[i] = (
                            tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity
                        )

                    else:
                        q_tank_in = 0
                        self.Q_DHWTank_in[i] = q_tank_in
                        tank_temperature[i] = (
                            tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity
                        )

                else:

                    if temp_start < temperature_min:
                        tank_in_necessary = (
                            temperature_min - temp_start
                        ) * tank_capacity
                        q_tank_in = tank_in_necessary
                        grid_demand_after_hot_water_tank[i] += q_tank_in / cop_tank[i]
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        tank_temperature[i] = (
                            tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity
                        )

                    else:
                        tank_out_limit = (temp_start - temperature_min) * tank_capacity
                        if tank_out_limit < hot_water_demand[i]:
                            q_tank_out = tank_out_limit
                        else:
                            q_tank_out = hot_water_demand[i]
                        self.Q_DHWTank_out[i] = q_tank_out
                        self.Q_DHWTank_bypass[i] -= q_tank_out
                        self.E_DHW_HP_out[i] -= q_tank_out / cop[i]
                        grid_demand_after_hot_water_tank[i] -= q_tank_out / cop[i]
                        if i == 0:
                            tank_temperature[i] = (
                                    tank_temperature[i] - (q_tank_out + tank_loss[i]) / tank_capacity
                            )
                        else:
                            tank_temperature[i] = (
                                tank_temperature[i - 1] - (q_tank_out + tank_loss[i]) / tank_capacity
                            )

                """
                Removed temporally because we set the power of heating element to zero. 
                However, with this code, the efficiency of heating element is still working.
                """
                # # check if the HP max power is exceeded due to additional load:
                # hp_power = self.E_DHW_HP_out[i] + self.E_Heating_HP_out[i]
                # if self.check_hp_max_power(hp_power):
                #     if self.HeatingElement_power == 0:
                #         logger.info(f"In scenario {self.scenario.scenario_id} the HP power is too low to maintain "
                #                     f"indoor comfort level.")
                #     else:
                #         # if max power is exceeded check if reducing the DHW power will solve the problem:
                #         if not self.check_hp_max_power(hp_power - self.E_DHW_HP_out[i]):
                #             # the max HP power can be achieved by using the heating element for DHW
                #             # hp DHW power is reduced by that amount:
                #             exceeded_power = hp_power - self.SpaceHeating_MaxBoilerPower
                #             self.E_DHW_HP_out[i] -= exceeded_power
                #             # Heating element is used instead:
                #             self.Q_HeatingElement_DHW[i] += exceeded_power * self.HeatingElement_efficiency
                #             # grid load is decreased by HP power and increased by heating element power:
                #             grid_demand_after_hot_water_tank[i] -= exceeded_power / cop_tank[i]
                #             grid_demand_after_hot_water_tank[i] += exceeded_power / self.HeatingElement_efficiency
                #
                #         else:  # the HP power must also be reduced for heating and the indoor temperature can not be held:
                #             logger.info(f"In scenario {self.scenario.scenario_id} the HP power is too low to maintain "
                #                         f"indoor comfort level.")

            self.Q_DHWTank = (tank_temperature + 273.15) * tank_capacity
            self.PV2Load += (pv_surplus - pv_surplus_after_hot_water_tank)

        else:
            grid_demand_after_hot_water_tank = grid_demand
            pv_surplus_after_hot_water_tank = pv_surplus

        self.Q_HeatingElement = self.Q_HeatingElement_heat + self.Q_HeatingElement_DHW
        return grid_demand_after_hot_water_tank, pv_surplus_after_hot_water_tank

    def calc_grid(self, grid_demand: np.array, pv_surplus: np.array):
        self.Grid = grid_demand
        self.Grid2Load = grid_demand
        self.PV2Grid = pv_surplus
        self.Feed2Grid = pv_surplus
        self.TotalCost = self.ElectricityPrice * grid_demand - pv_surplus * self.FiT

    def set_boiler_parameters_to_zero(self):
        self.Fuel = np.zeros(shape=self.Grid.shape)
        self.FuelPrice = np.zeros(shape=self.Grid.shape)
        self.Q_DHW_Boiler_out = np.zeros(shape=self.Grid.shape)
        self.Q_Heating_Boiler_out = np.zeros(shape=self.Grid.shape)

    """
    Fuel Boiler REF
    """
    def run_fuel_boiler_ref(self):
        self.calc_space_heating_demand_fuel_boiler()
        self.calc_space_cooling_demand()
        self.calc_hot_water_demand_fuel_boiler()
        self.calc_fuel_demand()
        grid_demand, pv_surplus = self.calc_load_fuel_boiler()
        grid_demand, pv_surplus = self.calc_battery_energy(grid_demand, pv_surplus)
        grid_demand, pv_surplus = self.calculate_ev_energy(grid_demand, pv_surplus)
        self.Fuel = self.calc_hot_water_tank_energy_fuel_boiler(self.Fuel)
        self.calc_grid_fuel_boiler(grid_demand, pv_surplus)
        self.set_heat_pump_parameters_to_zero()

        return self

    def calc_space_heating_demand_fuel_boiler(self):
        boiler_max = self.SpaceHeating_MaxBoilerPower
        self.Q_HeatingElement_heat = np.where(self.Q_RoomHeating - boiler_max < 0, 0, self.Q_RoomHeating - boiler_max)
        self.Q_HeatingTank_bypass = self.Q_RoomHeating
        self.Q_Heating_Boiler_out = self.Q_RoomHeating - self.Q_HeatingElement_heat
        self.Q_HeatingTank = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_in = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_out = np.zeros(self.Q_HeatingTank_bypass.shape)

    def calc_hot_water_demand_fuel_boiler(self):
        self.Q_DHWTank_bypass = copy.deepcopy(self.HotWaterProfile)
        self.Q_DHW_Boiler_out = self.Q_DHWTank_bypass

    def calc_fuel_demand(self):
        self.Fuel = (self.Q_DHW_Boiler_out + self.Q_Heating_Boiler_out) / self.fuel_boiler_efficiency

    def calc_load_fuel_boiler(self):
        self.Load = (
            self.BaseLoadProfile
            + self.Q_HeatingElement_heat
            + self.E_RoomCooling
        )
        grid_demand = np.where(self.Load - self.PhotovoltaicProfile < 0, 0, self.Load - self.PhotovoltaicProfile)
        pv_surplus = np.where(self.PhotovoltaicProfile - self.Load < 0, 0, self.PhotovoltaicProfile - self.Load)
        self.PV2Load = self.PhotovoltaicProfile - pv_surplus
        return grid_demand, pv_surplus

    def calc_hot_water_tank_energy_fuel_boiler(self, gas_demand: np.array):
        """
        calculates the usage and the energy in the domestic hot water tank.
        Whenever there is surplus of PV electricity, the DHW tank is charged by:
        (1) heat pump with the COP for hot water;
        (2) heating element;
        It is discharged by the DHW-usage.
        IF a DHW tank is utilized, the energy for DHW will always be solemnly provided by the DHW tank.
        Therefore, the charge into the DHW tank must be in accordance with demand + losses.
        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """
        self.Q_DHWTank = np.ones(gas_demand.shape) * self.scenario.hot_water_tank.temperature_min
        self.Q_DHWTank_out = np.zeros(gas_demand.shape)
        self.Q_DHWTank_in = np.zeros(gas_demand.shape)
        self.Q_HeatingElement_DHW = np.zeros(gas_demand.shape)

        if self.scenario.hot_water_tank.size > 0:

            # setup parameters
            hot_water_demand = self.HotWaterProfile
            temperature_min = self.scenario.hot_water_tank.temperature_min
            size = self.scenario.hot_water_tank.size
            surface_area = self.A_SurfaceTank_DHW
            loss = self.U_LossTank_DHW
            surrounding_temperature = self.T_TankSurrounding_DHW
            tank_capacity = size * self.CPWater

            # return values of this function
            gas_demand_after_hot_water_tank = np.copy(gas_demand)
            tank_temperature = np.zeros(gas_demand.shape)
            tank_loss = np.zeros(gas_demand.shape)

            for i in range(0, len(gas_demand)):

                if i == 0:
                    temp_start = self.T_TankStart_DHW
                    tank_temperature[i] = temp_start
                else:
                    temp_start = tank_temperature[i - 1]
                tank_loss[i] = ((temp_start - surrounding_temperature) * surface_area * loss)  # W

                if temp_start < temperature_min:
                    tank_in_necessary = (temperature_min - temp_start) * tank_capacity
                    q_tank_in = tank_in_necessary
                    gas_demand_after_hot_water_tank[i] += q_tank_in / self.fuel_boiler_efficiency
                    self.Q_DHWTank_in[i] = q_tank_in
                    self.Q_DHW_Boiler_out[i] += q_tank_in
                    tank_temperature[i] = (
                        tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity
                    )

                else:  # temperature is above minimum temperature in tank
                    tank_out_limit = (temp_start - temperature_min) * tank_capacity
                    if tank_out_limit < hot_water_demand[i]:
                        q_tank_out = tank_out_limit
                    else:
                        q_tank_out = hot_water_demand[i]
                    self.Q_DHWTank_out[i] = q_tank_out
                    self.Q_DHWTank_bypass[i] -= q_tank_out
                    self.Q_DHW_Boiler_out[i] -= q_tank_out
                    gas_demand_after_hot_water_tank[i] -= q_tank_out / self.fuel_boiler_efficiency
                    if i == 0:
                        tank_temperature[i] = (tank_temperature[i] - (q_tank_out + tank_loss[i]) / tank_capacity)
                    else:
                        tank_temperature[i] = (tank_temperature[i - 1] - (q_tank_out + tank_loss[i]) / tank_capacity)

            self.Q_DHWTank = (tank_temperature + 273.15) * tank_capacity
            self.Q_DHWTank_bypass = self.Q_DHWTank_bypass.clip(min=0)  # TODO: for some unknown reason, for some scenarios, this line is necessary

        else:
            gas_demand_after_hot_water_tank = gas_demand

        self.Q_HeatingElement = self.Q_HeatingElement_heat + self.Q_HeatingElement_DHW
        return gas_demand_after_hot_water_tank

    def calc_grid_fuel_boiler(self, grid_demand: np.array, pv_surplus: np.array):
        self.Grid = grid_demand
        self.Grid2Load = grid_demand
        self.PV2Grid = pv_surplus
        self.Feed2Grid = pv_surplus
        self.FuelPrice = self.scenario.energy_price.__dict__[self.scenario.boiler.type]
        self.TotalCost = self.ElectricityPrice * grid_demand - pv_surplus * self.FiT + \
                         self.Fuel * self.FuelPrice

    def set_heat_pump_parameters_to_zero(self):
        self.E_Heating_HP_out = np.zeros(shape=self.Grid.shape)
        self.E_DHW_HP_out = np.zeros(shape=self.Grid.shape)



