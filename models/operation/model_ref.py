import numpy as np
import copy
from models.operation.model_base import OperationModel
from basics.kit import get_logger

logger = get_logger(__name__)


class RefOperationModel(OperationModel):
    def calc_space_heating_demand(self):
        (
            self.Q_RoomHeating,
            self.Q_RoomCooling,
            self.T_Room,
            self.T_BuildingMass,
        ) = self.calculate_heating_and_cooling_demand(
            thermal_start_temperature=self.BuildingMassTemperatureStartValue,
            static=False,
        )
        hp_max = (
            self.SpaceHeating_HeatPumpMaximalElectricPower * self.SpaceHeatingHourlyCOP
        )

        self.Q_HeatingElement = np.where(
            self.Q_RoomHeating - hp_max < 0, 0, self.Q_RoomHeating - hp_max
        )
        self.Q_HeatingTank_bypass = self.Q_RoomHeating - self.Q_HeatingElement
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
            + self.Q_HeatingElement
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
                        if pv_surplus[i] <= max_charge_power:
                            charge_amount = pv_surplus[i]
                        else:
                            charge_amount = max_charge_power
                        pv_surplus_after_battery[i] -= charge_amount
                        self.BatSoC[i] = (
                            bat_soc_start + charge_amount * charge_efficiency
                        )
                        self.PV2Bat[i] = charge_amount * charge_efficiency
                        if self.BatSoC[i] > capacity:
                            right_charge_amount = (
                                capacity - bat_soc_start
                            ) / charge_efficiency
                            pv_surplus_after_battery[i] += (
                                charge_amount - right_charge_amount
                            )
                            self.BatSoC[i] = capacity
                            self.PV2Bat[i] = capacity - bat_soc_start
                        else:
                            pass
                    else:
                        self.BatSoC[i] = capacity
                        self.PV2Bat[i] = 0
                else:
                    if bat_soc_start > 0:
                        if bat_soc_start < max_discharge_power:
                            discharge_amount = bat_soc_start
                        else:
                            discharge_amount = max_discharge_power
                        grid_demand_after_battery[i] = (
                            grid_demand[i] - discharge_amount * discharge_efficiency
                        )
                        self.BatSoC[i] = bat_soc_start - discharge_amount
                        self.Bat2Load[i] = discharge_amount * discharge_efficiency
                        if grid_demand_after_battery[i] < 0:
                            discharge_amount = grid_demand[i] / discharge_efficiency
                            grid_demand_after_battery[i] = 0
                            self.BatSoC[i] = bat_soc_start - discharge_amount
                            self.Bat2Load[i] = grid_demand[i]
                        else:
                            pass
                    else:
                        grid_demand_after_battery[i] = grid_demand[i]
                        self.BatSoC[i] = 0
                        self.Bat2Load[i] = 0

        else:
            grid_demand_after_battery = grid_demand
            pv_surplus_after_battery = pv_surplus

        return grid_demand_after_battery, pv_surplus_after_battery

    def calculate_ev_energy(self, grid_demand, pv_surplus):

        self.EVDemandProfile = self.scenario.behavior.vehicle_demand
        self.EVAtHomeProfile = np.array(self.scenario.behavior.vehicle_at_home, dtype=int)

        self.EVSoC = np.zeros(pv_surplus.shape)
        self.EVCharge = np.zeros(pv_surplus.shape)
        self.EVDischarge = self.EVDemandProfile
        self.EV2Bat = np.zeros(pv_surplus.shape)
        self.EV2Load = np.zeros(pv_surplus.shape)
        self.Grid2EV = np.zeros(pv_surplus.shape)
        self.PV2EV = np.zeros(pv_surplus.shape)
        self.Bat2EV = np.zeros(pv_surplus.shape)

        if self.scenario.vehicle.capacity > 0:

            capacity = self.scenario.vehicle.capacity  # kWh
            max_charge_power = self.scenario.vehicle.charge_power_max  # kW
            charge_efficiency = self.scenario.vehicle.charge_efficiency

            grid_demand_after_ev = np.copy(grid_demand)
            pv_surplus_after_ev = np.copy(pv_surplus)

            for i in range(0, len(pv_surplus)):

                if i == 0:
                    ev_soc_start = self.scenario.vehicle.capacity
                else:
                    ev_soc_start = self.EVSoC[i - 1]

                if self.EVAtHomeProfile[i] == 1:

                    if ev_soc_start <= capacity:

                        charge_necessary = capacity - ev_soc_start
                        if charge_necessary <= max_charge_power:
                            charge_amount = charge_necessary
                        else:
                            charge_amount = max_charge_power

                        if pv_surplus[i] > 0:
                            if pv_surplus[i] * charge_efficiency <= charge_amount:
                                pv_surplus_after_ev[i] -= pv_surplus[i]
                                grid_demand_after_ev[i] += (
                                    charge_amount / charge_efficiency - pv_surplus[i]
                                )
                                self.Grid2EV[i] = (
                                    charge_amount - pv_surplus[i] * charge_efficiency
                                )
                            else:
                                pv_surplus_after_ev[i] -= (
                                    charge_amount / charge_efficiency
                                )

                        else:
                            grid_demand_after_ev[i] += charge_amount / charge_efficiency
                            self.Grid2EV[i] = charge_amount

                        self.EVCharge[i] = charge_amount
                        self.EVSoC[i] = ev_soc_start + charge_amount

                    else:
                        pass

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
        Therefore the heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.

        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """

        self.Q_DHWTank = np.zeros(pv_surplus.shape)
        self.Q_DHWTank_out = np.zeros(pv_surplus.shape)
        self.Q_DHWTank_in = np.zeros(pv_surplus.shape)

        if self.scenario.hot_water_tank.size > 0:

            # setup parameters
            hot_water_demand = self.HotWaterProfile
            temperature_min = self.scenario.hot_water_tank.temperature_min
            temperature_max = self.scenario.hot_water_tank.temperature_max
            size = self.scenario.hot_water_tank.size
            surface_area = self.scenario.hot_water_tank.surface_area
            loss = self.U_LossTank_DHW
            surrounding_temperature = self.T_TankSurrounding_DHW
            start_temperature = self.T_TankStart_DHW
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
                    temp_start = start_temperature
                else:
                    temp_start = tank_temperature[i - 1]
                tank_loss[i] = (
                    (temp_start - surrounding_temperature) * surface_area * loss
                )  # W

                if pv_surplus[i] > 0:

                    if temp_start < temperature_min:
                        tank_in_necessary = (
                            temperature_min - temp_start
                        ) * tank_capacity
                        tank_in_space = (temperature_max - temp_start) * tank_capacity
                        if pv_surplus[i] * cop_tank[i] < tank_in_necessary:
                            q_tank_in = tank_in_necessary
                            pv_surplus_after_hot_water_tank[i] -= pv_surplus[i]
                            grid_demand_after_hot_water_tank[i] += (
                                tank_in_necessary / cop_tank[i] - pv_surplus[i]
                            )
                        elif (
                            tank_in_necessary
                            < pv_surplus[i] * cop_tank[i]
                            < tank_in_space
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
                            tank_temperature[i - 1]
                            + (q_tank_in - tank_loss[i]) / tank_capacity
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
                            tank_temperature[i - 1]
                            + (q_tank_in - tank_loss[i]) / tank_capacity
                        )

                    else:
                        q_tank_in = 0
                        self.Q_DHWTank_in[i] = q_tank_in
                        tank_temperature[i] = (
                            tank_temperature[i - 1]
                            + (q_tank_in - tank_loss[i]) / tank_capacity
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
                            tank_temperature[i - 1]
                            + (q_tank_in - tank_loss[i]) / tank_capacity
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
                        tank_temperature[i] = (
                            tank_temperature[i - 1]
                            - (q_tank_out + tank_loss[i]) / tank_capacity
                        )

            self.Q_DHWTank = (tank_temperature + 273.15) * tank_capacity

        else:
            grid_demand_after_hot_water_tank = grid_demand
            pv_surplus_after_hot_water_tank = pv_surplus

        return grid_demand_after_hot_water_tank, pv_surplus_after_hot_water_tank

    def calc_grid(self, grid_demand: np.array, pv_surplus: np.array):
        self.Grid = grid_demand
        self.Grid2Load = grid_demand
        self.PV2Grid = pv_surplus
        self.Feed2Grid = pv_surplus
        self.TotalCost = self.ElectricityPrice * grid_demand - pv_surplus * self.FiT

    def run_fuel_boiler_ref(self):
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
        self.run_heatpump_ref()
        self.T_outside = outside_temperature
        self.Q_Solar = q_solar
        self.Q_RoomHeating = heating_demand
        self.Q_RoomCooling = cooling_demand
        self.E_RoomCooling = cooling_demand / self.CoolingCOP
        self.T_Room = room_temperature
        self.T_BuildingMass = building_mass_temperature
        self.HotWaterProfile = hot_water_demand
        self.TotalCost += (heating_cost + cooling_cost + hot_water_cost) / 8760
        return self

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
        grid_demand, pv_surplus = self.calc_hot_water_tank_energy(
            grid_demand, pv_surplus
        )
        self.calc_grid(grid_demand, pv_surplus)
        return self

    def run(self):
        if self.scenario.boiler.type in ["Air_HP", "Ground_HP"]:
            model_ref = self.run_heatpump_ref()
        else:
            model_ref = self.run_fuel_boiler_ref()
        logger.info(f"RefCost: {round(self.TotalCost.sum(), 2)}")
        return model_ref

