import numpy as np
from models.operation.abstract import OperationModel
from basics.kit import get_logger

logger = get_logger(__name__)


class RefOperationModel(OperationModel):

    def calculate_tank_energy(self,
                              grid_demand: np.array,
                              pv_surplus: np.array,
                              hot_water_demand: np.array,
                              temperature_min: float,
                              temperature_max: float,
                              size: float,
                              surface_area: float,
                              loss: float,
                              surrounding_temperature: float,
                              start_temperature: float,
                              cop: np.array,
                              cop_tank: np.array
                              ):


        """
        calculates the usage and the energy in the domestic hot water tank. The tank is charged by the heat pump
        with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized the energy for DHW will always be solemnly provided by the DHW tank. Therefore the
        heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.
        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """
        # this gets altered through the calculation and returned
        electricity_surplus_after_tank = np.copy(pv_surplus)
        electricity_deficit = np.zeros(pv_surplus.shape)  # extra electricity needed to keep above min temp
        grid_demand_after_tank = np.copy(grid_demand)  # gets altered through  the calculation
        Q_tank_in = np.zeros(pv_surplus.shape)  # hot water charged into the tank
        Q_tank_out = np.zeros(pv_surplus.shape)  # hot water provided by tank
        Q_HP = np.copy(hot_water_demand)  # hot water produced by HP
        E_HP = Q_HP / cop  # electricity consumption of HP
        current_tank_temperature = np.zeros(pv_surplus.shape)
        tank_loss_hourly = np.zeros(pv_surplus.shape)

        for i, element in enumerate(electricity_surplus_after_tank):
            if i == 0:
                # calculate hourly temperature loss:
                tank_loss_hourly[i] = (start_temperature - surrounding_temperature) * \
                                      surface_area * loss  # W
                # surplus of PV electricity is used to charge the tank:
                current_tank_temperature[i] = start_temperature + (
                        pv_surplus[i] * cop_tank[i] - tank_loss_hourly[i]) \
                                              / (size * self.CPWater)
                # Q_HP_DHW is increased:
                Q_HP[i] = pv_surplus[i] * cop_tank[i] + hot_water_demand[i]
                # electric consumption is increased by surplus
                E_HP[i] = pv_surplus[i] + hot_water_demand[i] / cop[i]
                Q_tank_in[i] = pv_surplus[i] * cop_tank[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if current_tank_temperature[i] > temperature_max:
                    electricity_surplus_after_tank[i] = (current_tank_temperature[i] - temperature_max) * (
                            size * self.CPWater) / cop_tank[i]  # W
                    current_tank_temperature[i] = temperature_max
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP[i] = (current_tank_temperature[i] - temperature_max) * \
                              (size * self.CPWater) + hot_water_demand[i]
                    Q_tank_in[i] = (current_tank_temperature[i] - temperature_max) * \
                                   (size * self.CPWater)
                    E_HP[i] = ((current_tank_temperature[i] - temperature_max) * (size * self.CPWater)) / \
                              cop_tank[i] + hot_water_demand / cop[i]

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if current_tank_temperature[i] <= temperature_min:
                    electricity_deficit[i] = (temperature_min - current_tank_temperature[i]) * \
                                             (size * self.CPWater) / cop_tank[i]
                    current_tank_temperature[i] = temperature_min
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_tank[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP[i] = electricity_deficit[i] * cop_tank[i] + hot_water_demand[i]
                    Q_tank_in[i] = electricity_deficit[i] * cop_tank[i]
                    E_HP[i] = electricity_deficit[i] + hot_water_demand[i] / cop[i]

                # if there is energy in the tank it will be used for heating:
                if current_tank_temperature[i] > temperature_min:
                    EnergyInTank = (current_tank_temperature[i] - temperature_min) * (
                            size * self.CPWater)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if grid_demand[i] > hot_water_demand[i] / cop[i]:
                            # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP[i] = 0  # DHW is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = hot_water_demand[i]
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min + SurplusEnergy / (
                                        size * self.CPWater)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP[i] = DeficitEnergy  # dhw partly provided by tank
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if pv_surplus[i] <= hot_water_demand[i] / cop[i]:
                            # calculate the part that could be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / cop[i] -
                                                        pv_surplus[i]) * cop[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP[i] = 0  # dhw is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = remaining_heating_Energy
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min + SurplusEnergy / (
                                        size * self.CPWater)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP[i] = DeficitEnergy
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

            if i > 0:
                # calculate hourly temperature loss:
                tank_loss_hourly[i] = (current_tank_temperature[i - 1] - surrounding_temperature) * \
                                      surface_area * loss  # W
                # surplus of PV electricity is used to charge the tank:
                current_tank_temperature[i] = current_tank_temperature[i - 1] + \
                                              (pv_surplus[i] *
                                               cop_tank[i] - tank_loss_hourly[i]) / \
                                              (size * self.CPWater)
                # Q_HP_DHW is increased:
                Q_HP[i] = pv_surplus[i] * cop_tank[i] + hot_water_demand[i]
                E_HP[i] = pv_surplus[i] + hot_water_demand[i] / cop[i]
                Q_tank_in[i] = pv_surplus[i] * cop_tank[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if current_tank_temperature[i] > temperature_max:
                    electricity_surplus_after_tank[i] = (current_tank_temperature[i] - temperature_max) * (
                            size * self.CPWater) / cop_tank[i]  # W
                    current_tank_temperature[i] = temperature_max
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP[i] = (current_tank_temperature[i] - temperature_max) * \
                              (size * self.CPWater) + hot_water_demand[i]
                    E_HP[i] = ((current_tank_temperature[i] - temperature_max) * \
                               (size * self.CPWater)) / cop_tank[i] + hot_water_demand[i] / cop[i]
                    Q_tank_in[i] = (current_tank_temperature[i] - temperature_max) * \
                                   (size * self.CPWater)

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if current_tank_temperature[i] <= temperature_min:
                    electricity_deficit[i] = (temperature_min - current_tank_temperature[i]) * \
                                             (size * self.CPWater) / cop_tank[i]
                    current_tank_temperature[i] = temperature_min
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_tank[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP[i] = electricity_deficit[i] * cop_tank[i] + hot_water_demand[i]
                    E_HP[i] = electricity_deficit[i] + hot_water_demand[i] / cop[i]
                    Q_tank_in[i] = electricity_deficit[i] * cop_tank[i]

                # if there is energy in the tank it will be used for heating:
                if current_tank_temperature[i] > temperature_min:
                    EnergyInTank = (current_tank_temperature[i] - temperature_min) * (
                            size * self.CPWater)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if grid_demand[i] > hot_water_demand[i] / cop[i]:
                            # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP[i] = 0  # DHW is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = hot_water_demand[i]
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min + SurplusEnergy / (
                                        size * self.CPWater)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP[i] = DeficitEnergy  # dhw partly provided by tank
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                if grid_demand_after_tank[i] < 0:
                                    current_tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if hot_water_demand[i] / cop[i] >= pv_surplus[i] > 0:
                            # calculate the part that can be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / cop[i] - pv_surplus[i]) * cop[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP[i] = 0  # dhw is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = remaining_heating_Energy
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min + SurplusEnergy / (
                                        size * self.CPWater)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP[i] = DeficitEnergy
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

        E_tank = (current_tank_temperature + 273.15) * self.CPWater * size
        return grid_demand_after_tank, electricity_surplus_after_tank, Q_tank_out, Q_tank_in, Q_HP, E_HP, E_tank

    def calc_space_heating_demand(self):
        self.Q_RoomHeating, self.Q_RoomCooling, self.T_Room, self.T_BuildingMass = \
            self.calculate_heating_and_cooling_demand(
                thermal_start_temperature=self.BuildingMassTemperatureStartValue,
                static=False)
        hp_max = self.SpaceHeating_HeatPumpMaximalElectricPower * self.SpaceHeatingHourlyCOP

        self.Q_HeatingElement = np.where(self.Q_RoomHeating - hp_max < 0, 0, self.Q_RoomHeating - hp_max)
        self.Q_HeatingTank_bypass = self.Q_RoomHeating - self.Q_HeatingElement
        self.E_Heating_HP_out = self.Q_HeatingTank_bypass / self.SpaceHeatingHourlyCOP

        self.Q_HeatingTank = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_in = np.zeros(self.Q_HeatingTank_bypass.shape)
        self.Q_HeatingTank_out = np.zeros(self.Q_HeatingTank_bypass.shape)

    def calc_space_cooling_demand(self):
        self.E_CoolingDemand = self.Q_RoomCooling / self.CoolingCOP

    def calc_hot_water_demand(self):
        self.Q_DHWTank_bypass = self.HotWaterProfile
        self.E_DHW_HP_out = self.Q_DHWTank_bypass / self.HotWaterHourlyCOP

    def calc_load(self):
        self.Load = self.BaseLoadProfile + self.E_Heating_HP_out + self.Q_HeatingElement + \
                    self.E_CoolingDemand + self.E_DHW_HP_out
        grid_demand = np.where(self.Load - self.PhotovoltaicProfile < 0, 0, self.Load - self.PhotovoltaicProfile)
        pv_surplus = np.where(self.PhotovoltaicProfile - self.Load < 0, 0, self.PhotovoltaicProfile - self.Load)
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
                    bat_soc_start = 0
                else:
                    bat_soc_start = self.BatSoC[i - 1]

                if pv_surplus[i] > 0:
                    if bat_soc_start < capacity:
                        if pv_surplus[i] <= max_charge_power:
                            charge_amount = pv_surplus[i]
                        else:
                            charge_amount = max_charge_power
                        pv_surplus_after_battery[i] -= charge_amount
                        self.BatSoC[i] = bat_soc_start + charge_amount * charge_efficiency
                        self.PV2Bat[i] = charge_amount * charge_efficiency
                        if self.BatSoC[i] > capacity:
                            right_charge_amount = (capacity - bat_soc_start) / charge_efficiency
                            pv_surplus_after_battery[i] += charge_amount - right_charge_amount
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
                        grid_demand_after_battery[i] = grid_demand[i] - discharge_amount * discharge_efficiency
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

    def calculate_ev_energy_backup(self, grid_demand, pv_surplus):

        self.EVSoC = np.zeros(pv_surplus.shape)
        self.EVCharge = np.zeros(pv_surplus.shape)
        self.EVDischarge = self.scenario.behavior.vehicle_demand
        self.EV2Bat = np.zeros(pv_surplus.shape)
        self.EV2Load = np.zeros(pv_surplus.shape)

        if self.scenario.vehicle.capacity > 0:

            capacity = self.scenario.vehicle.capacity  # kWh
            ev_demand = self.scenario.behavior.vehicle_demand
            home_status = np.array(self.scenario.behavior.vehicle_at_home, dtype=int)
            max_charge_power = self.scenario.vehicle.charge_power_max  # kW
            max_discharge_power = self.scenario.vehicle.discharge_power_max  # kW
            charge_efficiency = self.scenario.vehicle.charge_efficiency

            pv_surplus_after_ev = np.copy(pv_surplus)
            grid_demand_after_ev = np.copy(grid_demand)
            self.EVCharge = np.zeros(pv_surplus.shape)

            for i in range(0, len(pv_surplus)):
                if i == 0:  # there will be no charging at 01:00 clock in the morning of the 1st january:
                    # EV will always be charged if it is at home and not fully charged:
                    if self.EVSoC[i] < capacity:
                        # check how much capacity is missing:
                        missing_capacity = capacity - self.EVSoC[i]
                        # if missing capacity is greater than max charge power, only charge with max charge power
                        if missing_capacity > max_discharge_power:
                            self.EVSoC[i] += max_charge_power  # charge with max charge power
                            grid_demand_after_ev[i] += max_charge_power / charge_efficiency  # load is increased
                            self.EVCharge[i] = max_charge_power / charge_efficiency
                        # if missing capacity is smaller than max charge power, charge the remainder:
                        else:
                            self.EVSoC[i] += missing_capacity
                            grid_demand_after_ev[i] += missing_capacity / charge_efficiency  # load is increased
                            self.EVCharge[i] = missing_capacity / charge_efficiency

                if i > 0:
                    if i == 13:
                        a=1
                    # check if the EV is at home:
                    if home_status[i] == 1:  # if it is at home it can be charged
                        # check if the EV is already fully charged:
                        if self.EVSoC[i - 1] >= capacity:
                            self.EVSoC[i] = self.EVSoC[i - 1]
                        else:  # EV is not fully charged and needs to be charged:
                            # determine the energy that needs to be charged:
                            missing_capacity = capacity - self.EVSoC[i - 1]
                            # check if the missing capacity is greater than the maximum charging power
                            # and determine the electricity that will be charged this hour:
                            if missing_capacity > max_charge_power:
                                charging_energy = max_charge_power / charge_efficiency  # electricity that is charged
                                self.EVSoC[i] = self.EVSoC[i-1] + max_charge_power  # EV is being charged
                                grid_demand_after_ev[i] += charging_energy  # load is increased
                                self.EVCharge[i] = charging_energy
                            else:
                                charging_energy = missing_capacity / charge_efficiency  # electricity that is charged
                                self.EVSoC[i] = self.EVSoC[i-1] + charging_energy  # EV is fully charged
                                grid_demand_after_ev[i] += charging_energy  # load is increased
                                self.EVCharge[i] = charging_energy

                    else:  # EV is not at home and looses power
                        # calculate the Battery SOC
                        self.EVSoC[i] = self.EVSoC[i - 1] - ev_demand[i]
                        # check if Battery SOC ever drops below 0:
                        if self.EVSoC[i] < 0:
                            print("EV Battery dropped below 0. Reference model is infeasible")

            # check if the resulting load that has been increased through the EV can be covered partly by remaining
            # PV generation:
            for index, surplus in enumerate(pv_surplus_after_ev):
                # check if surplus is available:
                if surplus > 0:
                    # check if load demand has risen in this time due to the EV:
                    if grid_demand_after_ev[index] > 0:
                        # check if the surplus is larger than the load:
                        if surplus > grid_demand_after_ev[index]:
                            # only part of the surplus is used to cover the demand
                            pv_surplus_after_ev[index] -= grid_demand_after_ev[index]  # surplus is reduced by load
                            grid_demand_after_ev[index] = 0  # load goes to 0
                        else:  # surplus is smaller than the load that needs to be covered
                            # only part of the load is covered by the surplus
                            grid_demand_after_ev[index] -= surplus  # load gets reduced by surplus
                            pv_surplus_after_ev[index] = 0  # surplus drops to 0
        else:
            grid_demand_after_ev = grid_demand
            pv_surplus_after_ev = pv_surplus

        return grid_demand_after_ev, pv_surplus_after_ev

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
                    ev_soc_start = 0
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
                                grid_demand_after_ev[i] += charge_amount / charge_efficiency - pv_surplus[i]
                                self.Grid2EV[i] = charge_amount - pv_surplus[i] * charge_efficiency
                            else:
                                pv_surplus_after_ev[i] -= charge_amount / charge_efficiency

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

    def calc_hot_water_tank_energy_backup(self, grid_demand: np.array, pv_surplus: np.array):
        """
        calculates the usage and the energy in the domestic hot water tank.
        The tank is charged by the heat pump with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized, the energy for DHW will always be solemnly provided by the DHW tank.
        Therefore the heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.

        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """

        self.Q_DHWTank = np.full(pv_surplus.shape,
                                 self.T_TankStart_DHW * self.CPWater * self.scenario.hot_water_tank.size)
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
            grid_demand_after_hot_water_tank = np.copy(grid_demand)  # gets altered through  the calculation
            pv_surplus_after_hot_water_tank = np.copy(pv_surplus)
            electricity_deficit = np.zeros(pv_surplus.shape)  # extra electricity needed to keep above min temp

            Q_HP = np.copy(hot_water_demand)  # hot water produced by HP
            tank_temperature = np.zeros(pv_surplus.shape)
            tank_loss = np.zeros(pv_surplus.shape)

            for i in range(0, len(pv_surplus)):

                if i == 0:
                    temp_start = start_temperature
                else:
                    temp_start = tank_temperature[i - 1]
                tank_loss[i] = (temp_start - surrounding_temperature) * surface_area * loss  # W
                tank_temperature[i] = temp_start + (pv_surplus[i] * cop_tank[i] - tank_loss[i]) / tank_capacity

                # Q_HP_DHW is increased:
                Q_HP[i] = pv_surplus[i] * cop_tank[i] + hot_water_demand[i]
                self.E_DHW_HP_out[i] = pv_surplus[i] + hot_water_demand[i] / cop[i]
                self.Q_DHWTank_in[i] = pv_surplus[i] * cop_tank[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if tank_temperature[i] > temperature_max:
                    pv_surplus_after_hot_water_tank[i] = (tank_temperature[i] - temperature_max) * tank_capacity / cop_tank[i]  # W
                    tank_temperature[i] = temperature_max
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP[i] = (tank_temperature[i] - temperature_max) * tank_capacity + hot_water_demand[i]
                    self.E_DHW_HP_out[i] = ((tank_temperature[i] - temperature_max) * tank_capacity) / cop_tank[i] + hot_water_demand[i] / cop[i]
                    self.Q_DHWTank_in[i] = (tank_temperature[i] - temperature_max) * tank_capacity

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if tank_temperature[i] <= temperature_min:
                    electricity_deficit[i] = (temperature_min - tank_temperature[i]) * tank_capacity / cop_tank[i]
                    tank_temperature[i] = temperature_min
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_hot_water_tank[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP[i] = electricity_deficit[i] * cop_tank[i] + hot_water_demand[i]
                    self.E_DHW_HP_out[i] = electricity_deficit[i] + hot_water_demand[i] / cop[i]
                    self.Q_DHWTank_in[i] = electricity_deficit[i] * cop_tank[i]

                # if there is energy in the tank it will be used for heating:
                if tank_temperature[i] > temperature_min:
                    EnergyInTank = (tank_temperature[i] - temperature_min) * tank_capacity

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if grid_demand[i] > hot_water_demand[i] / cop[i]:
                            # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP[i] = 0  # DHW is provided by tank
                                self.E_DHW_HP_out[i] = 0
                                self.Q_DHWTank_out[i] = hot_water_demand[i]
                                grid_demand_after_hot_water_tank[i] -= self.Q_DHWTank_out[i] / cop[i]
                                tank_temperature[i] = temperature_min + SurplusEnergy / tank_capacity  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP[i] = DeficitEnergy  # dhw partly provided by tank
                                self.E_DHW_HP_out[i] = DeficitEnergy / cop[i]
                                self.Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_hot_water_tank[i] -= self.Q_DHWTank_out[i] / cop[i]
                                if grid_demand_after_hot_water_tank[i] < 0:
                                    tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if hot_water_demand[i] / cop[i] >= pv_surplus[i] > 0:
                            # calculate the part that can be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / cop[i] - pv_surplus[i]) * cop[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP[i] = 0  # dhw is provided by tank
                                self.E_DHW_HP_out[i] = 0
                                self.Q_DHWTank_out[i] = remaining_heating_Energy
                                grid_demand_after_hot_water_tank[i] -= self.Q_DHWTank_out[i] / cop[i]
                                tank_temperature[i] = temperature_min + SurplusEnergy / tank_capacity
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP[i] = DeficitEnergy
                                self.E_DHW_HP_out[i] = DeficitEnergy / cop[i]
                                self.Q_DHWTank_out[i] = EnergyInTank
                                grid_demand_after_hot_water_tank[i] -= self.Q_DHWTank_out[i] / cop[i]
                                tank_temperature[i] = temperature_min  # the tank temperature
                                # drops to minimal energy

            self.Q_DHWTank = (tank_temperature + 273.15) * tank_capacity

        else:
            grid_demand_after_hot_water_tank = grid_demand
            pv_surplus_after_hot_water_tank = pv_surplus

        return grid_demand_after_hot_water_tank, pv_surplus_after_hot_water_tank

    def calc_hot_water_tank_energy(self, grid_demand: np.array, pv_surplus: np.array):
        """
        calculates the usage and the energy in the domestic hot water tank.
        The tank is charged by the heat pump with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized, the energy for DHW will always be solemnly provided by the DHW tank.
        Therefore the heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses.

        Returns: grid_demand_after_DHW, electricity_surplus_after_DHW
        """

        self.Q_DHWTank = np.full(pv_surplus.shape,
                                 self.T_TankStart_DHW * self.CPWater * self.scenario.hot_water_tank.size)
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
                tank_loss[i] = (temp_start - surrounding_temperature) * surface_area * loss  # W

                if pv_surplus[i] > 0:

                    if temp_start < temperature_min:
                        tank_in_necessary = (temperature_min - temp_start) * tank_capacity
                        tank_in_space = (temperature_max - temp_start) * tank_capacity
                        if pv_surplus[i] * cop_tank[i] < tank_in_necessary:
                            q_tank_in = tank_in_necessary
                            pv_surplus_after_hot_water_tank[i] -= pv_surplus[i]
                            grid_demand_after_hot_water_tank[i] += tank_in_necessary / cop_tank[i] - pv_surplus[i]
                        elif tank_in_necessary < pv_surplus[i] * cop_tank[i] < tank_in_space:
                            q_tank_in = pv_surplus[i] * cop_tank[i]
                            pv_surplus_after_hot_water_tank[i] -= pv_surplus[i]
                        else:
                            q_tank_in = tank_in_space
                            pv_surplus_after_hot_water_tank[i] -= q_tank_in / cop_tank[i]
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        tank_temperature[i] = tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity

                    elif temperature_min <= temp_start < temperature_max:
                        tank_in_space = (temperature_max - temp_start) * tank_capacity
                        if pv_surplus[i] * cop_tank[i] > tank_in_space:
                            q_tank_in = tank_in_space
                        else:
                            q_tank_in = pv_surplus[i] * cop_tank[i]
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        pv_surplus_after_hot_water_tank[i] -= q_tank_in / cop_tank[i]
                        tank_temperature[i] = tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity

                    else:
                        q_tank_in = 0
                        self.Q_DHWTank_in[i] = q_tank_in
                        tank_temperature[i] = tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity

                else:

                    if temp_start < temperature_min:
                        tank_in_necessary = (temperature_min - temp_start) * tank_capacity
                        q_tank_in = tank_in_necessary
                        grid_demand_after_hot_water_tank[i] += q_tank_in / cop_tank[i]
                        self.Q_DHWTank_in[i] = q_tank_in
                        self.E_DHW_HP_out[i] += q_tank_in / cop_tank[i]
                        tank_temperature[i] = tank_temperature[i - 1] + (q_tank_in - tank_loss[i]) / tank_capacity

                    else:
                        tank_out_limit = (temp_start - temperature_min) * tank_capacity
                        if tank_out_limit < hot_water_demand[i]:
                            q_tank_out = tank_out_limit
                        else:
                            q_tank_out = hot_water_demand[i]
                        self.Q_DHWTank_out[i] = q_tank_out
                        self.Q_DHWTank_bypass[i] -= q_tank_out
                        self.E_DHW_HP_out[i] -= q_tank_out / cop[i]
                        grid_demand_after_hot_water_tank[i] -= q_tank_out / cop[i]  # ?
                        tank_temperature[i] = tank_temperature[i - 1] - (q_tank_out + tank_loss[i]) / tank_capacity

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
        logger.info(f'Reference operation cost: {round(self.TotalCost.sum(), 2)}')

    def run(self):
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
        # grid_demand, pv_surplus = self.calculate_ev_energy_backup(grid_demand, pv_surplus)
        grid_demand, pv_surplus = self.calculate_ev_energy(grid_demand, pv_surplus)
        # grid_demand, pv_surplus = self.calc_hot_water_tank_energy_backup(grid_demand, pv_surplus)
        grid_demand, pv_surplus = self.calc_hot_water_tank_energy(grid_demand, pv_surplus)






        # # space heating tank storage
        # if self.scenario.space_heating_tank.size > 0:
        #     grid_demand, pv_surplus, Q_tank_out_heating, Q_tank_in_heating, Q_HP_heating, E_HP_heating, E_tank_heating = \
        #         self.calculate_tank_energy(
        #             grid_demand=grid_demand,
        #             pv_surplus=pv_surplus,
        #             hot_water_demand=self.Q_HeatingTank_bypass,
        #             temperature_min=self.scenario.space_heating_tank.temperature_min,
        #             temperature_max=self.scenario.space_heating_tank.temperature_max,
        #             size=self.M_WaterTank_heating,
        #             surface_area=self.A_SurfaceTank_heating,
        #             loss=self.U_LossTank_heating,
        #             surrounding_temperature=self.T_TankSurrounding_heating,
        #             start_temperature=self.T_TankStart_heating,
        #             cop=self.SpaceHeatingHourlyCOP,
        #             cop_tank=self.SpaceHeatingHourlyCOP_tank
        #             )
        #     self.Q_HeatingTank_out = Q_tank_out_heating
        #     self.Q_HeatingTank_in = Q_tank_in_heating
        #     self.E_Heating_HP_out = E_HP_heating
        #     self.Q_HeatingTank = E_tank_heating
        # else:
        #     self.Q_HeatingTank_out = np.full((8760,), 0)
        #     self.Q_HeatingTank_in = np.full((8760,), 0)
        #     self.E_Heating_HP_out = self.Q_HeatingTank_bypass / self.SpaceHeatingHourlyCOP
        #     self.Q_HeatingTank = np.full((8760,), 0)  # wrong. should be initial energy constant





        self.calc_grid(grid_demand, pv_surplus)




        return self


