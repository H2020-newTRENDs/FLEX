from flex.flex_operation.abstract import AbstractOperationModel
from core.household.abstract_scenario import AbstractScenario
from core.elements.rc_model import R5C1Model

import numpy as np


class RefOperationModel(AbstractOperationModel):
    def calculate_tank_energy(self,
                              electricity_grid_demand: np.array,
                              electricity_surplus: np.array,
                              hot_water_demand: np.array,
                              TankMinTemperature: float,
                              TankMaxTemperature: float,
                              TankSize: float,
                              TankSurfaceArea: float,
                              TankLoss: float,
                              TankSurroundingTemperature,
                              TankStartTemperature: float,
                              cop: float,
                              cop_tank
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
        electricity_surplus_after_tank = np.copy(electricity_surplus)
        electricity_deficit = np.zeros(electricity_surplus.shape)  # extra electricity needed to keep above min temp
        grid_demand_after_tank = np.copy(electricity_grid_demand)  # gets altered through  the calculation
        Q_tank_in = np.zeros(electricity_surplus.shape)  # hot water charged into the tank
        Q_tank_out = np.zeros(electricity_surplus.shape)  # hot water provided by tank
        Q_HP = np.copy(hot_water_demand)  # hot water produced by HP
        E_HP = Q_HP / cop  # electricity consumption of HP
        current_tank_temperature = np.zeros(electricity_surplus.shape)
        tank_loss_hourly = np.zeros(electricity_surplus.shape)

        for i, element in enumerate(electricity_surplus_after_tank):
            if i == 0:
                # calculate hourly temperature loss:
                tank_loss_hourly[i] = (TankStartTemperature - TankSurroundingTemperature) * \
                                      TankSurfaceArea * TankLoss  # W
                # surplus of PV electricity is used to charge the tank:
                current_tank_temperature[i] = TankStartTemperature + (
                        electricity_surplus[i] * cop_tank[i] - tank_loss_hourly[i]) \
                                              / (TankSize * self.cp_water)
                # Q_HP_DHW is increased:
                Q_HP[i] = electricity_surplus[i] * cop_tank[i] + hot_water_demand[i]
                E_HP[i] = electricity_surplus[i] + hot_water_demand[i] / cop[
                    i]  # electric consumption is increased by surplus
                Q_tank_in[i] = electricity_surplus[i] * cop_tank[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if current_tank_temperature[i] > TankMaxTemperature:
                    electricity_surplus_after_tank[i] = (current_tank_temperature[i] - TankMaxTemperature) * (
                            TankSize * self.cp_water) / cop_tank[i]  # W
                    current_tank_temperature[i] = TankMaxTemperature
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP[i] = (current_tank_temperature[i] - TankMaxTemperature) * \
                              (TankSize * self.cp_water) + hot_water_demand[i]
                    Q_tank_in[i] = (current_tank_temperature[i] - TankMaxTemperature) * \
                                   (TankSize * self.cp_water)
                    E_HP[i] = ((current_tank_temperature[i] - TankMaxTemperature) * \
                               (TankSize * self.cp_water)) / cop_tank[i] + hot_water_demand / cop[i]

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if current_tank_temperature[i] <= TankMinTemperature:
                    electricity_deficit[i] = (TankMinTemperature - current_tank_temperature[i]) * \
                                             (TankSize * self.cp_water) / cop_tank[i]
                    current_tank_temperature[i] = TankMinTemperature
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_tank[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP[i] = electricity_deficit[i] * cop_tank[i] + hot_water_demand[i]
                    Q_tank_in[i] = electricity_deficit[i] * cop_tank[i]
                    E_HP[i] = electricity_deficit[i] + hot_water_demand[i] / cop[i]

                # if there is energy in the tank it will be used for heating:
                if current_tank_temperature[i] > TankMinTemperature:
                    EnergyInTank = (current_tank_temperature[i] - TankMinTemperature) * (
                            TankSize * self.cp_water)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if electricity_grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if electricity_grid_demand[i] > hot_water_demand[i] / cop[i]:
                            # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP[i] = 0  # DHW is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = hot_water_demand[i]
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP[i] = DeficitEnergy  # dhw partly provided by tank
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if electricity_surplus[i] <= hot_water_demand[i] / cop[i]:
                            # calculate the part that could be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / cop[i] -
                                                        electricity_surplus[i]) * cop[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP[i] = 0  # dhw is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = remaining_heating_Energy
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP[i] = DeficitEnergy
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

            if i > 0:
                # calculate hourly temperature loss:
                tank_loss_hourly[i] = (current_tank_temperature[i - 1] - TankSurroundingTemperature) * \
                                      TankSurfaceArea * TankLoss  # W
                # surplus of PV electricity is used to charge the tank:
                current_tank_temperature[i] = current_tank_temperature[i - 1] + \
                                              (electricity_surplus[i] *
                                               cop_tank[i] - tank_loss_hourly[i]) / \
                                              (TankSize * self.cp_water)
                # Q_HP_DHW is increased:
                Q_HP[i] = electricity_surplus[i] * cop_tank[i] + hot_water_demand[i]
                E_HP[i] = electricity_surplus[i] + hot_water_demand[i] / cop[i]
                Q_tank_in[i] = electricity_surplus[i] * cop_tank[i]

                # if temperature exceeds maximum temperature, surplus of electricity is calculated
                # and temperature is kept at max temperature:
                if current_tank_temperature[i] > TankMaxTemperature:
                    electricity_surplus_after_tank[i] = (current_tank_temperature[i] - TankMaxTemperature) * (
                            TankSize * self.cp_water) / cop_tank[i]  # W
                    current_tank_temperature[i] = TankMaxTemperature
                    # Q_HP_DHW and Q_DHWTank_in:
                    Q_HP[i] = (current_tank_temperature[i] - TankMaxTemperature) * \
                              (TankSize * self.cp_water) + hot_water_demand[i]
                    E_HP[i] = ((current_tank_temperature[i] - TankMaxTemperature) * \
                               (TankSize * self.cp_water)) / cop_tank[i] + hot_water_demand[i] / cop[i]
                    Q_tank_in[i] = (current_tank_temperature[i] - TankMaxTemperature) * \
                                   (TankSize * self.cp_water)

                # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                # and necessary electricity is calculated
                if current_tank_temperature[i] <= TankMinTemperature:
                    electricity_deficit[i] = (TankMinTemperature - current_tank_temperature[i]) * \
                                             (TankSize * self.cp_water) / cop_tank[i]
                    current_tank_temperature[i] = TankMinTemperature
                    # electric energy is raised by the amount the tank has to be heated:
                    grid_demand_after_tank[i] += electricity_deficit[i]
                    # Q_DHW_HP is raised by the amount the tank has to be heated:
                    Q_HP[i] = electricity_deficit[i] * cop_tank[i] + hot_water_demand[i]
                    E_HP[i] = electricity_deficit[i] + hot_water_demand[i] / cop[i]
                    Q_tank_in[i] = electricity_deficit[i] * cop_tank[i]

                # if there is energy in the tank it will be used for heating:
                if current_tank_temperature[i] > TankMinTemperature:
                    EnergyInTank = (current_tank_temperature[i] - TankMinTemperature) * (
                            TankSize * self.cp_water)

                    # if the PV does not cover the whole electricity in this hour, the tank will cover for dhw
                    if electricity_grid_demand[i] > 0:
                        # if the PV does not cover any energy for the dhw:
                        if electricity_grid_demand[i] > hot_water_demand[i] / cop[i]:
                            # means that the heating is not covered at all by the PV (it covers other things first)
                            # if the Energy in the tank is enough to provide dhw, the hot_water_demand goes to 0
                            if EnergyInTank > hot_water_demand[i]:
                                SurplusEnergy = EnergyInTank - hot_water_demand[i]
                                Q_HP[i] = 0  # DHW is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = hot_water_demand[i]
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)  # the tank temperature drops to minimal
                                # temperature + the energy that is left
                            # if the energy in the tank is not enough Q_HP_DHW will be just reduced
                            if EnergyInTank <= hot_water_demand[i]:
                                DeficitEnergy = hot_water_demand[i] - EnergyInTank
                                Q_HP[i] = DeficitEnergy  # dhw partly provided by tank
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                if grid_demand_after_tank[i] < 0:
                                    current_tank_temperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

                        # if the PV does cover part of the dhw demand:
                        if hot_water_demand[i] / cop[i] >= electricity_surplus[i] > 0:
                            # calculate the part that can be covered by the tank:
                            remaining_heating_Energy = (hot_water_demand[i] / cop[i] - electricity_surplus[i]) * cop[i]
                            # if the energy in the tank is enough to cover the remaining heating energy:
                            if EnergyInTank > remaining_heating_Energy:
                                SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                Q_HP[i] = 0  # dhw is provided by tank
                                E_HP[i] = 0
                                Q_tank_out[i] = remaining_heating_Energy
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature + SurplusEnergy / (
                                        TankSize * self.cp_water)
                            # if the energy in the tank is not enough to cover the remaining heating energy:
                            if EnergyInTank <= remaining_heating_Energy:
                                DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                Q_HP[i] = DeficitEnergy
                                E_HP[i] = DeficitEnergy / cop[i]
                                Q_tank_out[i] = EnergyInTank
                                grid_demand_after_tank[i] -= Q_tank_out[i] / cop[i]
                                current_tank_temperature[i] = TankMinTemperature  # the tank temperature
                                # drops to minimal energy

        E_tank = (current_tank_temperature + 273.15) * self.cp_water * TankSize
        return grid_demand_after_tank, electricity_surplus_after_tank, Q_tank_out, Q_tank_in, Q_HP, E_HP, E_tank

    def calculate_battery_energy(self, grid_demand, electricity_surplus):
        """
        Assumption: The battery is not discharging itself.
        """
        Capacity = self.scenario.battery.capacity  # kWh
        MaxChargePower = self.scenario.battery.charge_power_max  # kW
        MaxDischargePower = self.scenario.battery.discharge_power_max  # kW
        ChargeEfficiency = self.scenario.battery.charge_efficiency
        DischargeEfficiency = self.scenario.battery.discharge_efficiency

        # Battery is not charged at the beginning of the simulation
        BatterySOC = np.zeros(electricity_surplus.shape)
        surplus_after_battery = np.zeros(electricity_surplus.shape)
        load_after_battery = np.copy(grid_demand)
        Battery2Load = np.zeros(electricity_surplus.shape)
        Battery_charge = np.zeros(electricity_surplus.shape)

        for i, element in enumerate(BatterySOC):
            if i == 0:  # there will be no charging at 01:00 clock in the morning of the 1st january:
                BatterySOC[i] = 0

            if i > 0:
                # if there is surplus energy, the battery will be charged
                if electricity_surplus[i] > 0:
                    # check if the battery can store the power or if its already fully charged:
                    if BatterySOC[i - 1] < Capacity:  # there is space to charge

                        if electricity_surplus[i] <= MaxChargePower:  # maximum charging power is not exceeded
                            # determine how much capacity is available:
                            capacity_left = Capacity - BatterySOC[i - 1]
                            BatterySOC[i] = BatterySOC[i - 1] + electricity_surplus[i] * ChargeEfficiency
                            Battery_charge[i] = electricity_surplus[i] * ChargeEfficiency
                            # check if the battery exceeds maximum charge limit, if yes:
                            if BatterySOC[i] > Capacity:
                                charging_energy = capacity_left / ChargeEfficiency
                                surplus_after_battery[i] = electricity_surplus[i] - charging_energy
                                BatterySOC[i] = Capacity
                                Battery_charge[i] = charging_energy

                        if electricity_surplus[i] > MaxChargePower:  # maximum charging power is exceeded
                            # determine how much capacity is available:
                            capacity_left = Capacity - BatterySOC[i - 1]
                            BatterySOC[i] = BatterySOC[i - 1] + MaxChargePower * ChargeEfficiency
                            surplus_after_battery[i] = electricity_surplus[i] - MaxChargePower
                            Battery_charge[i] = MaxChargePower * ChargeEfficiency
                            # check if the battery exceeds maximum capacity limit, if yes:
                            if BatterySOC[i] > Capacity:
                                charging_energy = capacity_left / ChargeEfficiency
                                surplus_after_battery[i] = electricity_surplus[i] - charging_energy
                                BatterySOC[i] = Capacity
                                Battery_charge[i] = charging_energy

                    # if battery can not be charged because its full:
                    if BatterySOC[i - 1] == Capacity:
                        BatterySOC[i] = Capacity
                        surplus_after_battery[i] = electricity_surplus[i]

                # if there is no surplus of energy and the battery has energy stored, it will provide energy:
                if electricity_surplus[i] == 0:
                    # check if battery has power:
                    if BatterySOC[i - 1] > 0:
                        # if the power in battery is enough to cover whole electricity demand:
                        if BatterySOC[i - 1] > grid_demand[i] / DischargeEfficiency:
                            load_after_battery[i] = 0
                            BatterySOC[i] = BatterySOC[i - 1] - grid_demand[i] / DischargeEfficiency
                            Battery2Load[i] = grid_demand[i] / DischargeEfficiency
                            # check if maximum discharge power is exceeded:
                            if load_after_battery[i] / DischargeEfficiency > MaxDischargePower:
                                load_after_battery[i] = grid_demand[i] - MaxDischargePower
                                BatterySOC[i] = BatterySOC[i - 1] - MaxDischargePower / DischargeEfficiency
                                Battery2Load[i] = MaxDischargePower

                        # if the power in the battery is not enough to cover the whole electricity demand:
                        if BatterySOC[i - 1] <= grid_demand[i] / DischargeEfficiency:
                            load_after_battery[i] = grid_demand[i] - BatterySOC[i - 1] * DischargeEfficiency
                            BatterySOC[i] = 0
                            Battery2Load[i] = BatterySOC[i - 1] * DischargeEfficiency
                            # check if max discharge power is exceeded:
                            if BatterySOC[i - 1] > MaxDischargePower:
                                BatterySOC[i] = BatterySOC[i - 1] - MaxDischargePower / DischargeEfficiency
                                load_after_battery[i] = grid_demand[i] - MaxDischargePower
                                Battery2Load[i] = MaxDischargePower

        self.BatCharge = Battery_charge
        self.PV2Bat = Battery_charge
        self.BatSoC = BatterySOC
        self.BatDischarge = Battery2Load  # as the battery gets only discharged by providing load coverage
        self.Bat2Load = Battery2Load
        return load_after_battery, surplus_after_battery, BatterySOC, Battery2Load

    def fill_parameter_values(self) -> None:
        """ fills all self parameter values with the input values"""
        # price
        self.electricity_price = self.scenario.electricityprice_class.electricity_price  # C/Wh
        # Feed in Tariff of Photovoltaic
        self.FiT = self.scenario.feedintariff_class.feed_in_tariff  # C/Wh
        # solar gains:
        self.Q_Solar = self.calculate_solar_gains()  # W
        # outside temperature
        self.T_outside = self.scenario.region.temperature  # °C

        # COP of cooling
        self.CoolingCOP = self.scenario.space_cooling_technology.efficiency  # single value because it is not dependent on time
        # electricity load profile
        self.BaseLoadProfile = self.scenario.electricitydemand_class.electricity_demand
        # PV profile
        self.PhotovoltaicProfile = self.scenario.pv.power
        # HotWater
        self.HotWaterProfile = self.scenario.hotwaterdemand_class.hot_water_demand

        # building data: is not saved in results (to much and not useful)

        # Heating Tank data
        # Mass of water in tank
        self.M_WaterTank_heating = self.scenario.space_heating_tank.size
        # Surface of Tank in m2
        self.A_SurfaceTank_heating = self.scenario.space_heating_tank.surface_area
        # insulation of tank, for calc of losses
        self.U_ValueTank_heating = self.scenario.space_heating_tank.loss
        self.T_TankStart_heating = self.scenario.space_heating_tank.temperature_start
        # surrounding temp of tank
        self.T_TankSurrounding_heating = self.scenario.space_heating_tank.temperature_surrounding

        # DHW Tank data
        # Mass of water in tank
        self.M_WaterTank_DHW = self.scenario.hot_water_tank.size
        # Surface of Tank in m2
        self.A_SurfaceTank_DHW = self.scenario.hot_water_tank.surface_area
        # insulation of tank, for calc of losses
        self.U_ValueTank_DHW = self.scenario.hot_water_tank.loss
        self.T_TankStart_DHW = self.scenario.hot_water_tank.temperature_start
        # surrounding temp of tank
        self.T_TankSurrounding_DHW = self.scenario.hot_water_tank.temperature_surrounding

        # heat pump
        self.SpaceHeating_HeatPumpMaximalElectricPower = self.scenario.boiler.power_max

        # Battery data
        self.ChargeEfficiency = self.scenario.battery.charge_efficiency
        self.DischargeEfficiency = self.scenario.battery.discharge_efficiency

    def run(self):
        """
        Assumption for the Reference scenario: the produced PV power is always used for the immediate electric demand,
        if there is a surplus of PV power, it will be used to charge the Battery,
        if the Battery is full or not available, the PV power will be used to heat the HotWaterTank,
        if the HotWaterTank is not available or full, the PV power will sold to the Grid.
        The surplus of PV energy is never used to Preheat or Precool the building.

        """
        self.fill_parameter_values()  # set input parameters to self values
        # calculate the heating and cooling energy and indoor air + thermal mass temperature:
        heating_demand, cooling_demand, T_Room, Tm_t = \
            R5C1Model(self.scenario).calculate_heating_and_cooling_demand(
                thermal_start_temperature=self.thermal_mass_start_temperature,
                static=False)
        room_heating = heating_demand
        self.Tm_t = Tm_t
        self.T_room = T_Room
        self.Q_RoomCooling = cooling_demand
        # check if heating element has to be used for the HP:
        heating_element = np.zeros(heating_demand.shape)
        for index, element in enumerate(heating_demand):
            if element > self.scenario.boiler.power_max * self.SpaceHeatingHourlyCOP[index]:
                heating_demand[index] = self.scenario.boiler.power_max * \
                                        self.SpaceHeatingHourlyCOP[index]
                heating_element[index] = element - self.scenario.boiler.power_max * \
                                         self.SpaceHeatingHourlyCOP[index]

        self.Q_HeatingElement = heating_element
        # electricity for hot water:
        hot_water_electricity_demand = self.HotWaterProfile / self.HotWaterHourlyCOP
        # electricity for heating:
        heating_electricity_demand = heating_demand / self.SpaceHeatingHourlyCOP
        # electricity for cooling:
        cooling_electricity_demand = cooling_demand / self.CoolingCOP

        # add up all electric loads but without Heating:
        total_electricity_load = self.scenario.electricitydemand_class.electricity_demand + \
                                 hot_water_electricity_demand + \
                                 heating_electricity_demand + \
                                 cooling_electricity_demand + \
                                 heating_element

        # The PV profile is subtracted from the total load
        total_load_minus_pv = total_electricity_load - self.PhotovoltaicProfile  # kW

        # determine the surplus PV power:
        PV_profile_surplus = np.copy(total_load_minus_pv)
        PV_profile_surplus[PV_profile_surplus > 0] = 0
        PV_profile_surplus = abs(PV_profile_surplus)
        # determine PV2Load
        self.PV2Load = self.PhotovoltaicProfile - PV_profile_surplus

        # Total load profile can not be negative: make negative numbers to 0
        total_load_minus_pv[total_load_minus_pv < 0] = 0

        # if neither battery nor hot water tank are used:
        grid_demand = np.copy(total_load_minus_pv)
        # electricity surplus is the energy that is sold back to the grid because it can not be used:
        electricity_sold = np.copy(PV_profile_surplus)  # kW

        # Battery storage:
        # if battery is used:
        if self.scenario.battery.capacity > 0:
            total_load_after_battery, Electricity_surplus_Battery, BatterySOC, Battery2Load = \
                self.calculate_battery_energy(grid_demand, PV_profile_surplus)
            grid_demand = total_load_after_battery
            # amount of electricity from PV to Battery:
            electricity_sold = Electricity_surplus_Battery
        else:
            self.BatCharge = np.full((8760,), 0)
            self.PV2Bat = np.full((8760,), 0)
            self.BatSoC = np.full((8760,), 0)
            self.BatDischarge = np.full((8760,), 0)
            self.Bat2Load = np.full((8760,), 0)

        # DHW storage:
        if self.scenario.hot_water_tank.size > 0:
            grid_demand_after_DHW, electricity_surplus_after_DHW, Q_tank_out_DHW, Q_tank_in_DHW, Q_HP_DHW, \
            E_HP_DHW, E_tank_DHW = self.calculate_tank_energy(
                    electricity_grid_demand=grid_demand,
                    electricity_surplus=electricity_sold,
                    hot_water_demand=self.HotWaterProfile,
                    TankMinTemperature=self.scenario.hot_water_tank.temperature_min,
                    TankMaxTemperature=self.scenario.hot_water_tank.temperature_max,
                    TankSize=self.scenario.hot_water_tank.size,
                    TankSurfaceArea=self.scenario.hot_water_tank.surface_area,
                    TankLoss=self.U_ValueTank_DHW,
                    TankSurroundingTemperature=self.T_TankSurrounding_DHW,
                    TankStartTemperature=self.T_TankStart_DHW,
                    cop=self.HotWaterHourlyCOP,
                    cop_tank=self.HotWaterHourlyCOP_tank
                )
            electricity_sold = electricity_surplus_after_DHW
            grid_demand = grid_demand_after_DHW
            self.Q_DHWTank_out = Q_tank_out_DHW
            self.Q_DHWTank_in = Q_tank_in_DHW
            self.E_DHW_HP_out = E_HP_DHW
            self.E_DHWTank = E_tank_DHW
        else:
            self.Q_DHWTank_out = np.full((8760,), 0)
            self.Q_DHWTank_in = np.full((8760,), 0)
            self.E_DHW_HP_out = self.HotWaterProfile / self.HotWaterHourlyCOP
            self.E_DHWTank = np.full((8760,), 0)

        # When there is PV surplus energy it either goes to a storage or is sold to the grid:
        # Water Tank as storage:
        if self.scenario.space_heating_tank.size > 0:
            grid_demand_after_heating, electricity_surplus_after_heating, Q_tank_out_heating, Q_tank_in_heating, \
            Q_HP_heating, E_HP_heating, E_tank_heating = self.calculate_tank_energy(
                electricity_grid_demand=grid_demand,
                electricity_surplus=electricity_sold,
                hot_water_demand=heating_demand,
                TankMinTemperature=self.scenario.space_heating_tank.temperature_min,
                TankMaxTemperature=self.scenario.space_heating_tank.temperature_max,
                TankSize=self.M_WaterTank_heating,
                TankSurfaceArea=self.A_SurfaceTank_heating,
                TankLoss=self.U_ValueTank_heating,
                TankSurroundingTemperature=self.T_TankSurrounding_heating,
                TankStartTemperature=self.T_TankStart_heating,
                cop=self.SpaceHeatingHourlyCOP,
                cop_tank=self.SpaceHeatingHourlyCOP_tank
            )
            # remaining surplus of electricity
            electricity_sold = electricity_surplus_after_heating
            grid_demand = grid_demand_after_heating
            self.Q_HeatingTank_out = Q_tank_out_heating
            self.Q_HeatingTank_in = Q_tank_in_heating
            self.E_Heating_HP_out = E_HP_heating
            self.E_HeatingTank = E_tank_heating
        else:
            self.Q_HeatingTank_out = np.full((8760,), 0)
            self.Q_HeatingTank_in = np.full((8760,), 0)
            self.E_Heating_HP_out = heating_demand / self.SpaceHeatingHourlyCOP
            self.E_HeatingTank = np.full((8760,), 0)

        # calculate the electricity cost:
        price_hourly = self.scenario.electricityprice_class.electricity_price
        FIT = self.scenario.feedintariff_class.feed_in_tariff
        self.total_operation_cost = price_hourly * grid_demand - electricity_sold * FIT
        print('Total Operation Cost reference: ' + str(round(self.total_operation_cost.sum(), 2)))

        # grid variables
        self.Grid = grid_demand
        self.Grid2Load = grid_demand
        self.Grid2Bat = np.full((8760,), 0)

        # PV variables
        self.PV2Grid = electricity_sold

        # electric load
        self.Load = total_electricity_load

        # electricity fed back to the grid
        self.Feed2Grid = electricity_sold

        # room heating
        self.Q_room_heating = room_heating
        self.Q_HeatingTank_bypass = room_heating - self.Q_HeatingTank_out

        # DHW
        self.Q_DHWTank_bypass = self.HotWaterProfile - self.Q_DHWTank_out


if __name__ == "__main__":
    from flex.flex_operation.data_collector import ReferenceDataCollector

    scenario = AbstractScenario(scenario_id=5)
    reference_model = RefOperationModel(scenario)
    reference_model.run()
    hourly_results = ReferenceDataCollector(reference_model).collect_reference_results_hourly()
    yearly_results = ReferenceDataCollector(reference_model).collect_reference_results_yearly()

    # save results to database
    ReferenceDataCollector(reference_model).save_hourly_results()
    ReferenceDataCollector(reference_model).save_yearly_results()

    pass
