from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table, REG_Var
from A_Infrastructure.A1_CONS import CONS
from B_Classes.B2_Building import HeatingCoolingNoSEMS
from B_Classes.B1_Household import Household
from C_Model_Operation.C2_DataCollector import DataCollector
from C_Model_Operation.C0_Model import MotherModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from functools import wraps


def performance_counter(func):
    @wraps(func)
    def wrapper():
        t_start = perf_counter()
        func()
        t_end = perf_counter()
        print("function >>{}<< time for execution: {}".format(func.__name__, t_end - t_start))

    return wrapper


class no_SEMS(MotherModel):

    def remove_household_IDs(self) -> pd.DataFrame:
        """ return a pandas dataframe of the Gen_OBJ_ID_Household frame but removes all building IDs but one because
        the reference model calculates all buidling IDs at the same time"""
        return self.ID_Household.loc[self.ID_Household.loc[:, "ID_Building"] == 1].reset_index(drop=True)


    def gen_Household(self, row_id):
        # TODO replace self.IDHousehold
        ParaSeries = self.remove_household_IDs().iloc[row_id]
        HouseholdAgent = Household(ParaSeries, self.Conn)
        return HouseholdAgent

    def gen_Environment(self, row_id):
        EnvironmentSeries = self.ID_Environment.iloc[row_id]
        return EnvironmentSeries

    def calculate_initial_thermal_mass_temp(self):
        """
        the thermal mass temperature of the building should be in a steady state with the indoor temperature at the level
        of the minimal indoor temperature and the outside temperature being the outside temperature at the first hour
        of the year. The solar gains are set to 0.
        """
        Buildings = HeatingCoolingNoSEMS()
        # start time steps to calculate constant values
        runtime = 100
        # start with 100 hours,  check if Tm_t is constant:
        Temperature_outside = np.array(
            [self.outside_temperature.to_numpy()[0]] * runtime)  # take first index of temperature vector
        Q_sol = np.array([[0] * len(self.Buildings)] * runtime)
        # initial thermal mass temperature should be between T_min_indoor and T_max_indoor (22°C should always be ok)
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_constant = Buildings.ref_HeatingCooling(
            Temperature_outside,
            Q_solar=Q_sol,
            initial_thermal_mass_temp=22,
            T_air_min=20,  # indoor set temperature is always at 20 for all scenarios, has marginal impact
            T_air_max=27)  # has no influence cause its winter -- > only heating
        # check if stationary condition has set in
        for i in range(T_thermalMass_constant.shape[1]):
            while -1e-6 >= T_thermalMass_constant[-1, i] - T_thermalMass_constant[-2, i] or \
                    1e-6 <= T_thermalMass_constant[-1, i] - T_thermalMass_constant[-2, i]:
                runtime += 100
                Temperature_outside = np.array([self.outside_temperature.to_numpy()[0]] * runtime)
                Q_sol = np.array([[0] * len(self.Buildings)] * runtime)
                Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_constant = Buildings.ref_HeatingCooling(
                    Temperature_outside,
                    Q_solar=Q_sol,
                    initial_thermal_mass_temp=22,
                    T_air_min=20,  # indoor set temperature is always at 20 for all scenarios, has marginal impact
                    T_air_max=27)  # has no influence cause its winter -- > only heating
        return T_thermalMass_constant[-1, :]

    def calculate_DHW_tank_energy(self):
        """calculates the usage and the energy in the domestic hot water tank. The tank is charged by the heat pump
        with the COP for hot water.
        Whenever there is surplus of PV electricity the DHW tank is charged and it is discharged by the DHW-usage.
        IF a DHW tank is utilized the energy for DHW will always be solemnly provided by the DHW tank. Therefore the
        heat pump input into the DHW tank must be in accordance with the output of the DHW tank + losses."""



    def calculate_heating_tank_energy(self, Total_Load_minusPV, Q_Heating_noDR, PV_profile_surplus, Household):
        """
        Calculates the energy/temperature inside the hot water tank and the heating energy that has to be actually used
        when the tank energy is always used for heating when necessary. Input parameters are all provided in kW
        (except COP which has no unit). Return values are also provided in kW.
        """
        T_outside = self.outside_temperature.Temperature.to_numpy()
        Total_Load_minusPV = Total_Load_minusPV * 1_000  # W
        PV_profile_surplus = PV_profile_surplus * 1_000  # W
        TankSize = float(Household.SpaceHeating.TankSize)
        TankMinTemperature = float(Household.SpaceHeating.TankMinimalTemperature)
        TankMaxTemperature = float(Household.SpaceHeating.TankMaximalTemperature)
        TankSurfaceArea = float(Household.SpaceHeating.TankSurfaceArea)
        TankLoss = float(Household.SpaceHeating.TankLoss)
        TankSurroundingTemperature = float(Household.SpaceHeating.TankSurroundingTemperature)
        TankStartTemperature = float(Household.SpaceHeating.TankStartTemperature)
        COP_SpaceHeating = self.COP_HP(self.outside_temperature, 35,
                                       Household.SpaceHeating.CarnotEfficiencyFactor,
                                       Household.SpaceHeating.Name_SpaceHeatingPumpType)  # 35 °C supply temperature

        # Assumption: Tank is always kept at minimum temperature except when it is charged with surplus energy:
        # Note: I completely neglect Hot Water here (but its also neglected in the optimization)
        TankLoss_hourly = np.zeros(PV_profile_surplus.shape)
        CurrentTankTemperature = np.zeros(PV_profile_surplus.shape)
        Electricity_surplus = np.zeros(PV_profile_surplus.shape)
        Electricity_deficit = np.zeros(PV_profile_surplus.shape)
        Q_heating_HP = np.copy(Q_Heating_noDR) * 1_000  # W
        Total_Load_WaterTank = np.copy(Total_Load_minusPV)  # W

        # TODO if t outside > 20°C dont charge tank (only minium to keep above 28°C)
        for index, row in enumerate(TankLoss_hourly):
            for column, element in enumerate(row):
                if index == 0:
                    TankLoss_hourly[index, column] = (TankStartTemperature - TankSurroundingTemperature) * \
                                                     TankSurfaceArea * TankLoss  # W
                    # surplus of PV electricity is used to charge the tank:
                    CurrentTankTemperature[index, column] = TankStartTemperature + \
                                                            (PV_profile_surplus[index, column] *
                                                             COP_SpaceHeating[index] - TankLoss_hourly[index, column]) / \
                                                            (TankSize * self.CPWater)

                    # if temperature exceeds maximum temperature, surplus of electricity is calculated
                    # and temperature is kept at max temperature:
                    if CurrentTankTemperature[index, column] > TankMaxTemperature:
                        Electricity_surplus[index, column] = (CurrentTankTemperature[
                                                                  index, column] - TankMaxTemperature) * (
                                                                     TankSize * self.CPWater) / COP_SpaceHeating[
                                                                 index]  # W
                        CurrentTankTemperature[index, column] = TankMaxTemperature

                    # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                    # and neccesary electricity is calculated
                    if CurrentTankTemperature[index, column] <= TankMinTemperature:
                        Electricity_deficit[index, column] = (TankMinTemperature - CurrentTankTemperature[
                            index, column]) * (TankSize * self.CPWater) / COP_SpaceHeating[index]
                        CurrentTankTemperature[index, column] = TankMinTemperature
                        # electric energy is raised by the amount the tank has to be heated:
                        Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] + \
                                                              Electricity_deficit[index, column]

                    # if there is energy in the tank it will be used for heating:
                    if CurrentTankTemperature[index, column] > TankMinTemperature:
                        EnergyInTank = (CurrentTankTemperature[index, column] - TankMinTemperature) * (
                                TankSize * self.CPWater)

                        # if the PV does not cover the whole elctricity in this hour, the tank will cover for heating
                        if Total_Load_minusPV[index, column] > 0:
                            # if the PV does not cover any energy for the heating:
                            if Total_Load_minusPV[index, column] > Q_heating_HP[index, column] / COP_SpaceHeating[
                                index]:  # means that the heating is not covered at all by the PV (it covers other things first)
                                # if the Energy in the tank is enough to heat the building, the Q_heating_HP goes to 0
                                if EnergyInTank > Q_heating_HP[index, column]:
                                    SurplusEnergy = EnergyInTank - Q_heating_HP[index, column]
                                    Q_heating_HP[index, column] = 0  # Building is heated by tank
                                    CurrentTankTemperature[index, column] = TankMinTemperature + SurplusEnergy / (
                                            TankSize * self.CPWater)  # the tank temperature drops to minimal temperature + the energy that is left
                                # if the energy in the tank is not enough to heat the building Q_heating_HP will be just reduced
                                if EnergyInTank <= Q_heating_HP[index, column]:
                                    DeficitEnergy = Q_heating_HP[index, column] - EnergyInTank
                                    Q_heating_HP[index, column] = DeficitEnergy  # Building is partly heated by tank
                                    CurrentTankTemperature[
                                        index, column] = TankMinTemperature  # the tank temperature drops to minimal energy

                            # if the PV does cover part of the heating energy:
                            if Total_Load_minusPV[index, column] <= Q_heating_HP[index, column] / COP_SpaceHeating[
                                index]:
                                # calculate the part than can be covered by the tank:
                                remaining_heating_Energy = (Q_heating_HP[index, column] / COP_SpaceHeating[index] -
                                                            Total_Load_minusPV[index, column]) * COP_SpaceHeating[index]
                                # if the energy in the tank is enough to cover the remaining heating energy:
                                if EnergyInTank > remaining_heating_Energy:
                                    SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                    Q_heating_HP[index, column] = 0  # Building is heated by tank and PV
                                    CurrentTankTemperature[index, column] = TankMinTemperature + SurplusEnergy / (
                                            TankSize * self.CPWater)
                                # if the energy in the tank is not enough to cover the remaining heating energy:
                                if EnergyInTank <= remaining_heating_Energy:
                                    DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                    Q_heating_HP[index, column] = DeficitEnergy
                                    CurrentTankTemperature[
                                        index, column] = TankMinTemperature  # the tank temperature drops to minimal energy

                if index > 0:

                    # if outside temperature is > 20°C the tank will not be charged except to keep min temperature:
                    # we assume that heating is not necessary when outside temp is above 20°C
                    if T_outside[index] > 20:
                        TankLoss_hourly[index, column] = (CurrentTankTemperature[
                                                              index - 1, column] - TankSurroundingTemperature) * \
                                                         TankSurfaceArea * TankLoss  # W
                        CurrentTankTemperature[index, column] = CurrentTankTemperature[index - 1, column] - \
                                                                (TankLoss_hourly[index, column]) / \
                                                                (TankSize * self.CPWater)

                        # if the tank is empty, it will be charged to keep it at minimum temperature
                        if CurrentTankTemperature[index, column] <= TankMinTemperature:
                            Electricity_deficit[index, column] = (TankMinTemperature - CurrentTankTemperature[
                                index, column]) * (TankSize * self.CPWater) / COP_SpaceHeating[index]
                            # if there is PV power left, the tank is kept at minimum temp with PV power:
                            if PV_profile_surplus[index, column] >= Electricity_deficit[index, column]:
                                CurrentTankTemperature[index, column] = TankMinTemperature
                                # the surplus of PV is sold to the grid minus the part used for heating at minimum:
                                Electricity_surplus[index, column] = PV_profile_surplus[index, column] - \
                                                                     Electricity_deficit[index, column]
                            # if PV power is not enough to keep the tank at minimum temp:
                            elif PV_profile_surplus[index, column] > 0 and PV_profile_surplus[index, column] < \
                                    Electricity_deficit[index, column]:
                                CurrentTankTemperature[index, column] = TankMinTemperature
                                # electricity deficit gets reduced by the available PV power
                                Electricity_deficit[index, column] = Electricity_deficit[index, column] - \
                                                                     PV_profile_surplus[index, column]
                                Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] + \
                                                                      Electricity_deficit[index, column]

                            # if no PV power available total load is increased
                            else:
                                CurrentTankTemperature[index, column] = TankMinTemperature
                                # electric energy is raised by the amount the tank has to be heated:
                                Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] + \
                                                                      Electricity_deficit[index, column]

                    # if outside temperature is < 20°C the tank will always be charged:
                    else:
                        TankLoss_hourly[index, column] = (CurrentTankTemperature[
                                                              index - 1, column] - TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
                        CurrentTankTemperature[index, column] = CurrentTankTemperature[index - 1, column] + \
                                                                (PV_profile_surplus[index, column] *
                                                                 COP_SpaceHeating[index] -
                                                                 TankLoss_hourly[index, column]) / \
                                                                (TankSize * self.CPWater)

                        # if temperature exceed maximum temperature, surplus of electricity is calculated
                        # and temperature is kept at max temperature:
                        if CurrentTankTemperature[index, column] > TankMaxTemperature:
                            Electricity_surplus[index, column] = (CurrentTankTemperature[
                                                                      index, column] - TankMaxTemperature) * (
                                                                         TankSize * self.CPWater) / COP_SpaceHeating[
                                                                     index]  # W
                            CurrentTankTemperature[index, column] = TankMaxTemperature

                        # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                        # and neccesary electricity is calculated
                        if CurrentTankTemperature[index, column] <= TankMinTemperature:
                            Electricity_deficit[index, column] = (TankMinTemperature - CurrentTankTemperature[
                                index, column]) * (TankSize * self.CPWater) / COP_SpaceHeating[index]
                            CurrentTankTemperature[index, column] = TankMinTemperature
                            # electric energy is raised by the amount the tank has to be heated:
                            Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] + \
                                                                  Electricity_deficit[index, column]

                        # if there is energy in the tank it will be used for heating:
                        if CurrentTankTemperature[index, column] > TankMinTemperature:
                            EnergyInTank = (CurrentTankTemperature[index, column] - TankMinTemperature) * (
                                    TankSize * self.CPWater)

                            # if the PV does not cover the whole electricity in this hour, the tank will cover for heating
                            if Total_Load_minusPV[index, column] > 0:
                                # if the PV does not cover any energy for the heating:
                                if Total_Load_minusPV[index, column] > Q_heating_HP[index, column] / COP_SpaceHeating[
                                    index]:  # means that the heating is not covered at all by the PV (it covers other things first)
                                    # if the Energy in the tank is enough to heat the building, the Q_heating_HP goes to 0
                                    if EnergyInTank > Q_heating_HP[index, column]:
                                        SurplusEnergy = EnergyInTank - Q_heating_HP[index, column]
                                        Q_heating_HP[index, column] = 0  # Building is heated by tank
                                        CurrentTankTemperature[index, column] = TankMinTemperature + SurplusEnergy / (
                                                TankSize * self.CPWater)  # the tank temperature drops to minimal temperature + the energy that is left
                                        # the total electric energy will be reduced by the amount of the HP electric energy:
                                        Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] - \
                                                                              Q_Heating_noDR[index, column] / \
                                                                              COP_SpaceHeating[index]
                                    # if the energy in the tank is not enough to heat the building Q_heating_HP will be just reduced
                                    if EnergyInTank <= Q_heating_HP[index, column]:
                                        DeficitEnergy = Q_heating_HP[index, column] - EnergyInTank
                                        Q_heating_HP[index, column] = DeficitEnergy  # Building is partly heated by tank
                                        CurrentTankTemperature[
                                            index, column] = TankMinTemperature  # the tank temperature drops to minimal energy
                                        # the total electric energy will be reduced by the part of the HP as well:
                                        Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] - \
                                                                              (Q_Heating_noDR[index, column] -
                                                                               Q_heating_HP[index, column]) / \
                                                                              COP_SpaceHeating[index]

                                # if the PV does cover part of the heating energy:
                                if Total_Load_minusPV[index, column] <= Q_heating_HP[index, column] / COP_SpaceHeating[
                                    index]:
                                    # calculate the part than can be covered by the tank:
                                    remaining_heating_Energy = (Q_heating_HP[index, column] / COP_SpaceHeating[index] -
                                                                Total_Load_minusPV[index, column]) * COP_SpaceHeating[
                                                                   index]
                                    # if the energy in the tank is enough to cover the remaining heating energy:
                                    if EnergyInTank > remaining_heating_Energy:
                                        SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                        Q_heating_HP[index, column] = 0  # Building is heated by tank and PV
                                        CurrentTankTemperature[index, column] = TankMinTemperature + SurplusEnergy / (
                                                TankSize * self.CPWater)
                                        # electric total energy is reduced by the part of the HP:
                                        Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] - \
                                                                              Q_Heating_noDR[index, column] / \
                                                                              COP_SpaceHeating[index]
                                    # if the energy in the tank is not enough to cover the remaining heating energy:
                                    if EnergyInTank <= remaining_heating_Energy:
                                        DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                        Q_heating_HP[index, column] = DeficitEnergy
                                        CurrentTankTemperature[
                                            index, column] = TankMinTemperature  # the tank temperature drops to minimal energy
                                        # electric energy is reduced by the part of the HP:
                                        Total_Load_WaterTank[index, column] = Total_Load_minusPV[index, column] - \
                                                                              (Q_Heating_noDR[index, column] -
                                                                               Q_heating_HP[index, column]) / \
                                                                              COP_SpaceHeating[index]

        #  Q_heating_HP is now the heating energy with the usage of the tank
        # Electricity_surplus is the electricity that can not be stored in the tank
        # Electricity_deficit is the electricity needed to keep the tank on temperature
        # TankLoss_hourly are the total losses of the tank
        # Total_Load_WaterTank is the actual electric load with the use of the water storage

        # convert the energie back to kW:
        TankLoss_hourly = TankLoss_hourly / 1_000
        Electricity_surplus = Electricity_surplus / 1_000
        Electricity_deficit = Electricity_deficit / 1_000
        Q_heating_HP = Q_heating_HP / 1_000
        Total_Load_WaterTank = Total_Load_WaterTank / 1_000  # TODO find mistake where total load water < 0!!
        Total_Load_WaterTank[Total_Load_WaterTank < 0] = 0
        return TankLoss_hourly, CurrentTankTemperature, Electricity_surplus, Electricity_deficit, Q_heating_HP, Total_Load_WaterTank  # all energies in kW, temperature in °C

    def calculate_battery_energy(self, grid_demand, electricity_surplus, Household):
        """

        Assumption: The battery is not discharging itself.
        """
        Capacity = float(Household.Battery.Capacity)  # kWh
        MaxChargePower = float(Household.Battery.MaxChargePower)  # kW
        MaxDischargePower = float(Household.Battery.MaxDischargePower)  # kW
        ChargeEfficiency = float(Household.Battery.ChargeEfficiency)
        DischargeEfficiency = float(Household.Battery.DischargeEfficiency)

        # Battery is not charged at the beginning of the simulation
        BatterySOC = np.zeros(electricity_surplus.shape)
        surplus_after_battery = np.zeros(electricity_surplus.shape)
        Total_load_battery = np.copy(grid_demand)
        Battery2Load = np.zeros(electricity_surplus.shape)

        for index, row in enumerate(BatterySOC):
            for column, element in enumerate(row):
                if index == 0:  # there will be no charging at 01:00 clock in the morning of the 1st january:
                    BatterySOC[index, column] = 0

                if index > 0:
                    # if there is surplus energy, the battery will be charged
                    if electricity_surplus[index, column] > 0:
                        # check if the battery can store the power or if its already fully charged:
                        if BatterySOC[index - 1, column] < Capacity:  # there is space to charge

                            if electricity_surplus[
                                index, column] <= MaxChargePower:  # maximum charging power is not exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index - 1, column]
                                BatterySOC[index, column] = BatterySOC[index - 1, column] + electricity_surplus[
                                    index, column] * ChargeEfficiency
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    surplus_after_battery[index, column] = electricity_surplus[
                                                                               index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                            if electricity_surplus[
                                index, column] > MaxChargePower:  # maximum charging power is exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index - 1, column]
                                BatterySOC[index, column] = BatterySOC[
                                                                index - 1, column] + MaxChargePower * ChargeEfficiency
                                surplus_after_battery[index, column] = electricity_surplus[
                                                                           index, column] - MaxChargePower
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    surplus_after_battery[index, column] = electricity_surplus[
                                                                               index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                        # if battery can not be charged because its full:
                        if BatterySOC[index - 1, column] == Capacity:
                            BatterySOC[index, column] = Capacity
                            surplus_after_battery[index, column] = electricity_surplus[index, column]

                    # if there is no surplus of energy and the battery has energy stored, it will provide energy:
                    if electricity_surplus[index, column] == 0:
                        # check if battery has power:
                        if BatterySOC[index - 1, column] > 0:
                            # if the power in battery is enough to cover whole electricity demand:
                            if BatterySOC[index - 1, column] > grid_demand[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = 0
                                BatterySOC[index, column] = BatterySOC[index - 1, column] - grid_demand[
                                    index, column] / DischargeEfficiency
                                Battery2Load[index, column] = grid_demand[index, column] / DischargeEfficiency
                                # check if maximum discharge power is exceeded:
                                if Total_load_battery[index, column] / DischargeEfficiency > MaxDischargePower:
                                    Total_load_battery[index, column] = grid_demand[
                                                                            index, column] - MaxDischargePower
                                    BatterySOC[index, column] = BatterySOC[
                                                                    index - 1, column] - MaxDischargePower / DischargeEfficiency
                                    Battery2Load[index, column] = MaxDischargePower

                            # if the power in the battery is not enough to cover the whole electricity demand:
                            if BatterySOC[index - 1, column] <= grid_demand[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = grid_demand[index, column] - BatterySOC[
                                    index - 1, column] * DischargeEfficiency
                                BatterySOC[index, column] = 0
                                Battery2Load[index, column] = BatterySOC[index - 1, column] * DischargeEfficiency
                                # check if max discharge power is exceeded:
                                if BatterySOC[index - 1, column] > MaxDischargePower:
                                    BatterySOC[index, column] = BatterySOC[
                                                                    index - 1, column] - MaxDischargePower / DischargeEfficiency
                                    Total_load_battery[index, column] = grid_demand[
                                                                            index, column] - MaxDischargePower
                                    Battery2Load[index, column] = MaxDischargePower

        return Total_load_battery, surplus_after_battery, BatterySOC, Battery2Load  # kW and kWh

    def calculate_noDR(self, household_RowID: int, environment_RowID: int) -> (np.array(str), np.array(str)):
        """
        Assumption for the Reference scenario: the produced PV power is always used for the immediate electric demand,
        if there is a surplus of PV power, it will be used to charge the Battery,
        if the Battery is full or not available, the PV power will be used to heat the HotWaterTank,
        if the HotWaterTank is not available or full, the PV power will sold to the Grid.
        The surplus of PV energy is never used to Preheat or Precool the building.

        The Washing Machine as well as Dryer and Dishwasher are used during the day on a random time between 06:00
        and 22:00.
        """
        Household = self.gen_Household(household_RowID)  # Gen_OBJ_ID_Household
        Environment = self.gen_Environment(environment_RowID)  # Gen_Sce_ID_Environment

        # ID_PVType
        # HeatPumpType
        # TODO specify how to use different set temperatures for different scenarios!
        if Household.Name_AgeGroup == "Conventional":
            T_indoorSetMin = self.TargetTemperature.HeatingTargetTemperatureYoungNightReduction.to_numpy()
            T_indoorSetMax = self.TargetTemperature.CoolingTargetTemperatureYoung.to_numpy()
        elif Household.Name_AgeGroup == "SmartHome":
            T_indoorSetMin = self.TargetTemperature.HeatingTargetTemperatureSmartHome.to_numpy()
            T_indoorSetMax = self.TargetTemperature.CoolingTargetTemperatureSmartHome.to_numpy()
        else:
            assert "no valid option for indoor set temperature"

        # calculate solar gains from database
        # solar gains from different celestial direction
        Q_solar = self.calculate_solar_gains()

        # calculate the heating and cooling energy for all buildings in IDBuildingOption:
        # initial_thermal_mass_temperature = self.calculate_initial_thermal_mass_temp()

        # if no cooling is adopted --> raise max air temperature to 100 so it will never cool:
        if Household.SpaceCooling.AdoptionStatus == 0:
            T_indoorSetMax = np.full((8760,), 100)
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t_noDR = HeatingCoolingNoSEMS().ref_HeatingCooling(
            initial_thermal_mass_temp=15,
            T_air_min=T_indoorSetMin,
            T_air_max=T_indoorSetMax)
        # check if heating element has to be used for the HP:
        HeatingElement = np.zeros(Q_Heating_noDR.shape)
        for index, row in enumerate(Q_Heating_noDR):
            for column, element in enumerate(row):
                if element > Household.SpaceHeating.HeatPumpMaximalThermalPower:
                    Q_Heating_noDR[index, column] = Household.SpaceHeating.HeatPumpMaximalThermalPower
                    HeatingElement[index, column] = element - Household.SpaceHeating.HeatPumpMaximalThermalPower

        # change values to kW:
        Q_Heating_noDR = Q_Heating_noDR / 1_000
        Q_Cooling_noDR = Q_Cooling_noDR / 1_000
        HeatingElement = HeatingElement / 1_000

        # PV production profile:
        if Household.PV.ID_PVType in self.PhotovoltaicProfile["ID_PVType"].to_numpy():
            PV_profile = pd.to_numeric(
                self.PhotovoltaicProfile.loc[self.PhotovoltaicProfile["ID_PVType"] == Household.PV.ID_PVType, :].PVPower).to_numpy()
        else:
            PV_profile = np.full((8760,), 0)

        # Dishwasher Profile
        # TODO select one of the random profiles (now i only use number 1)
        RandomProfileNumber = 0
        # Dish Washer Profile
        # DishWasherStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherStartingHours, self.Conn)[
        #     "DishWasherHours " + str(RandomProfileNumber)].to_numpy()
        # DishWasherPower = float(self.AppliancePower.DishWasherPower)
        # DishWasherProfile = DishWasherStartingHours * DishWasherPower

        # Washing Machine Profile
        # WashingMachineStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineStartingHours, self.Conn)[
        #     "WashingMachineHours " + str(RandomProfileNumber)].to_numpy()
        # WashingMachinePower = float(self.AppliancePower.WashingMachinePower)
        # WashingMachineProfile = WashingMachineStartingHours * WashingMachinePower

        # Dryer Profile
        # DryerStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DryerStartingHours, self.Conn)[
        #     "DryerHours " + str(RandomProfileNumber)].to_numpy()
        # DryerPower = float(self.AppliancePower.DryerPower)
        # DryerProfile = DryerStartingHours * DryerPower

        # electricity for DHW:
        COP_SpaceHeating = self.COP_HP(self.outside_temperature, 35,
                                       Household.SpaceHeating.CarnotEfficiencyFactor,
                                       Household.SpaceHeating.Name_SpaceHeatingPumpType)  # 35 °C supply temperature

        COP_HotWater = self.COP_HP(self.outside_temperature, 55,
                                   Household.SpaceHeating.CarnotEfficiencyFactor,
                                   Household.SpaceHeating.Name_SpaceHeatingPumpType)  # 55 °C supply temperature

        Q_HotWater = self.HotWaterProfile.HotWater.to_numpy()
        DHWProfile = Q_HotWater / COP_HotWater

        # electricity for heating:
        HeatingProfile = np.zeros(Q_Heating_noDR.shape)
        for column in range(Q_Heating_noDR.shape[1]):
            HeatingProfile[:, column] = Q_Heating_noDR[:, column] / COP_SpaceHeating

        # electricity for cooling:
        AC_hourly_COP = self.AC_HourlyCOP .ACHourlyCOP.to_numpy()
        CoolingProfile = np.zeros(Q_Cooling_noDR.shape)
        for column in range(Q_Cooling_noDR.shape[1]):
            CoolingProfile[:, column] = Q_Cooling_noDR[:, column] / AC_hourly_COP

        # add up all electric loads but without Heating:
        Total_Load = np.zeros(Q_Heating_noDR.shape)
        for column in range(Q_Heating_noDR.shape[1]):
            Total_Load[:, column] = self.BaseLoadProfile + \
                                    DHWProfile + \
                                    HeatingProfile[:, column] + \
                                    CoolingProfile[:, column] + \
                                    HeatingElement[:, column]  # + \
            # DryerProfile + \
            # WashingMachineProfile + \
            # DishWasherProfile

        # only load of electric appliances:
        # Load_Appliances = DryerProfile + WashingMachineProfile + DishWasherProfile

        # The PV profile is substracted from the total load
        Total_Load_minusPV = np.zeros(Total_Load.shape)
        for column in range(Total_Load_minusPV.shape[1]):
            Total_Load_minusPV[:, column] = Total_Load[:, column] - PV_profile  # kW

        # determine the surplus PV power:
        PV_profile_surplus = np.copy(Total_Load_minusPV)
        PV_profile_surplus[PV_profile_surplus > 0] = 0
        PV_profile_surplus = abs(PV_profile_surplus)

        # Total load profile can not be negative: make negative numbers to 0
        Total_Load_minusPV[Total_Load_minusPV < 0] = 0

        # if neither battery nor hotwater tank are used:
        elec_grid_demand = np.copy(Total_Load_minusPV)
        # electricity surplus is the energy that is sold back to the grid because it can not be used:
        Electricity_surplus = np.copy(PV_profile_surplus)  # kW

        # Battery storage:
        # if battery is used:
        if float(Household.Battery.Capacity) > 0:
            Total_load_battery, Electricity_surplus_Battery, BatterySOC, Battery2Load = \
                self.calculate_battery_energy(elec_grid_demand, Electricity_surplus, Household)
            elec_grid_demand = Total_load_battery
            # amount of electricity from PV to Battery:
            PV2Battery = Electricity_surplus - Electricity_surplus_Battery
            Electricity_surplus = Electricity_surplus_Battery
        else:
            PV2Battery = np.zeros(Electricity_surplus.shape)
            BatterySOC = np.zeros(Electricity_surplus.shape)
            Battery2Load = np.zeros(Electricity_surplus.shape)

        # When there is PV surplus energy it either goes to a storage or is sold to the grid:
        # Water Tank as storage:
        # if float(Household.SpaceHeating.TankSize) > 0:
        #     TankLoss_hourly, CurrentTankTemperature, Electricity_surplus_tank, Electricity_deficit, Q_heating_withTank, Total_Load_WaterTank = \
        #         self.calculate_tank_energy(elec_grid_demand, Q_Heating_noDR, Electricity_surplus, Household)
        #     # remaining surplus of electricity
        #     Electricity_surplus = Electricity_surplus_tank
        #     elec_grid_demand = Total_Load_WaterTank
        #     Q_Heating_noDR = Q_heating_withTank

        # calculate the electricity cost:
        PriceHourly = \
            self.electricity_price.loc[self.electricity_price['ID_PriceType'] == Environment["ID_ElectricityPriceType"]][
                "ElectricityPrice"].to_numpy()

        FIT = self.feed_in_tariff.loc[(self.feed_in_tariff['ID_FeedinTariffType'] == Environment["ID_FeedinTariffType"])][
            "HourlyFeedinTariff"].to_numpy()  # \&
        # (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyFeedinTariff'].to_numpy()

        total_elec_cost = np.zeros(elec_grid_demand.shape)
        for column in range(elec_grid_demand.shape[1]):
            total_elec_cost[:, column] = PriceHourly * elec_grid_demand[:, column] - Electricity_surplus[:,
                                                                                     column] * FIT

        print("Total Operation Cost: " + str(round(total_elec_cost.sum(axis=0) / 100)))

        # Dataframe for yearly values:
        YearlyFrame = np.column_stack([np.full((elec_grid_demand.shape[1]), Household.ID),
                                       np.full((elec_grid_demand.shape[1]),
                                               np.arange(1, elec_grid_demand.shape[1] + 1)),  # ID Building from 1 to n
                                       np.full((elec_grid_demand.shape[1]), Environment.ID),
                                       total_elec_cost.sum(axis=0) / 100,  # €

                                       np.full((elec_grid_demand.shape[1]), self.BaseLoadProfile.sum()),
                                       # Base electricity load
                                       # Load_Appliances.sum(axis=0),
                                       Q_Heating_noDR.sum(axis=0),
                                       # Note that Roomheating and Tankheating are the same
                                       HeatingProfile.sum(axis=0),  # electricity for heating
                                       Q_Heating_noDR.sum(axis=0) / HeatingProfile.sum(axis=0),
                                       # heatpump Performance factor
                                       Q_Heating_noDR.sum(axis=0) - HeatingProfile.sum(axis=0),  # ambient heat
                                       HeatingElement.sum(axis=0),  # heating element
                                       Q_Heating_noDR.sum(axis=0),
                                       # Note that Roomheating and Tankheating are the same

                                       Q_Cooling_noDR.sum(axis=0),  # cooling energy
                                       CoolingProfile.sum(axis=0),  # cooling electricity

                                       np.full((elec_grid_demand.shape[1]), Q_HotWater.sum(axis=0)),  # hot water energy
                                       np.full((elec_grid_demand.shape[1]), DHWProfile.sum(axis=0)),
                                       # hot water electricity

                                       elec_grid_demand.sum(axis=0),
                                       elec_grid_demand.sum(axis=0),
                                       np.full((elec_grid_demand.shape[1]), 0),

                                       np.full((elec_grid_demand.shape[1]), PV_profile.sum()),
                                       np.full((elec_grid_demand.shape[1]), PV_profile.sum()) - Electricity_surplus.sum(
                                           axis=0) -
                                       PV2Battery.sum(axis=0),  # PV2Load
                                       PV2Battery.sum(axis=0),  # PV2Bat
                                       Electricity_surplus.sum(axis=0),  # PV2Grid

                                       PV2Battery.sum(axis=0),  # Battery charge
                                       PV2Battery.sum(
                                           axis=0) * Household.Battery.DischargeEfficiency * Household.Battery.ChargeEfficiency,
                                       # Battery discharge
                                       Battery2Load.sum(axis=0),  # Battery 2 Load

                                       Total_Load.sum(axis=0),  # Yearly total load
                                       (np.tile(PV_profile,
                                                (elec_grid_demand.shape[1], 1)).T - Electricity_surplus).sum(axis=0),
                                       # PV self use
                                       (np.tile(PV_profile,
                                                (elec_grid_demand.shape[1], 1)).T - Electricity_surplus).sum(
                                           axis=0) / PV_profile.sum(),
                                       # PV self consumption rate
                                       (np.tile(PV_profile,
                                                (elec_grid_demand.shape[1], 1)).T - Electricity_surplus).sum(
                                           axis=0) / Total_Load.sum(axis=0),  # PV self Sufficiency rate

                                       np.full((elec_grid_demand.shape[1]), Household.Building.hwb_norm1),
                                       # Household.ApplianceGroup.DishWasherShifting,
                                       # Household.ApplianceGroup.WashingMachineShifting,
                                       # Household.ApplianceGroup.DryerShifting,
                                       np.full((elec_grid_demand.shape[1]), Household.SpaceHeating.TankSize),
                                       np.full((elec_grid_demand.shape[1]), Household.SpaceCooling.AdoptionStatus),
                                       np.full((elec_grid_demand.shape[1]), Household.PV.PVPower),
                                       np.full((elec_grid_demand.shape[1]), Household.Battery.Capacity * 1_000),
                                       np.full((elec_grid_demand.shape[1]), Household.ID_AgeGroup),
                                       np.full((elec_grid_demand.shape[1]),
                                               Household.SpaceHeating.Name_SpaceHeatingPumpType),
                                       np.full((elec_grid_demand.shape[1]), Environment["ID_ElectricityPriceType"])
                                       ])

        # hourly values:
        HourlyFrame = np.column_stack([
            np.full((len(elec_grid_demand)*elec_grid_demand.shape[1],), Household.ID),  # household ID
            np.repeat(np.arange(1, elec_grid_demand.shape[1] + 1), len(elec_grid_demand)),  # ID Building
            np.full((len(elec_grid_demand)*elec_grid_demand.shape[1],), Environment.ID),  # environment ID
            np.tile(np.arange(1, len(elec_grid_demand) + 1), (elec_grid_demand.shape[1],)),  # Hours ID
            np.tile(PriceHourly, (elec_grid_demand.shape[1],)),  # elec price
            np.tile(FIT, (elec_grid_demand.shape[1],)),  # Feed in tarif
            np.tile(np.full((len(elec_grid_demand),), Household.ID_AgeGroup), (elec_grid_demand.shape[1],)),  # Age Group
            np.tile(np.full((len(elec_grid_demand),), Household.SpaceHeating.Name_SpaceHeatingPumpType),
                    (elec_grid_demand.shape[1],)),  # space heating type
            np.tile(self.outside_temperature.to_numpy(), (elec_grid_demand.shape[1],)),  # outside temperature

            np.tile(self.BaseLoadProfile, (elec_grid_demand.shape[1],)),  # base load
            # DishWasherProfile,   # E dish washer
            # WashingMachineProfile,  # E washing machine
            # DryerProfile,  # E dryer
            # Load_Appliances,  # E all 3 appliances

            Q_Heating_noDR.flatten(),  # Q Heat pump (tank heating)
            np.tile(COP_SpaceHeating, (elec_grid_demand.shape[1],)),  # Hourly COP of HP
            HeatingProfile.flatten(),  # E of HP
            Q_Heating_noDR.flatten() - HeatingProfile.flatten(),
            # ambient energy
            HeatingElement.flatten(),  # E of heating element
            Q_Heating_noDR.flatten(),  # room heating
            # room heating is same as tank heating

            T_Room_noDR.flatten(),  # room temperature
            Tm_t_noDR.flatten(),  # thermal mass temperature
            Q_solar.flatten(),  # solar gains
            Q_Cooling_noDR.flatten(),  # Q cooling
            CoolingProfile.flatten(),  # E cooling

            np.tile(Q_HotWater, (elec_grid_demand.shape[1],)),  # Q hotwater
            np.tile(DHWProfile, (elec_grid_demand.shape[1],)),  # E hotwater
            elec_grid_demand.flatten(),
            elec_grid_demand.flatten(),
            # Grid 2 Load is the same as Grid because grid is only used for load
            np.tile(np.zeros(len(Total_Load, )), (elec_grid_demand.shape[1],)),
            # grid 2 bat = 0 because the battery is only charged by PV

            np.tile(PV_profile, (elec_grid_demand.shape[1],)),
            np.tile(PV_profile, (elec_grid_demand.shape[1],)) - Electricity_surplus.flatten(),  # PV 2 Load
            PV2Battery.flatten(),  # PV 2 Battery
            Electricity_surplus.flatten(),  # PV2Grid

            PV2Battery.flatten(),  # Battery charge
            PV2Battery.flatten() * Household.Battery.DischargeEfficiency *
            Household.Battery.ChargeEfficiency,  # Battery discharge,
            Battery2Load.flatten(),  # Battery 2 Load,
            BatterySOC.flatten(),  # Battery SOC

            Total_Load.flatten()  # kW
        ])
        return YearlyFrame, HourlyFrame

    def save_to_database(self, yearly_frame, hourly_frame):

        DB().add_DataFrame(table=hourly_frame,
                           table_name="Res_Reference_HeatingCooling_new",
                           column_names=DataCollector(self.Conn).SystemOperationHour_Column.keys(),
                           conn=self.Conn,
                           dtype=DataCollector(self.Conn).SystemOperationHour_Column)
        # Year:
        DB().add_DataFrame(table=yearly_frame,
                           table_name="Res_Reference_HeatingCooling_Year_new",
                           column_names=DataCollector(self.Conn).SystemOperationYear_Column.keys(),
                           conn=self.Conn,
                           dtype=DataCollector(self.Conn).SystemOperationYear_Column)

    # @performance_counter
    def run(self):
        small_household_frame = self.remove_household_IDs()
        runs = len(small_household_frame)
        for total_iterations, household_RowID in enumerate(range(0, runs)):
            for environment_RowID in range(0, 2):
                yearly_results, hourly_results = self.calculate_noDR(household_RowID, environment_RowID)
                self.save_to_database(yearly_results, hourly_results)
            print(f"configuration {total_iterations} of {runs} done")


if __name__ == "__main__":
    no_SEMS().run()
    print("finished")
