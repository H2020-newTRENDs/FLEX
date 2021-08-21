from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table, REG_Var
from A_Infrastructure.A1_CONS import CONS
from B_Classes.B2_Building import HeatingCooling_noDR
from B_Classes.B1_Household import Household
from C_Model_Operation.C2_DataCollector import DataCollector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class no_DR:

    def __init__(self, conn=DB().create_Connection(CONS().RootDB)):
        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.Conn)

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile,
                                                   self.Conn).BaseElectricityProfile.to_numpy()
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.outsideTemperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature
        self.Buildings = DB().read_DataFrame(REG_Table().ID_BuildingOption, self.Conn)
        self.IndoorSetTemperatures = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG_Table().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)
        self.Battery = DB().read_DataFrame(REG_Table().ID_BatteryType, self.Conn)
        self.HotWaterTank = DB().read_DataFrame(REG_Table().ID_SpaceHeatingTankType, self.Conn)
        self.AppliancePower = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_ApplianceGroup, self.Conn)
        self.HeatPump_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)

        self.CPWater = 4200 / 3600

        self.HourlyFrame = []
        self.YearlyFrame = []

    def gen_Household(self, row_id):
        ParaSeries = self.ID_Household.iloc[row_id]
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
        Buildings = HeatingCooling_noDR()
        # start time steps to calculate constant values
        runtime = 100
        # start with 100 hours,  check if Tm_t is constant:
        Temperature_outside = np.array(
            [self.outsideTemperature.to_numpy()[0]] * runtime)  # take first index of temperature vector
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
                Temperature_outside = np.array([self.outsideTemperature.to_numpy()[0]] * runtime)
                Q_sol = np.array([[0] * len(self.Buildings)] * runtime)
                Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_constant = Buildings.ref_HeatingCooling(
                    Temperature_outside,
                    Q_solar=Q_sol,
                    initial_thermal_mass_temp=22,
                    T_air_min=20,  # indoor set temperature is always at 20 for all scenarios, has marginal impact
                    T_air_max=27)  # has no influence cause its winter -- > only heating
        return T_thermalMass_constant[-1, :]

    def calculate_tank_energy(self, Total_Load_minusPV, Q_Heating_noDR, PV_profile_surplus, Household):
        """
        Calculates the energy/temperature inside the hot water tank and the heating energy that has to be actually used
        when the tank energy is always used for heating when necessary. Input parameters are all provided in kW
        (except COP which has no unit). Return values are also provided in kW.
        """
        Total_Load_minusPV = Total_Load_minusPV * 1_000  # W
        PV_profile_surplus = PV_profile_surplus * 1_000  # W
        TankSize = float(Household.SpaceHeating.TankSize)
        TankMinTemperature = float(Household.SpaceHeating.TankMinimalTemperature)
        TankMaxTemperature = float(Household.SpaceHeating.TankMaximalTemperature)
        TankSurfaceArea = float(Household.SpaceHeating.TankSurfaceArea)
        TankLoss = float(Household.SpaceHeating.TankLoss)
        TankSurroundingTemperature = float(Household.SpaceHeating.TankSurroundingTemperature)
        TankStartTemperature = float(Household.SpaceHeating.TankStartTemperature)
        COP_SpaceHeating = self.HeatPump_HourlyCOP.loc[
            self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] ==
            Household.SpaceHeating.ID_SpaceHeatingBoilerType]['SpaceHeatingHourlyCOP'].to_numpy()

        # Assumption: Tank is always kept at minimum temperature except when it is charged with surplus energy:
        # Note: I completely neglect Hot Water here (but its also neglected in the optimization)
        TankLoss_hourly = np.zeros(PV_profile_surplus.shape)
        CurrentTankTemperature = np.zeros(PV_profile_surplus.shape)
        Electricity_surplus = np.zeros(PV_profile_surplus.shape)
        Electricity_deficit = np.zeros(PV_profile_surplus.shape)
        Q_heating_HP = np.copy(Q_Heating_noDR) * 1_000  # W
        Total_Load_WaterTank = np.copy(Total_Load_minusPV)  # W
        for index, row in enumerate(TankLoss_hourly):
            for column, element in enumerate(row):
                if index == 0:
                    TankLoss_hourly[index, column] = (
                                                             TankStartTemperature - TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
                    # surplus of PV electricity is used to charge the tank:
                    CurrentTankTemperature[index, column] = TankStartTemperature + \
                                                            (PV_profile_surplus[index, column] *
                                                             COP_SpaceHeating[index] - TankLoss_hourly[index, column]) / \
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
                                                            Total_Load_minusPV[index, column]) * COP_SpaceHeating[index]
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
        Total_Load_WaterTank = Total_Load_WaterTank / 1_000
        return TankLoss_hourly, CurrentTankTemperature, Electricity_surplus, Electricity_deficit, Q_heating_HP, Total_Load_WaterTank  # all energies in kW, temperature in °C

    def calculate_battery_energy(self, Total_Load_minusPV, PV_profile_surplus, Household):
        """

        Assumption: The battery is not discharging itself.
        """
        Capacity = float(Household.Battery.Capacity)  # kWh
        MaxChargePower = float(Household.Battery.MaxChargePower)  # kW
        MaxDischargePower = float(Household.Battery.MaxDischargePower)  # kW
        ChargeEfficiency = float(Household.Battery.ChargeEfficiency)
        DischargeEfficiency = float(Household.Battery.DischargeEfficiency)

        # Battery is not charged at the beginning of the simulation
        BatterySOC = np.zeros(PV_profile_surplus.shape)
        Electricity_surplus = np.zeros(PV_profile_surplus.shape)
        Total_load_battery = np.copy(Total_Load_minusPV)

        for index, row in enumerate(BatterySOC):
            for column, element in enumerate(row):
                if index == 0:  # there will be no charging at 01:00 clock in the morning of the 1st january:
                    BatterySOC[index, column] = 0

                if index > 0:
                    # if there is surplus energy, the battery will be charged
                    if PV_profile_surplus[index, column] > 0:
                        # check if the battery can store the power or if its already fully charged:
                        if BatterySOC[index - 1, column] < Capacity:  # there is space to charge

                            if PV_profile_surplus[
                                index, column] <= MaxChargePower:  # maximum charging power is not exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index - 1, column]
                                BatterySOC[index, column] = BatterySOC[index - 1, column] + PV_profile_surplus[
                                    index, column] * ChargeEfficiency
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    Electricity_surplus[index, column] = PV_profile_surplus[
                                                                             index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                            if PV_profile_surplus[index, column] > MaxChargePower:  # maximum charging power is exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index - 1, column]
                                BatterySOC[index, column] = BatterySOC[
                                                                index - 1, column] + MaxChargePower * ChargeEfficiency
                                Electricity_surplus[index, column] = PV_profile_surplus[index, column] - MaxChargePower
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    Electricity_surplus[index, column] = PV_profile_surplus[
                                                                             index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                        # if battery can not be charged because its full:
                        if BatterySOC[index - 1, column] == Capacity:
                            BatterySOC[index, column] = Capacity
                            Electricity_surplus[index, column] = PV_profile_surplus[index, column]

                    # if there is no surplus of energy and the battery has energy stored, it will provide energy:
                    if PV_profile_surplus[index, column] == 0:
                        # check if battery has power:
                        if BatterySOC[index - 1, column] > 0:
                            # if the power in battery is enough to cover whole electricity demand:
                            if BatterySOC[index - 1, column] > Total_Load_minusPV[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = 0
                                BatterySOC[index, column] = BatterySOC[index - 1, column] - Total_Load_minusPV[
                                    index, column] / DischargeEfficiency
                                # check if maximum discharge power is exceeded:
                                if Total_load_battery[index, column] / DischargeEfficiency > MaxDischargePower:
                                    Total_load_battery[index, column] = Total_Load_minusPV[
                                                                            index, column] - MaxDischargePower
                                    BatterySOC[index, column] = BatterySOC[
                                                                    index - 1, column] - MaxDischargePower / DischargeEfficiency

                            # if the power in the battery is not enough to cover the whole electricity demand:
                            if BatterySOC[index - 1, column] <= Total_Load_minusPV[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = Total_Load_minusPV[index, column] - BatterySOC[
                                    index - 1, column] * DischargeEfficiency
                                BatterySOC[index, 1] = 0
                                # check if max discharge power is exceeded:
                                if BatterySOC[index - 1, column] > MaxDischargePower:
                                    BatterySOC[index, column] = BatterySOC[
                                                                    index - 1, column] - MaxDischargePower / DischargeEfficiency
                                    Total_load_battery[index, column] = Total_Load_minusPV[
                                                                            index, column] - MaxDischargePower

        return Total_load_battery, Electricity_surplus, BatterySOC  # kW and kWh

    def calculate_noDR(self, household_RowID, environment_RowID):
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
        T_indoorSetMin = self.IndoorSetTemperatures.HeatingTargetTemperatureYoung.to_numpy()
        T_indoorSetMax = self.IndoorSetTemperatures.CoolingTargetTemperatureYoung.to_numpy()

        # calculate solar gains from database
        # solar gains from different celestial directions
        radiation = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections,
                                        self.Conn)  # TODO careful when there are more profiles
        Q_sol_north = np.outer(radiation.north.to_numpy(), HeatingCooling_noDR().AreaWindowSouth)
        Q_sol_east = np.outer(radiation.east.to_numpy(), HeatingCooling_noDR().AreaWindowEastWest / 2)
        Q_sol_south = np.outer(radiation.south.to_numpy(), HeatingCooling_noDR().AreaWindowSouth)
        Q_sol_west = np.outer(radiation.west.to_numpy(), HeatingCooling_noDR().AreaWindowEastWest / 2)

        Q_solar = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())

        # calculate the heating anf cooling energy for all buildings in IDBuildingOption:
        initial_thermal_mass_temperature = self.calculate_initial_thermal_mass_temp()
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t_noDR = HeatingCooling_noDR().ref_HeatingCooling(
            initial_thermal_mass_temp=initial_thermal_mass_temperature,
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
        PV_production = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        PV_profile = pd.to_numeric(PV_production.loc[PV_production["ID_PVType"] == Household.PV.ID_PVType, :].PVPower).to_numpy()

        # Dishwasher Profile
        # TODO select one of the random profiles (now i only use number 1)
        RandomProfileNumber = 0
        # Dish Washer Profile
        DishWasherStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherStartingHours, self.Conn)[
            "DishWasherHours " + str(RandomProfileNumber)].to_numpy()
        DishWasherPower = float(self.AppliancePower.DishWasherPower)
        DishWasherProfile = DishWasherStartingHours * DishWasherPower

        # Washing Machine Profile
        WashingMachineStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineStartingHours, self.Conn)[
            "WashingMachineHours " + str(RandomProfileNumber)].to_numpy()
        WashingMachinePower = float(self.AppliancePower.WashingMachinePower)
        WashingMachineProfile = WashingMachineStartingHours * WashingMachinePower

        # Dryer Profile
        DryerStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DryerStartingHours, self.Conn)[
            "DryerHours " + str(RandomProfileNumber)].to_numpy()
        DryerPower = float(self.AppliancePower.DryerPower)
        DryerProfile = DryerStartingHours * DryerPower

        # electricity for DHW:
        COP_SpaceHeating = self.HeatPump_HourlyCOP.loc[
            self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] ==
            Household.SpaceHeating.ID_SpaceHeatingBoilerType]['SpaceHeatingHourlyCOP'].to_numpy()

        COP_HotWater = self.HeatPump_HourlyCOP.loc[
            self.HeatPump_HourlyCOP['ID_SpaceHeatingBoilerType'] ==
            Household.SpaceHeating.ID_SpaceHeatingBoilerType].HotWaterHourlyCOP.to_numpy()

        Q_HotWater = self.HotWaterProfile.HotWaterPart1.to_numpy() + self.HotWaterProfile.HotWaterPart2.to_numpy()

        DHWProfile = self.HotWaterProfile.HotWaterPart1.to_numpy() / COP_SpaceHeating + \
                     self.HotWaterProfile.HotWaterPart2.to_numpy() / COP_HotWater

        # electricity for heating:
        HeatingProfile = np.zeros(Q_Heating_noDR.shape)
        for column in range(Q_Heating_noDR.shape[1]):
            HeatingProfile[:, column] = Q_Heating_noDR[:, column] / COP_SpaceHeating

        # electricity for cooling:
        AC_hourly_COP = DB().read_DataFrame(REG_Table().Gen_Sce_AC_HourlyCOP, self.Conn).ACHourlyCOP.to_numpy()
        CoolingProfile = np.zeros(Q_Cooling_noDR.shape)
        for column in range(Q_Cooling_noDR.shape[1]):
            CoolingProfile[:, column] = Q_Cooling_noDR[:, column] / AC_hourly_COP

        # add up all electric loads but without Heating:
        Total_Load = np.zeros(Q_Heating_noDR.shape)
        for column in range(Q_Heating_noDR.shape[1]):
            Total_Load[:, column] = self.BaseLoadProfile + \
                                    DHWProfile + \
                                    DryerProfile + \
                                    WashingMachineProfile + \
                                    DishWasherProfile + \
                                    HeatingProfile[:, column] + \
                                    CoolingProfile[:, column] + \
                                    HeatingElement[:, column]

        # only load of electric appliances:
        Load_Appliances = DryerProfile + WashingMachineProfile + DishWasherProfile

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
        final_Load = np.copy(Total_Load_minusPV)
        # electricity surplus is the energy that is sold back to the grid because it can not be used:
        Electricity_surplus = np.copy(PV_profile_surplus)  # kW

        # Battery storage:
        # if battery is used:
        if float(Household.Battery.Capacity) > 0:
            Total_load_battery, Electricity_surplus_Battery, BatterySOC = \
                self.calculate_battery_energy(final_Load, Electricity_surplus, Household)
            final_Load = Total_load_battery
            # amount of electricity from PV to Battery:
            PV2Battery = Electricity_surplus - Electricity_surplus_Battery
            Electricity_surplus = Electricity_surplus_Battery
        else:
            PV2Battery = np.zeros(Electricity_surplus.shape)

        # When there is PV surplus energy it either goes to a storage or is sold to the grid:
        # Water Tank as storage:
        if float(Household.SpaceHeating.TankSize) > 0:
            TankLoss_hourly, CurrentTankTemperature, Electricity_surplus, Electricity_deficit, Q_heating_withTank, Total_Load_WaterTank = \
                self.calculate_tank_energy(final_Load, Q_Heating_noDR, Electricity_surplus, Household)
            # remaining surplus of electricity
            Electricity_surplus = Electricity_surplus
            final_Load = Total_Load_WaterTank

        # calculate the electricity cost:
        PriceHourly = self.ElectricityPrice.loc[
            (self.ElectricityPrice['ID_ElectricityPriceType'] == Environment["ID_ElectricityPriceType"]) &
            (self.ElectricityPrice['ID_Country'] == Household.ID_Country)]['HourlyElectricityPrice'].to_numpy()

        FIT = self.FeedinTariff.loc[(self.FeedinTariff['ID_FeedinTariffType'] == Environment["ID_FeedinTariffType"]) &
                                    (self.ElectricityPrice['ID_Country'] == Household.ID_Country)][
            'HourlyFeedinTariff'].to_numpy()

        total_elec_cost = np.zeros(final_Load.shape)
        for column in range(final_Load.shape[1]):
            total_elec_cost[:, column] = PriceHourly * final_Load[:, column] - Electricity_surplus[:, column] * FIT

        # Dataframe for yearly values:
        self.YearlyFrame.append([Household.ID,
                                 Environment.ID,
                                 total_elec_cost.sum(axis=0)[Household.ID - 1],

                                 self.BaseLoadProfile.sum(),  # Base electricity load
                                 Load_Appliances.sum(axis=0),
                                 Q_Heating_noDR.sum(axis=0)[Household.ID - 1],
                                 # Note that Roomheating and Tankheating are the same
                                 HeatingProfile.sum(axis=0)[Household.ID - 1],  # electricity for heating
                                 Q_Heating_noDR.sum(axis=0)[Household.ID - 1] / HeatingProfile.sum(axis=0)[
                                     Household.ID - 1],  # heatpump Performance factor
                                 Q_Heating_noDR.sum(axis=0)[Household.ID - 1] - HeatingProfile.sum(axis=0)[
                                     Household.ID - 1],  # ambient heat
                                 HeatingElement.sum(axis=0)[Household.ID - 1],  # heating element
                                 Q_Heating_noDR.sum(axis=0)[Household.ID - 1],
                                 # Note that Roomheating and Tankheating are the same

                                 Q_Cooling_noDR.sum(axis=0)[Household.ID - 1],  # cooling energy
                                 CoolingProfile.sum(axis=0)[Household.ID - 1],  # cooling electricity

                                 Q_HotWater.sum(axis=0),  # hot water energy
                                 DHWProfile.sum(axis=0),  # hot water electricity

                                 final_Load.sum(axis=0)[Household.ID - 1],
                                 final_Load.sum(axis=0)[Household.ID - 1],
                                 0,

                                 PV_profile.sum(),
                                 PV_profile.sum() - Electricity_surplus.sum(axis=0)[Household.ID - 1],  # PV2Load
                                 PV2Battery.sum(axis=0)[Household.ID - 1],  # PV2Bat
                                 Electricity_surplus.sum(axis=0)[Household.ID - 1],  # PV2Grid

                                 PV2Battery.sum(axis=0)[Household.ID - 1],  # Battery charge
                                 PV2Battery.sum(axis=0)[Household.ID - 1] *
                                 Household.Battery.DischargeEfficiency *
                                 Household.Battery.ChargeEfficiency,  # Battery discharge
                                 PV2Battery.sum(axis=0)[Household.ID - 1] *
                                 Household.Battery.DischargeEfficiency *
                                 Household.Battery.ChargeEfficiency,  # Battery 2 Load

                                 Total_Load.sum(axis=0)[Household.ID - 1],  # Yearly total load
                                 (PV_profile - PV_profile_surplus[:, Household.ID - 1]).sum(),  # PV self use
                                 (PV_profile - PV_profile_surplus[:, Household.ID - 1]).sum() / PV_profile.sum(),
                                 # PV self consumption rate
                                 (PV_profile - PV_profile_surplus[:, Household.ID - 1]).sum() / Total_Load.sum(axis=0)[
                                     Household.ID - 1],  # PV self Sufficiency rate

                                 Household.Building.hwb_norm1,
                                 Household.ApplianceGroup.DishWasherShifting,
                                 Household.ApplianceGroup.WashingMachineShifting,
                                 Household.ApplianceGroup.DryerShifting,
                                 Household.SpaceHeating.TankSize,
                                 Household.SpaceCooling.AdoptionStatus,
                                 Household.PV.PVPower,
                                 Household.Battery.Capacity,
                                 Environment["ID_ElectricityPriceType"]
                                 ])

        # hourly values:
        self.HourlyFrame.append(np.column_stack([np.full((len(Total_Load), ), Household.ID),  # household ID
                                                  np.full((len(Total_Load), ), Environment.ID),  # environment ID
                                                  np.arange(1, len(Total_Load) + 1),   # Hours ID
                                                  PriceHourly,  # elec price
                                                  FIT,  # Feed in tarif
                                                  self.outsideTemperature.to_numpy(),  # outside temperature
                                                  # change when more temperature profiles are being saved!

                                                  self.BaseLoadProfile,   # base load
                                                  DishWasherProfile,   # E dish washer
                                                  WashingMachineProfile,  # E washing machine
                                                  DryerProfile,  # E dryer
                                                  Load_Appliances,  # E all 3 appliances

                                                  Q_Heating_noDR[:, Household.ID - 1],   # Q Heat pump (tank heating)
                                                  COP_SpaceHeating,  # Hourly COP of HP
                                                  HeatingProfile[:, Household.ID - 1],  # E of HP
                                                  Q_Heating_noDR[:, Household.ID - 1] - HeatingProfile[:, Household.ID - 1],  # ambient energy
                                                  HeatingElement[:, Household.ID - 1],  # E of heating element
                                                  Q_Heating_noDR[:, Household.ID - 1],  # room heating
                                                  # room heating is same as tank heating

                                                  T_Room_noDR[:, Household.ID - 1],  # room temperature
                                                  Tm_t_noDR[:, Household.ID - 1],  # thermal mass temperature
                                                  Q_solar[:, Household.ID - 1],  # solar gains

                                                  Q_Cooling_noDR[:, Household.ID - 1],  # Q cooling
                                                  CoolingProfile[:, Household.ID - 1],  # E cooling

                                                  Q_HotWater,  # Q hotwater
                                                  DHWProfile,  # E hotwater

                                                  final_Load[:, Household.ID - 1],
                                                  final_Load[:, Household.ID - 1], # Grid 2 Load is the same as Grid because grid is only used for load
                                                  np.zeros(len(Total_Load, )), # grid 2 bat = 0 because the battery is only charged by PV

                                                  PV_profile,
                                                  PV_profile - Electricity_surplus[:, Household.ID - 1],  # PV 2 Load
                                                  PV2Battery[:, Household.ID - 1],  # PV 2 Battery
                                                  Electricity_surplus[:, Household.ID - 1],  # PV2Grid

                                                  PV2Battery[:, Household.ID - 1],  # Battery charge
                                                  PV2Battery[:, Household.ID - 1] *
                                                  Household.Battery.DischargeEfficiency *
                                                  Household.Battery.ChargeEfficiency,  # Battery discharge,
                                                  PV2Battery[:, Household.ID - 1] *
                                                  Household.Battery.DischargeEfficiency *
                                                  Household.Battery.ChargeEfficiency,  # Battery 2 Load,
                                                  BatterySOC[:, Household.ID - 1],  # Battery SOC

                                                  Total_Load[:, Household.ID - 1]
                                                  ])
                                 )

    def save_ReferenzeResults(self):
        DB().write_DataFrame(np.vstack(self.YearlyFrame), REG_Table().Res_Reference_HeatingCooling_Year,
                             DataCollector(self.Conn).SystemOperationYear_Column.keys(), self.Conn,
                             dtype=DataCollector(self.Conn).SystemOperationYear_Column)

        DB().write_DataFrame(np.vstack(self.HourlyFrame), REG_Table().Res_Reference_HeatingCooling,
                             DataCollector(self.Conn).SystemOperationHour_Column.keys(), self.Conn,
                             dtype=DataCollector(self.Conn).SystemOperationHour_Column)




    # plotanfang = 4000
    # plotende = 4050
    # # check results
    # x_achse = np.arange(plotanfang, plotende)

    # fig = plt.figure()
    # ax1 = plt.gca()
    # ax1.plot(x_achse, Q_Heating_noDR[plotanfang:plotende, 1], label="Q_heating no Tank", alpha=0.8)
    # ax1.plot(x_achse, Q_heating_withTank[plotanfang:plotende, 1], label="Q_heating with Tank", alpha=0.8)
    #
    # ax2 = ax1.twinx()
    # ax1.plot(x_achse, PV_profile_surplus[plotanfang:plotende, 1], label="PV surplus", alpha=0.4, color="yellow")
    #
    # ax2.plot(x_achse, CurrentTankTemperature[plotanfang:plotende, 1], label="tank Temp", color="red", alpha=0.2)
    # ax1.legend()
    # plt.show()

    # fig = plt.figure()
    # ax1 = plt.gca()
    #
    # ax1.plot(x_achse, PV_profile[plotanfang:plotende], color="yellow", label="pv profile")
    # ax1.plot(x_achse, Total_Load[plotanfang:plotende, 1], color="black", linewidth=0.2, label="total load")
    # ax1.plot(x_achse, PV_profile_surplus[plotanfang:plotende, 1] / 1000, label="PV surplus", alpha=0.8,
    #          color="orange")
    # ax1.plot(x_achse, Total_Load_minusPV[plotanfang:plotende, 1], label="total load - PV", alpha=0.7, color="red")
    #
    # ax1.legend()
    # plt.show()
    #
    # fig = plt.figure()
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # ax1.plot(x_achse, Total_Load_minusPV[plotanfang:plotende, 1], label="total load - PV", alpha=0.7, color="red",
    #          linewidth=0.2)
    # plt.plot(x_achse, PV_profile[plotanfang:plotende], color="yellow", label="pv profile", alpha=0.8, linewidth=0.7)
    # ax1.plot(x_achse, Total_Load[plotanfang:plotende, 1], color="black", linewidth=0.2, label="total load")
    # ax1.plot(x_achse, Total_load_battery[plotanfang:plotende, 1], color="blue", linewidth=0.3,
    #          label="load with battery")
    #
    # ax2.plot(x_achse, BatterySOC[plotanfang:plotende, 1], color="green", linewidth=0.3, alpha=0.7)
    #
    # ax1.legend()
    # plt.show()


    def run(self):
        for household_RowID in range(0, 3):
            for environment_RowID in range(0, 1):
                self.calculate_noDR(household_RowID, environment_RowID)
        self.save_ReferenzeResults()


if __name__ == "__main__":
    no_DR().run()
