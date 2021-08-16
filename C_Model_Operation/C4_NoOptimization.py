from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A1_CONS import CONS
from B_Classes.B2_Building import HeatingCooling_noDR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




class no_DR:

    def __init__(self, conn=DB().create_Connection(CONS().RootDB)):
        self.Conn = conn

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile, self.Conn).BaseElectricityProfile.to_numpy()
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.outsideTemperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature
        self.Buildings = DB().read_DataFrame(REG_Table().ID_BuildingOption, self.Conn)
        self.IndoorSetTemperatures = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)
        self.ElectricityPrice = DB().read_DataFrame(REG_Table().Sce_Price_HourlyElectricityPrice, self.Conn)
        self.FeedinTariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)
        self.Battery = DB().read_DataFrame(REG_Table().ID_BatteryType, self.Conn)
        self.HotWaterTank = DB().read_DataFrame(REG_Table().ID_SpaceHeatingTankType, self.Conn)
        self.AppliancePower = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_ApplianceGroup, self.Conn)
        self.HeatPumpCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)

        self.CPWater = 4200 / 3600


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
        Temperature_outside = np.array([self.outsideTemperature.to_numpy()[0]] * runtime)  # take first index of temperature vector
        Q_sol = np.array([[0] * len(self.Buildings)] * runtime)
        # initial thermal mass temperature should be between T_min_indoor and T_max_indoor (22°C should always be ok)
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_constant = Buildings.ref_HeatingCooling(
            Temperature_outside,
            Q_solar=Q_sol,
            initial_thermal_mass_temp=22,
            T_air_min=20,    # indoor set temperature is always at 20 for all scenarios, has marginal impact
            T_air_max=27)   # has no influence cause its winter -- > only heating
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
                    T_air_min=20,   # indoor set temperature is always at 20 for all scenarios, has marginal impact
                    T_air_max=27)  # has no influence cause its winter -- > only heating
        return T_thermalMass_constant[-1, :]


    def calculate_tank_energy(self, Total_Load_minusPV, Q_Heating_noDR, PV_profile_surplus, COP_SpaceHeating):
        """
        Calculates the energy/temperature inside the hot water tank and the heating energy that has to be actually used
        when the tank energy is always used for heating when necessary. Input parameters are all provided in kW
        (except COP which has no unit). Return values are also provided in kW.
        """
        Total_Load_minusPV = Total_Load_minusPV * 1_000  # W
        PV_profile_surplus = PV_profile_surplus * 1_000  # W
        ID_SpaceHeatingTankType = 1  # TODO choose TankType automatically
        TankSize = float(
            self.HotWaterTank[self.HotWaterTank["ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankSize)
        TankMinTemperature = float(self.HotWaterTank[self.HotWaterTank[
                                                         "ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankMinimalTemperature)
        TankMaxTemperature = float(self.HotWaterTank[self.HotWaterTank[
                                                         "ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankMaximalTemperature)
        TankSurfaceArea = float(
            self.HotWaterTank[self.HotWaterTank["ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankSurfaceArea)
        TankLoss = float(
            self.HotWaterTank[self.HotWaterTank["ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankLoss)
        TankSurroundingTemperature = float(self.HotWaterTank[self.HotWaterTank[
                                                                 "ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankSurroundingTemperature)
        TankStartTemperature = float(self.HotWaterTank[self.HotWaterTank[
                                                           "ID_SpaceHeatingTankType"] == ID_SpaceHeatingTankType].TankStartTemperature)

        # Assumption: Tank is always kept at minimum temperature except when it is charged with surplus energy:
        # Note: I completely neglect Hot Water here (but its also neglected in the optimization)
        TankLoss_hourly = np.zeros(PV_profile_surplus.shape)
        CurrentTankTemperature = np.zeros(PV_profile_surplus.shape)
        Electricity_surplus = np.zeros(PV_profile_surplus.shape)
        Electricity_deficit = np.zeros(PV_profile_surplus.shape)
        Q_heating_HP = np.copy(Q_Heating_noDR) * 1_000  # W
        for index, row in enumerate(TankLoss_hourly):
            for column, element in enumerate(row):
                if index == 0:
                    TankLoss_hourly[index, column] = (TankStartTemperature - TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
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
                                                                     TankSize * self.CPWater) / COP_SpaceHeating[index]  # W
                        CurrentTankTemperature[index, column] = TankMaxTemperature

                    # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                    # and neccesary electricity is calculated
                    if CurrentTankTemperature[index, column] <= TankMinTemperature:
                        Electricity_deficit[index, column] = (TankMinTemperature - CurrentTankTemperature[
                            index, column]) * (TankSize * self.CPWater)
                        CurrentTankTemperature[index, column] = TankMinTemperature

                    # if there is energy in the tank it will be used for heating:
                    if CurrentTankTemperature[index, column] > TankMinTemperature:
                        EnergyInTank = (CurrentTankTemperature[index, column] - TankMinTemperature) * (TankSize * self.CPWater)

                        # if the PV does not cover the whole elctricity in this hour, the tank will cover for heating
                        if Total_Load_minusPV[index, column] > 0:
                            # if the PV does not cover any energy for the heating:
                            if Total_Load_minusPV[index, column] > Q_heating_HP[index, column]/COP_SpaceHeating[index]:  # means that the heating is not covered at all by the PV (it covers other things first)
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
                                    CurrentTankTemperature[index, column] = TankMinTemperature  # the tank temperature drops to minimal energy

                            # if the PV does cover part of the heating energy:
                            if Total_Load_minusPV[index, column] <= Q_heating_HP[index, column]/COP_SpaceHeating[index]:
                                # calculate the part than can be covered by the tank:
                                remaining_heating_Energy = (Q_heating_HP[index, column]/COP_SpaceHeating[index] - Total_Load_minusPV[index, column])*COP_SpaceHeating[index]
                                # if the energy in the tank is enough to cover the remaining heating energy:
                                if EnergyInTank > remaining_heating_Energy:
                                    SurplusEnergy = EnergyInTank - remaining_heating_Energy
                                    Q_heating_HP[index, column] = 0   # Building is heated by tank and PV
                                    CurrentTankTemperature[index, column] = TankMinTemperature + SurplusEnergy / (
                                            TankSize * self.CPWater)
                                # if the energy in the tank is not enough to cover the remaining heating energy:
                                if EnergyInTank <= remaining_heating_Energy:
                                    DeficitEnergy = remaining_heating_Energy - EnergyInTank
                                    Q_heating_HP[index, column] = DeficitEnergy
                                    CurrentTankTemperature[index, column] = TankMinTemperature  # the tank temperature drops to minimal energy


                if index > 0:
                    TankLoss_hourly[index, column] = (CurrentTankTemperature[index - 1, column] - TankSurroundingTemperature) * TankSurfaceArea * TankLoss  # W
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
                                                                     TankSize * self.CPWater) / COP_SpaceHeating[index] # W
                        CurrentTankTemperature[index, column] = TankMaxTemperature

                    # if temperature drops below minimum temperature, temperature is kept at minimum temperature
                    # and neccesary electricity is calculated
                    if CurrentTankTemperature[index, column] <= TankMinTemperature:
                        Electricity_deficit[index, column] = (TankMinTemperature - CurrentTankTemperature[
                            index, column]) * (TankSize * self.CPWater)
                        CurrentTankTemperature[index, column] = TankMinTemperature

                    # if there is energy in the tank it will be used for heating:
                    if CurrentTankTemperature[index, column] > TankMinTemperature:
                        EnergyInTank = (CurrentTankTemperature[index, column] - TankMinTemperature) * (TankSize * self.CPWater)

                        # if the PV does not cover the whole electricity in this hour, the tank will cover for heating
                        if Total_Load_minusPV[index, column] > 0:
                            # if the PV does not cover any energy for the heating:
                            if Total_Load_minusPV[index, column] > Q_heating_HP[index, column]/COP_SpaceHeating[index]:  # means that the heating is not covered at all by the PV (it covers other things first)
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
                                    CurrentTankTemperature[index, column] = TankMinTemperature  # the tank temperature drops to minimal energy

                            # if the PV does cover part of the heating energy:
                            if Total_Load_minusPV[index, column] <= Q_heating_HP[index, column]/COP_SpaceHeating[index]:
                                # calculate the part than can be covered by the tank:
                                remaining_heating_Energy = (Q_heating_HP[index, column]/COP_SpaceHeating[index] - Total_Load_minusPV[index, column]) * COP_SpaceHeating[index]
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
                                    CurrentTankTemperature[index, column] = TankMinTemperature  # the tank temperature drops to minimal energy

        #  Q_heating_HP is now the heating energy with the usage of the tank
        # Electricity_surplus is the electricity that can not be stored in the tank
        # Electricity_deficit is the electricity needed to keep the tank on temperature
        # TankLoss_hourly are the total losses of the tank

        # convert the energie back to kW:
        TankLoss_hourly = TankLoss_hourly / 1_000
        Electricity_surplus = Electricity_surplus / 1_000
        Electricity_deficit = Electricity_deficit / 1_000
        Q_heating_HP = Q_heating_HP / 1_000
        return TankLoss_hourly, CurrentTankTemperature, Electricity_surplus, Electricity_deficit, Q_heating_HP # all energies in kW, temperature in °C


    def calculate_battery_energy(self, Total_Load_minusPV, PV_profile_surplus):
        """

        Assumption: The battery is not discharging itself.
        """
        # TODO define ID battery
        ID_BatteryType = 3
        Capacity = float(self.Battery[self.Battery["ID_BatteryType"] == ID_BatteryType].Capacity)  # kWh
        MaxChargePower = float(self.Battery[self.Battery["ID_BatteryType"] == ID_BatteryType].MaxChargePower)  # kW
        MaxDischargePower = float(self.Battery[self.Battery["ID_BatteryType"] == ID_BatteryType].MaxDischargePower)  # kW
        ChargeEfficiency = float(self.Battery[self.Battery["ID_BatteryType"] == ID_BatteryType].ChargeEfficiency)
        DischargeEfficiency = float(self.Battery[self.Battery["ID_BatteryType"] == ID_BatteryType].DischargeEfficiency)

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
                        if BatterySOC[index-1, column] < Capacity:  # there is space to charge

                            if PV_profile_surplus[index, column] <= MaxChargePower:  # maximum charging power is not exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index-1, column]
                                BatterySOC[index, column] = BatterySOC[index-1, column] + PV_profile_surplus[index, column] * ChargeEfficiency
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    Electricity_surplus[index, column] = PV_profile_surplus[index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                            if PV_profile_surplus[index, column] > MaxChargePower:  # maximum charging power is exceeded
                                # determine how much capacity is available:
                                capacity_left = Capacity - BatterySOC[index-1, column]
                                BatterySOC[index, column] = BatterySOC[index-1, column] + MaxChargePower * ChargeEfficiency
                                Electricity_surplus[index, column] = PV_profile_surplus[index, column] - MaxChargePower
                                # check if the battery exceeds maximum charge limit, if yes:
                                if BatterySOC[index, column] > Capacity:
                                    charging_energy = capacity_left / ChargeEfficiency
                                    Electricity_surplus[index, column] = PV_profile_surplus[index, column] - charging_energy
                                    BatterySOC[index, column] = Capacity

                        # if battery can not be charged because its full:
                        if BatterySOC[index-1, column] == Capacity:
                            BatterySOC[index, column] = Capacity
                            Electricity_surplus[index, column] = PV_profile_surplus[index, column]



                    # if there is no surplus of energy and the battery has energy stored, it will provide energy:
                    if PV_profile_surplus[index, column] == 0:
                        # check if battery has power:
                        if BatterySOC[index-1, column] > 0:
                            # if the power in battery is enough to cover whole electricity demand:
                            if BatterySOC[index-1, column] > Total_Load_minusPV[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = 0
                                BatterySOC[index, column] = BatterySOC[index-1, column] - Total_Load_minusPV[index, column] / DischargeEfficiency
                                # check if maximum discharge power is exceeded:
                                if Total_load_battery[index, column] / DischargeEfficiency > MaxDischargePower:
                                    Total_load_battery[index, column] = Total_Load_minusPV[index, column] - MaxDischargePower
                                    BatterySOC[index, column] = BatterySOC[index-1, column] - MaxDischargePower / DischargeEfficiency

                            # if the power in the battery is not enough to cover the whole electricity demand:
                            if BatterySOC[index-1, column] <= Total_Load_minusPV[index, column] / DischargeEfficiency:
                                Total_load_battery[index, column] = Total_Load_minusPV[index, column] - BatterySOC[index-1, column]*DischargeEfficiency
                                BatterySOC[index, 1] = 0
                                # check if max discharge power is exceeded:
                                if BatterySOC[index-1, column] > MaxDischargePower:
                                    BatterySOC[index, column] = BatterySOC[index-1, column] - MaxDischargePower/DischargeEfficiency
                                    Total_load_battery[index, column] = Total_Load_minusPV[index, column] - MaxDischargePower


        return Total_load_battery, Electricity_surplus, BatterySOC  # kW and kWh








    def calculate_noDR(self):
        """
        Assumption for the Reference scenario: the produced PV power is always used for the immediate electric demand,
        if there is a surplus of PV power, it will be used to charge the Battery,
        if the Battery is full or not available, the PV power will be used to heat the HotWaterTank,
        if the HotWaterTank is not available or full, the PV power will sold to the Grid.
        The surplus of PV energy is never used to Preheat or Precool the building.

        The Washing Machine as well as Dryer and Dishwasher are used during the day on a random time between 06:00
        and 22:00.
        """
        # TODO specify how to use different set temperatures for different scenarios!
        T_indoorSetMin = self.IndoorSetTemperatures.HeatingTargetTemperatureYoung.to_numpy()
        T_indoorSetMax = self.IndoorSetTemperatures.CoolingTargetTemperatureYoung.to_numpy()

        # calculate the heating anf cooling energy for all buildings in IDBuildingOption:
        initial_thermal_mass_temperature = self.calculate_initial_thermal_mass_temp()
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t_noDR = HeatingCooling_noDR().ref_HeatingCooling(
                                                                                       initial_thermal_mass_temp=initial_thermal_mass_temperature,
                                                                                       T_air_min=T_indoorSetMin,
                                                                                       T_air_max=T_indoorSetMax)
        # change values to kW:
        Q_Heating_noDR = Q_Heating_noDR / 1_000
        Q_Cooling_noDR = Q_Cooling_noDR / 1_000


        # PV production profile:
        PV_production = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        ID_PVType = 3  # TODO automatic selection of PV Type
        PV_profile = pd.to_numeric(PV_production.loc[PV_production["ID_PVType"] == ID_PVType, :].PVPower).to_numpy()

        # Dishwasher Profile
        # TODO select one of the random profiles (now i only use number 1)
        RandomProfileNumber = 0
        # Dish Washer Profile
        DishWasherStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DishWasherStartingHours, self.Conn)["DishWasherHours " + str(RandomProfileNumber)].to_numpy()
        DishWasherPower = float(self.AppliancePower.DishWasherPower)
        DishWasherProfile = DishWasherStartingHours * DishWasherPower

        # Washing Machine Profile
        WashingMachineStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_WashingMachineStartingHours, self.Conn)["WashingMachineHours " + str(RandomProfileNumber)].to_numpy()
        WashingMachinePower = float(self.AppliancePower.WashingMachinePower)
        WashingMachineProfile = WashingMachineStartingHours * WashingMachinePower

        # Dryer Profile
        DryerStartingHours = DB().read_DataFrame(REG_Table().Gen_Sce_DryerStartingHours, self.Conn)["DryerHours " + str(RandomProfileNumber)].to_numpy()
        DryerPower = float(self.AppliancePower.DryerPower)
        DryerProfile = DryerStartingHours * DryerPower


        # electricity for DHW:
        # TODO define heat source of HP, 1 is Air, 2 is Water:
        SpaceHeatingBoilerType = 1
        COP_SpaceHeating = self.HeatPumpCOP[self.HeatPumpCOP["ID_SpaceHeatingBoilerType"] == SpaceHeatingBoilerType
                                            ].SpaceHeatingHourlyCOP.to_numpy()
        COP_HotWater = self.HeatPumpCOP[self.HeatPumpCOP["ID_SpaceHeatingBoilerType"] == SpaceHeatingBoilerType
                                        ].HotWaterHourlyCOP.to_numpy()
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
                                    CoolingProfile[:, column]

        # The PV profile is substracted from the total load
        Total_Load_minusPV = np.zeros(Total_Load.shape)
        for column in range(Total_Load_minusPV.shape[1]):
            Total_Load_minusPV[:, column] = Total_Load[:, column] - PV_profile

        # determine the surplus PV power:
        PV_profile_surplus = np.copy(Total_Load_minusPV)
        PV_profile_surplus[PV_profile_surplus > 0] = 0
        PV_profile_surplus = abs(PV_profile_surplus)

        # Total load profile can not be negative: make negative numbers to 0
        Total_Load_minusPV[Total_Load_minusPV < 0] = 0

        # When there is PV surplus energy it either goes to a storage or is sold to the grid:
        # Water Tank as storage:
        # TODO Abfrage if tank is used
        # TankLoss_hourly, CurrentTankTemperature, Electricity_surplus, Electricity_deficit, Q_heating_withTank = \
        #     self.calculate_tank_energy(Total_Load_minusPV, Q_Heating_noDR, PV_profile_surplus, COP_SpaceHeating)
        # # remaining surplus of electricity
        # PV_profile_surplus = Electricity_surplus


        # Battery storage:
        Total_load_battery, Electricity_surplus, BatterySOC = \
            self.calculate_battery_energy(Total_Load_minusPV, PV_profile_surplus)

        PV_profile_surplus = Electricity_surplus



        plotanfang = 4000
        plotende = 4050
        # check results
        x_achse = np.arange(plotanfang, plotende)

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

        fig = plt.figure()
        ax1 = plt.gca()

        ax1.plot(x_achse, PV_profile[plotanfang:plotende], color="yellow", label="pv profile")
        ax1.plot(x_achse, Total_Load[plotanfang:plotende, 1], color="black", linewidth=0.2, label="total load")
        ax1.plot(x_achse, PV_profile_surplus[plotanfang:plotende, 1] / 1000, label="PV surplus", alpha=0.8, color="orange")
        ax1.plot(x_achse, Total_Load_minusPV[plotanfang:plotende, 1], label="total load - PV", alpha=0.7, color="red")

        ax1.legend()
        plt.show()


        fig = plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(x_achse, Total_Load_minusPV[plotanfang:plotende, 1], label="total load - PV", alpha=0.7, color="red", linewidth=0.2)
        plt.plot(x_achse, PV_profile[plotanfang:plotende], color="yellow", label="pv profile", alpha=0.8, linewidth=0.7)
        ax1.plot(x_achse, Total_Load[plotanfang:plotende, 1], color="black", linewidth=0.2, label="total load")
        ax1.plot(x_achse, Total_load_battery[plotanfang:plotende, 1], color="blue", linewidth=0.3, label="load with battery")

        ax2.plot(x_achse, BatterySOC[plotanfang:plotende, 1], color="green", linewidth=0.3, alpha=0.7)

        ax1.legend()
        plt.show()


        a = 1



if __name__=="__main__":
    no_DR().calculate_noDR()