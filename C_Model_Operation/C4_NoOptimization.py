from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A1_CONS import CONS
from B_Classes.B2_Building import HeatingCooling_noDR
import pandas as pd
import numpy as np




class no_DR:

    def __init__(self, conn=DB().create_Connection(CONS().RootDB)):
        self.Conn = conn

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.outsideTemperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature
        self.Buildings = DB().read_DataFrame(REG_Table().ID_BuildingOption, self.Conn)
        self.IndoorSetTemperatures = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)

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



    def calculate_noDR(self):
        # TODO specify how to use different set temperatures for different scenarios!
        T_indoorSetMin = self.IndoorSetTemperatures.HeatingTargetTemperatureYoung.to_numpy()
        T_indoorSetMax = self.IndoorSetTemperatures.CoolingTargetTemperatureYoung.to_numpy()

        # calculate the heating anf cooling energy for all buildings in IDBuildingOption:
        initial_thermal_mass_temperature = self.calculate_initial_thermal_mass_temp()
        Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t_noDR = HeatingCooling_noDR().ref_HeatingCooling(
                                                                                       initial_thermal_mass_temp=initial_thermal_mass_temperature,
                                                                                       T_air_min=T_indoorSetMin,
                                                                                       T_air_max=T_indoorSetMax)




        a=1




if __name__=="__main__":
    no_DR().calculate_noDR()