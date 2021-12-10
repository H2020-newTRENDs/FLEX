# -*- coding: utf-8 -*-
__author__ = 'Philipp, Thomas, Songmin'

import numpy as np
from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
from A_Infrastructure.A1_CONS import CONS


class MotherModel:
    def __init__(self):
        self.Conn = DB().create_Connection(CONS().RootDB)
        self.ID_Household = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Household, self.Conn)
        self.ID_Environment = DB().read_DataFrame(REG_Table().Gen_Sce_ID_Environment, self.Conn)
        self.Buildings = DB().read_DataFrame(REG_Table().ID_BuildingOption, self.Conn)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.outside_temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)

        self.electricity_price = DB().read_DataFrame(REG_Table().Gen_Sce_ElectricityProfile, self.Conn)
        self.feed_in_tariff = DB().read_DataFrame(REG_Table().Sce_Price_HourlyFeedinTariff, self.Conn)

        self.BaseLoadProfile = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile,
                                                   self.Conn).BaseElectricityProfile.to_numpy()

        self.PhotovoltaicProfile = DB().read_DataFrame(REG_Table().Gen_Sce_PhotovoltaicProfile, self.Conn)
        self.HotWaterProfile = DB().read_DataFrame(REG_Table().Gen_Sce_HotWaterProfile, self.Conn)
        self.Radiation_SkyDirections = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections,
                                                           self.Conn)
        # self.HeatPump_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_HeatPump_HourlyCOP, self.Conn)
        self.AC_HourlyCOP = DB().read_DataFrame(REG_Table().Gen_Sce_AC_HourlyCOP, self.Conn)

        self.TargetTemperature = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)

        # C_Water
        self.CPWater = 4_200 / 3_600

    def COP_HP(self, outside_temperature, supply_temperature, efficiency, source):
        """
        calculates the COP based on a performance factor and the massflow temperatures
        """
        if source == "Air_HP":
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - temp) for temp in
                            outside_temperature])
        elif source == "Water_HP":
            # for the ground source heat pump the temperature of the source is 10°C
            COP = np.array([efficiency * (supply_temperature + 273.15) / (supply_temperature - 10) for temp in
                            outside_temperature])
        else:
            assert "only >>air<< and >>ground<< are valid arguments"
        return COP
