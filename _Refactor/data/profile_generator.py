import pandas as pd
import numpy as np
import sqlalchemy.types

from _Refactor.basic.db import DB
from _Refactor.data.table_generator import MotherTableGenerator
from _Refactor.basic.reg import Table
import _Refactor.core.household.components as components
import _Refactor.basic.config as config



class ProfileGenerator(MotherTableGenerator):
    def __init__(self):
        super().__init__()
        self.id_day_hour = np.tile(np.arange(1, 25), 365)

    def generate_dishwasher_hours(self,
                                  NumberOfDishWasherProfiles):  # TODO if more than 1 dishwasher type is implemented, rewrite the function
        """
        creates Dish washer days where the dishwasher can be used on a random basis. Hours of the days where
        the Dishwasher has to be used are index = 1 and hours of days where the dishwasher is not used are indexed = 0.
        Dishwasher is not used during the night, only between 06:00 and 22:00.

        Dishwasher starting hours are randomly generated between 06:00 and 22:00 and a seperate dataframe is created
        """
        Demand_DishWasher = DB().read_DataFrame(REG_Table().Sce_Demand_DishWasher, self.conn)
        DishWasherDuration = int(Demand_DishWasher.DishWasherDuration)
        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)

        # Assumption: Dishwasher runs maximum once a day:
        UseDays = int(Demand_DishWasher.DishWasherCycle)
        TotalDays = TimeStructure.ID_Day.to_numpy()[-1]

        # create array with 0 or 1 for every hour a day and random choice of 0 and 1 but with correct numbers of 1
        # (UseDays)
        def rand_bin_array(usedays, totaldays):
            arr = np.zeros(totaldays)
            arr[:usedays] = 1
            np.random.shuffle(arr)
            arr = np.repeat(arr, 24, axis=0)
            return arr

        TargetTable = TimeStructure.ID_Hour.to_numpy()
        TargetTable = np.column_stack([TargetTable, TimeStructure.ID_DayHour.to_numpy()])
        TargetTable_columns = {"ID_Hour": "INTEGER",
                               "ID_DayHour": "INTEGER"}
        for i in range(NumberOfDishWasherProfiles + 1):
            TargetTable = np.column_stack([TargetTable, rand_bin_array(UseDays, TotalDays)])
            TargetTable_columns["DishWasherHours " + str(i)] = "INTEGER"

        # iterate through table and assign random values between 06:00 and 21:00 on UseDays:
        # this table is for reference scenarios or if the dishwasher is not optimized: First a random starting time is
        # specified and from there the timeslots for the duration of the dishwasher are set to 1:
        TargetTable2 = np.copy(TargetTable)
        for index in range(0, len(TargetTable2), 24):
            for column in range(2, TargetTable2.shape[1]):
                if TargetTable2[index, column] == 1:
                    HourOfTheDay = np.random.randint(low=6, high=22) - 1
                    TargetTable2[index + HourOfTheDay:index + HourOfTheDay + DishWasherDuration, column] = 1
                    TargetTable2[index:index + HourOfTheDay, column] = 0
                    TargetTable2[index + HourOfTheDay + DishWasherDuration:index + 24, column] = 0
        # write dataframe with starting hours to database:
        DB().write_DataFrame(TargetTable2, REG_Table().Gen_Sce_DishWasherStartingHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

        # set values to 0 when it is before 6 am:
        TargetTable[:, 2:][TargetTable[:, 1] < 6] = 0
        # set values to 0 when it is after 10 pm:
        TargetTable[:, 2:][TargetTable[:, 1] > 21] = 0

        # save arrays to database:
        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_DishWasherHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

    def generate_washing_machine_and_dryer_hours(self, NumberOfWashingMachineProfiles):
        """
        same as dish washer function above
        washingmachine starting hours are between 06:00 and 20:00 because the dryer might be used afterwards
        Note: The last day is not being used! because in the optimization the dryer would run until
        """
        # Dryer has no own function because it will always be used after the washing machine
        Demand_WashingMachine = DB().read_DataFrame(REG_Table().Sce_Demand_WashingMachine, self.conn)
        WashingMachineDuration = int(Demand_WashingMachine.WashingMachineDuration)
        Demand_Dryer = DB().read_DataFrame(REG_Table().Sce_Demand_Dryer, self.conn)
        DryerDuration = int(Demand_Dryer.DryerDuration)
        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)

        # Assumption: Washing machine runs maximum once a day:
        UseDays = int(Demand_WashingMachine.WashingMachineCycle)
        TotalDays = TimeStructure.ID_Day.to_numpy()[-1] - 1  # last day is added later

        # create array with 0 or 1 for every hour a day and random choice of 0 and 1 but with correct numbers of 1
        # (UseDays)
        def rand_bin_array(usedays, totaldays):
            arr = np.zeros(totaldays)
            arr[:usedays] = 1
            np.random.shuffle(arr)
            arr = np.repeat(arr, 24, axis=0)
            return arr

        TargetTable = TimeStructure.ID_Hour.to_numpy()[:-24]
        TargetTable = np.column_stack([TargetTable, TimeStructure.ID_DayHour.to_numpy()[:-24]])
        TargetTable_columns = {"ID_Hour": "INTEGER",
                               "ID_DayHour": "INTEGER"}
        Dryer_columns = {"ID_Hour": "INTEGER",
                         "ID_DayHour": "INTEGER"}
        for i in range(NumberOfWashingMachineProfiles + 1):
            TargetTable = np.column_stack([TargetTable, rand_bin_array(UseDays, TotalDays)])
            TargetTable_columns["WashingMachineHours " + str(i)] = "INTEGER"
            Dryer_columns["DryerHours " + str(i)] = "INTEGER"

        # append the last day to the target table:
        lastDay_hours = TimeStructure.ID_Hour.to_numpy()[-24:]
        lastDay_IDHours = TimeStructure.ID_DayHour.to_numpy()[-24:]
        lastDay_zeros = np.zeros((len(lastDay_IDHours), TargetTable.shape[1] - 2))
        lastDay = np.column_stack([lastDay_hours, lastDay_IDHours, lastDay_zeros])

        # merge last day to target table:
        TargetTable = np.vstack([TargetTable, lastDay])

        # iterate through table and assign random values between 06:00 and 19:00 on UseDays (dryer has
        # to be used as well):
        # this table is for reference scenarios or if the dishwasher is not optimized:
        TargetTable_washmachine = np.copy(TargetTable)
        TargetTable_dryer = np.copy(TargetTable)  # for the dryer
        for index in range(0, len(TargetTable_washmachine), 24):
            for column in range(2, TargetTable_washmachine.shape[1]):
                if TargetTable_washmachine[index, column] == 1:
                    HourOfTheDay = np.random.randint(low=6,
                                                     high=21) - 1  # weil bei 0 zu zählen anfängt (Hour of day 6 ist 7 uhr)
                    TargetTable_washmachine[index + HourOfTheDay:index + HourOfTheDay + WashingMachineDuration,
                    column] = 1
                    TargetTable_washmachine[index:index + HourOfTheDay, column] = 0
                    TargetTable_washmachine[index + HourOfTheDay + WashingMachineDuration:index + 24, column] = 0

                    # Dryer always starts 1 hour after the washing machine:
                    TargetTable_dryer[
                    index + HourOfTheDay + WashingMachineDuration + 1:index + HourOfTheDay + WashingMachineDuration + 1 + DryerDuration,
                    column] = 1
                    TargetTable_dryer[index:index + HourOfTheDay + WashingMachineDuration + 1, column] = 0
                    TargetTable_dryer[index + HourOfTheDay + WashingMachineDuration + 1 + DryerDuration:index + 24,
                    column] = 0

        # write dataframe with starting hours to database:
        DB().write_DataFrame(TargetTable_washmachine, REG_Table().Gen_Sce_WashingMachineStartingHours,
                             TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)
        DB().write_DataFrame(TargetTable_dryer, REG_Table().Gen_Sce_DryerStartingHours, Dryer_columns.keys(),
                             self.conn, dtype=Dryer_columns)

        # set values to 0 when it is before 6 am:
        TargetTable[:, 2:][TargetTable[:, 1] < 6] = 0
        # set values to 0 when it is after 8 pm:
        TargetTable[:, 2:][TargetTable[:, 1] > 20] = 0
        # save starting days for optimization:
        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_WashingMachineHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

    def generate_target_indoor_temperature(self,
                                           temperature_min: int,
                                           temperature_max: int,
                                           night_reduction: int) -> (np.array, np.array):
        """
        generates the target indoor temperature table from input lists which contain max and min temperature.
        Night reduction is done from 22:00 until 06:00 where the minimum temperature is reduced by the value of
        the night reduction value. Returns the arrays for minimum and maximum temperature.
        """
        minimum_temperature = np.array([])
        maximum_temperature = np.array([])
        for daytime in self.id_day_hour:
            if 6 >= daytime >= 22:
                minimum_temperature = np.append(minimum_temperature, temperature_min - night_reduction)
                maximum_temperature = np.append(maximum_temperature, temperature_max)
            else:
                minimum_temperature = np.append(minimum_temperature, temperature_min)
                maximum_temperature = np.append(maximum_temperature, temperature_max)
        return minimum_temperature, maximum_temperature

    def generate_target_indoor_temperature_with_outside_temperature_dependency(self,
                                                                               temperature_min: int,
                                                                               temperature_max: int
                                                                               ) -> (np.array, np.array):
        """
        set temperature with smart system that adapts according to outside temperature to save energy:
        the heating temperature will be calculated by a function, if the heating temperature is above temperature_min,
        it is capped. The maximum temperature will be the minimum temperature + 5°C in winter and when the outside
        temperature is above temperature_min  the maximum temperature is set to temperature_max for cooling.
        Fct: Tset = 20°C + (T_outside - 20°C + 12)/8
        returns minimum and maximum indoor set temperatures as array
        """
        engine = config.root_connection.connect()
        assert engine.dialect.has_table(engine, Table().temperature)  # check if outside temperature table exists

        outside_temperature = DB().read_dataframe(Table().temperature)["temperature"]
        minimum_temperature = np.array([])
        maximum_temperature = np.array([])
        for T_out in outside_temperature:
            if 20 + (T_out - 20 + 12) / 8 <= temperature_min:
                minimum_temperature = np.append(minimum_temperature, 20 + (T_out - 20 + 12) / 8)
                maximum_temperature = np.append(maximum_temperature, 20 + (T_out - 20 + 12) / 8 + 5)
            else:
                minimum_temperature = np.append(minimum_temperature, temperature_min)
                maximum_temperature = np.append(maximum_temperature, temperature_max)
        return minimum_temperature, maximum_temperature

    def generate_hourly_COP_air_conditioner(self):
        """
        returns the hourly COP of the AC. For now we only use a constant COP of 3.
        This can be changed in the future...
        """
        # constant COP of 3 is estimated:
        COP = 3
        COP_array = np.full((len(self.id_hour),), COP)
        return COP_array

    def generate_electricity_profile(self, fixed_price: float):
        """fixed price has to be in cent/kWh"""
        # load electricity price
        variable_electricity_price = pd.read_excel(
            "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/Elec_price_per_hour.xlsx",
            engine="openpyxl").to_numpy() / 10 + 15  # cent/kWh

        mean_price_vector = np.full((8760,), fixed_price)  # cent/kWh

        variable_price_to_db = np.column_stack(
            [np.full((8760,), 1),  # ID
             self.id_hour,  # id_hour
             variable_electricity_price,
             np.full((8760,), "cent/kWh")]
        )
        fixed_price_to_db = np.column_stack(
            [np.full((8760,), 2),  # ID
             self.id_hour,
             mean_price_vector,
             np.full((8760,), "cent/kWh")]
        )
        price_to_db = np.vstack([variable_price_to_db, fixed_price_to_db])
        price_columns = {"ID_PriceType": "INT",
                         "ID_Country": "TEXT",
                         "ID_Hour": "INTEGER",
                         "ElectricityPrice": "REAL",
                         "Unit": "TEXT"}
        # save to database

        DB().write_DataFrame(price_to_db, "Gen_Sce_ElectricityProfile",
                             column_names=price_columns.keys(),
                             conn=self.conn,
                             dtype=price_columns)
        # FIT
        feed_in_tariff = np.column_stack(
            [np.full((8760,), 1), np.arange(8760), np.full((8760,), 7.67), np.full((8760,), "cent/kWh")])
        feed_in_columns = {"ID_FeedinTariffType": "INT",
                           "ID_Hour": "INTEGER",
                           "HourlyFeedinTariff": "REAL",
                           "HourlyFeedinTariff_unit": "TEXT"}

        DB().write_DataFrame(feed_in_tariff, "Sce_Price_HourlyFeedinTariff",
                             column_names=feed_in_columns.keys(),
                             conn=self.conn,
                             dtype=feed_in_columns)

    def gen_Sce_Demand_BaseElectricityProfile(self):
        baseload = pd.read_csv(Path().absolute().parent.resolve() / Path("_Philipp/inputdata/AUT/synthload2019.csv"),
                               sep=None, engine="python")
        baseload_h0 = baseload.loc[baseload["Typnummer"] == 1].set_index("Zeit", drop=True).drop(columns="Typnummer")
        baseload_h0.index = pd.to_datetime(baseload_h0.index)
        baseload_h0["Wert"] = pd.to_numeric(baseload_h0["Wert"].str.replace(",", "."))
        baseload_h0 = baseload_h0[3:].resample("1H").sum()
        baseload_h0 = baseload_h0.reset_index(drop=True).rename(columns={"Wert": "BaseElectricityProfile"})

        columns = {"BaseElectricityProfile": "REAL"}
        DB().write_DataFrame(table=baseload_h0, table_name="Sce_Demand_BaseElectricityProfile", conn=self.conn,
                             column_names=columns.keys(), dtype=columns)

    def gen_Sce_HotWaterProfile(self):
        hot_water = pd.read_excel(
            Path().absolute().parent.resolve() / Path("_Philipp/inputdata/AUT/Hot_water_profile.xlsx"),
            engine="openpyxl")
        hot_water_profile = np.column_stack([hot_water["Profile"].to_numpy(), np.full((8760,), "kWh")])
        columns = {REG_Var().HotWater: "REAL", "Unit": "TEXT"}
        DB().write_DataFrame(hot_water_profile, "Gen_Sce_HotWaterProfile", columns.keys(), self.conn, dtype=columns)

    def gen_Sce_ID_Environment(self):

        ElectricityPriceType = DB().read_DataFrame(REG_Table().ID_ElectricityPrice, self.conn)
        FeedinTariffType = DB().read_DataFrame(REG_Table().Sce_ID_FeedinTariffType, self.conn)
        # HotWaterProfileType = DB().read_DataFrame(REG_Table().Sce_ID_HotWaterProfileType, self.Conn)
        # PhotovoltaicProfileType = DB().read_DataFrame(REG_Table().Sce_ID_PhotovoltaicProfileType, self.Conn)
        # TargetTemperatureType = DB().read_DataFrame(REG_Table().Sce_ID_TargetTemperatureType, self.Conn)
        # BaseElectricityProfileType = DB().read_DataFrame(REG_Table().Sce_ID_BaseElectricityProfileType, self.Conn)
        # EnergyCostType = DB().read_DataFrame(REG_Table().Sce_ID_EnergyCostType, self.Conn)

        TargetTable_list = []

        TargetTable_columns = ["ID"]

        # TargetTable_columns += ["ID_ElectricityPriceType", "ID_TargetTemperatureType", "ID_FeedinTariffType",
        #                         "ID_HotWaterProfileType", "ID_PhotovoltaicProfileType", "ID_BaseElectricityProfileType",
        #                         "ID_EnergyCostType"]
        TargetTable_columns += ["ID_ElectricityPriceType", "ID_FeedinTariffType"]

        ID = 1

        for row1 in range(0, len(ElectricityPriceType)):
            # for row2 in range(0, len(TargetTemperatureType)):
            for row3 in range(0, len(FeedinTariffType)):
                # for row4 in range(0, len(HotWaterProfileType)):
                #     for row5 in range(0, len(PhotovoltaicProfileType)):
                #         for row6 in range(0, len(BaseElectricityProfileType)):
                #             for row7 in range(0, len(EnergyCostType)):
                TargetTable_list.append([ID] +
                                        [ElectricityPriceType.iloc[row1][
                                             "ID_ElectricityPriceType"]] +
                                        # [TargetTemperatureType.iloc[row2][
                                        #      "ID_TargetTemperatureType"]] +
                                        [FeedinTariffType.iloc[row3]["ID_FeedinTariffType"]])  # +
                # [HotWaterProfileType.iloc[row4]["ID_HotWaterProfileType"]] +
                # [PhotovoltaicProfileType.iloc[row5][
                #      "ID_PhotovoltaicProfile"]] +
                # [BaseElectricityProfileType.iloc[row6][
                #      "ID_BaseElectricityProfileType"]] +
                # [EnergyCostType.iloc[row6]["ID_EnergyCostType"]]

                ID += 1

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_ID_Environment, TargetTable_columns, self.conn)

    def run(self):
        self.generate_dishwasher_hours(
            10)  # TODO this number should be externally set or we dont need the profiles anyways
        self.generate_washing_machine_and_dryer_hours(10)
        self.generate_target_indoor_temperature()
        self.gen_Sce_AC_HourlyCOP()
        self.gen_Sce_electricity_price()
        self.gen_Sce_Demand_BaseElectricityProfile()
        self.gen_Sce_HotWaterProfile()
        self.gen_Sce_ID_Environment()

