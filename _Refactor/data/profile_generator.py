import pandas as pd
import numpy as np
import sqlalchemy.types
from pathlib import Path

from _Refactor.basic.db import DB
from _Refactor.basic.reg import Table
import _Refactor.core.household.components as components
import _Refactor.basic.config as config
from _Refactor.data import input_data_structure as structure


class ProfileGenerator:
    def __init__(self):
        self.id_hour = np.arange(1, 8761)
        self.id_day_hour = np.tile(np.arange(1, 25), 365)

    def generate_target_indoor_temperature_fixed(self,
                                                 temperature_min: int,
                                                 temperature_max: int,
                                                 night_reduction: int):
        """
        generates the target indoor temperature table from input lists which contain max and min temperature.
        Night reduction is done from 22:00 until 06:00 where the minimum temperature is reduced by the value of
        the night reduction value. Returns the arrays for minimum and maximum temperature.
        """
        minimum_temperature = np.array([])
        maximum_temperature = np.array([])
        for daytime in self.id_day_hour:
            if daytime <= 6 or daytime >= 22:
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

    def generate_behaviour_table(self):
        minimum_indoor_temperature, maximum_indoor_temperature = self.generate_target_indoor_temperature_fixed(
            temperature_min=20,
            temperature_max=27,
            night_reduction=2
        )
        behaviour_dict = {"ID_Behavior": np.full((8760,), 1),
                          "indoor_set_temperature_min": minimum_indoor_temperature,
                          "indoor_set_temperature_max": maximum_indoor_temperature}
        behaviour_table = pd.DataFrame(behaviour_dict)
        assert list(behaviour_table.columns).sort() == list(structure.BehaviorData().__dict__.keys()).sort()

        DB().write_dataframe(table_name=Table().behavior,
                             data_frame=behaviour_table,
                             data_types=structure.BehaviorData().__dict__,
                             if_exists="replace"
                             )


    def generate_electricity_price_profile(self, fixed_price: float):
        """fixed price has to be in cent/kWh"""
        # load electricity price
        variable_electricity_price = pd.read_excel(
            "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/Elec_price_per_hour.xlsx",
            engine="openpyxl").to_numpy() / 10 + 15  # cent/kWh

        fixed_price_vector = np.full((8760,), fixed_price)  # cent/kWh

        variable_price_to_db = np.column_stack(
            [np.full((8760,), 1),  # ID
             self.id_hour,  # id_hour
             variable_electricity_price,
             np.full((8760,), "cent/kWh")]
        )
        fixed_price_to_db = np.column_stack(
            [np.full((8760,), 2),  # ID
             self.id_hour,
             fixed_price_vector,
             np.full((8760,), "cent/kWh")]
        )
        price_to_db = np.vstack([variable_price_to_db, fixed_price_to_db])
        price_table = pd.DataFrame(price_to_db, columns=list(structure.ElectricityPriceData().__dict__.keys()))
        # save to database

        DB().write_dataframe(table_name=Table().electricity_price,
                             data_frame=price_table,
                             data_types=structure.ElectricityPriceData().__dict__,
                             if_exists="replace"
                             )

    def generate_feed_in_price_profile(self, constant_feed_in: float):
        # FIT
        feed_in_dict = {"ID_FeedInTariff": np.full((8760,), 1),
                        "id_hour": self.id_hour,
                        "feed_in_tariff": np.full((8760,), constant_feed_in),
                        "unit": np.full((8760,), "cent/kWh")}
        feed_in_table = pd.DataFrame(feed_in_dict)
        assert list(feed_in_table.columns).sort() == list(structure.FeedInTariffData().__dict__.keys()).sort()
        DB().write_dataframe(table_name=Table().feedin_tariff,
                             data_frame=feed_in_table,
                             data_types=structure.FeedInTariffData().__dict__,
                             if_exists="replace"
                             )

    def generate_base_electricity_demand(self):
        baseload = pd.read_csv(
            Path().absolute().parent.parent.resolve() / Path("_Philipp/inputdata/AUT/synthload2019.csv"),
            sep=None, engine="python")
        baseload_h0 = baseload.loc[baseload["Typnummer"] == 1].set_index("Zeit", drop=True).drop(columns="Typnummer")
        baseload_h0.index = pd.to_datetime(baseload_h0.index)
        baseload_h0["Wert"] = pd.to_numeric(baseload_h0["Wert"].str.replace(",", "."))
        baseload_h0 = baseload_h0[3:].resample("1H").sum()
        baseload_h0 = baseload_h0.reset_index(drop=True).rename(columns={"Wert": "electricity_demand"})
        baseload_h0 = baseload_h0.to_numpy() * 1_000  # from kWh in Wh

        baseload_dict = {"ID_ElectricityDemand": np.full((8760,), 1),
                         "electricity_demand": baseload_h0.flatten(),
                         "unit": np.full((8760,), "Wh")}
        baseload_table = pd.DataFrame(baseload_dict)
        assert list(baseload_table.columns).sort() == list(structure.ElectricityDemandData().__dict__.keys()).sort()
        DB().write_dataframe(table_name=Table().electricity_demand,
                             data_frame=baseload_table,
                             data_types=structure.ElectricityDemandData().__dict__,
                             if_exists="replace"
                             )

    def generate_hot_water_profile(self):
        hot_water = pd.read_excel(
            Path().absolute().parent.parent.resolve() / Path("_Philipp/inputdata/AUT/Hot_water_profile.xlsx"),
            engine="openpyxl")
        hot_water_dict = {"ID_HotWaterDemand": np.full((8760,), 1),
                          "hot_water_demand": hot_water["Profile"].to_numpy(),
                          "unit": np.full((8760,), "kWh")}
        hot_water_table = pd.DataFrame(hot_water_dict)
        assert list(hot_water_table.columns).sort() == list(structure.HotWaterDemandData().__dict__.keys()).sort()

        DB().write_dataframe(table_name=Table().hot_water_demand,
                             data_frame=hot_water_table,
                             data_types=structure.HotWaterDemandData().__dict__,
                             if_exists="replace"
                             )

    def run(self):

        self.generate_behaviour_table()
        self.generate_hot_water_profile()
        self.generate_base_electricity_demand()
        self.generate_electricity_price_profile(fixed_price=20)
        self.generate_feed_in_price_profile(constant_feed_in=7.67)


if __name__ == "__main__":
    ProfileGenerator().run()
