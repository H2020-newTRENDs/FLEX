import pandas as pd

from _Refactor.core.elements.data_collector import MotherDataCollector
import numpy as np


class OptimizationDataCollector(MotherDataCollector):

    def __init__(self):
        pass

    def extend_to_array(self, array_or_value: np.array):
        # check if array_or_value is an array or a single value:
        if len(array_or_value) == 1:
            return np.full((8760, ), array_or_value)  # return array with same value
        else:
            return array_or_value

    def collect_OptimizationResult(self, instance) -> (np.array, np.array):
        # empty dataframe
        dataframe = pd.DataFrame()
        # iterate over all variables in the instance
        for result_name in instance.__dict__.keys():
            if result_name.startswith("_") \
                    or result_name == "doc" \
                    or result_name == "statistics" \
                    or result_name == "config" \
                    or result_name == "solutions" \
                    or result_name == "t":  # do not save pyomo implemented types and the time set
                continue
            elif "rule" in result_name:  # do not save rules
                continue

            result_class = getattr(instance, result_name)
            result_value = np.array(list(result_class.extract_values().values()))
            dataframe[result_name] = self.extend_to_array(result_value)





        ElectricityPrice_array = self.extract_Result2Array(
            instance.electricity_price.extract_values()) * 1_000  # Cent/kWh
        FeedinTariff_array = self.extract_Result2Array(instance.FiT.extract_values()) * 1_000  # Cent/kWh
        OutsideTemperature_array = self.extract_Result2Array(instance.T_outside.extract_values())

        E_BaseLoad_array = self.extract_Result2Array(instance.BaseLoadProfile.extract_values()) / 1000  # kW

