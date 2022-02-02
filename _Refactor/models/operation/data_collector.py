import pandas as pd

from _Refactor.core.elements.data_collector import MotherDataCollector
import numpy as np


class OptimizationDataCollector(MotherDataCollector):

    def extend_to_array(self, array_or_value: np.array) -> np.array:
        # check if array_or_value is an array or a single value:
        if len(array_or_value) == 1:
            return np.full((8760, ), array_or_value)  # return array with same value
        else:
            return array_or_value

    def collect_optimization_results_hourly(self, instance) -> pd.DataFrame:
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

        return dataframe

    def collect_optimization_results_yearly(self, instance) -> pd.DataFrame:
        # empty dictionary
        dictionary = {}
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
            result_array = np.array(list(result_class.extract_values().values()))
            if len(result_array) > 1:  # if its hourly value, take sum
                # check if the values in every hour are exactly the same (eg. fixed feed in tariff):
                if np.all(result_array == result_array[0]):
                    result_value = result_array[0]
                else:  # else take the sum
                    result_value = result_array.sum()
            else:  # if its already a single value, just take that value
                result_value = result_array
            # add the value to the dictionary with its name:
            dictionary[result_name] = result_value
        # create the dataframe
        dataframe = pd.DataFrame(dictionary)
        return dataframe



class ReferenceDataCollector(MotherDataCollector):
    def __init__(self, reference_model):
        self.reference_model = reference_model

    def collect_reference_results_hourly(self):
        # empty dataframe
        dataframe = pd.DataFrame()
        # iterate over all variables in the instance
        for result_name in self.reference_model.__dict__.keys():
            if result_name == "scenario" \
                    or result_name == "cp_water":  # do not save pyomo implemented types and the time set
                continue

            result_class = getattr(self.reference_model, result_name)
            # check if value is array or single value

            result_value = np.array(list(result_class.extract_values().values()))
            dataframe[result_name] = self.extend_to_array(result_value)

        # for input_data in self.reference_model.scenario.__dict__.keys()
        pass

    def collect_reference_results_yearly(self):

        pass






