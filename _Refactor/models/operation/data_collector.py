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

    def collect_optimization_results_hourly(self, instance, scenario_id: int) -> pd.DataFrame:
        # empty dataframe
        dataframe = pd.DataFrame()
        # iterate over all variables in the instance
        for result_name in instance.__dict__.keys():
            if result_name.startswith("_") \
                    or result_name == "doc" \
                    or result_name == "statistics" \
                    or result_name == "config" \
                    or result_name == "solutions" \
                    or result_name == "t" \
                    or result_name == "Am" \
                    or result_name == "Atot" \
                    or result_name == "Qi" \
                    or result_name == "Htr_w" \
                    or result_name == "Htr_em" \
                    or result_name == "Htr_3" \
                    or result_name == "Htr_1" \
                    or result_name == "Htr_2" \
                    or result_name == "Hve" \
                    or result_name == "Htr_ms" \
                    or result_name == "Htr_is" \
                    or result_name == "PHI_ia" \
                    or result_name == "Cm" \
                    or result_name == "BuildingMassTemperatureStartValue":  # do not save pyomo implemented types and the time set
                continue
            elif "rule" in result_name:  # do not save rules
                continue

            result_class = getattr(instance, result_name)
            result_value = np.array(list(result_class.extract_values().values()))
            dataframe[result_name] = self.extend_to_array(result_value)
            # add the scenario id to the results
            dataframe["scenario_id"] = scenario_id
        return dataframe

    def collect_optimization_results_yearly(self, instance, scenario_id) -> pd.DataFrame:
        # empty dictionary
        dictionary = {}
        # iterate over all variables in the instance
        for result_name in instance.__dict__.keys():
            # do not save pyomo implemented types and the time set and building parameters
            if result_name.startswith("_") \
                    or result_name == "doc" \
                    or result_name == "statistics" \
                    or result_name == "config" \
                    or result_name == "solutions" \
                    or result_name == "t" \
                    or result_name == "DayHour" \
                    or result_name == "Am" \
                    or result_name == "Atot" \
                    or result_name == "Qi" \
                    or result_name == "Htr_w" \
                    or result_name == "Htr_em" \
                    or result_name == "Htr_3" \
                    or result_name == "Htr_1" \
                    or result_name == "Htr_2" \
                    or result_name == "Hve" \
                    or result_name == "Htr_ms" \
                    or result_name == "Htr_is" \
                    or result_name == "PHI_ia" \
                    or result_name == "Cm" \
                    or result_name == "BuildingMassTemperatureStartValue":
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
        # add the scenario id to the results
        dataframe["scenario_id"] = scenario_id
        return dataframe


class ReferenceDataCollector(MotherDataCollector):

    def __init__(self, reference_model):
        self.reference_model = reference_model

    def check_type_hourly(self, value) -> np.array:
        if isinstance(value, float) or isinstance(value, int):  # if its a single float, int return array of same value
            return_value = np.full((8760, ), value)
        elif isinstance(value, np.ndarray):
            if len(value) == 1:  # if its array with only one entry return array with 8760 entries
                return_value = np.full((8760, ), value)
            else:
                return_value = value  # if its array return the array
        elif isinstance(value, list):
            return_value = np.array(value)  # if its list, return list as array
        elif isinstance(value, type(None)):
            return_value = np.full((8760,), 0)
        else:
            return_value = value
        return return_value

    def check_type_yearly(self, value) -> float:
        if isinstance(value, float) or isinstance(value, int):  # if its a single float, int return it
            return_value = value

        elif isinstance(value, np.ndarray):
            if len(value) == 1:  # if its array with only one entry return array with 8760 entries
                return_value = float(value)
            else:  # if its array return the array
                # check if the values in every hour are exactly the same (eg. fixed feed in tariff):
                if np.all(value == value[0]):
                    return_value = float(value[0])
                else:  # else take the sum
                    return_value = float(value.sum())

        elif isinstance(value, list):
            # check if the values in every hour are exactly the same (eg. fixed feed in tariff):
            if np.all(np.array(value) == np.array(value)[0]):
                return_value = float(value[0])
            else:  # else take the sum
                return_value = np.array(value).sum()  # if its list, return the sum
        elif isinstance(value, type(None)):  # if its NoneType return 0
            return_value = 0

        else:  # else return the same value
            return_value = value
        return return_value

    def collect_reference_results_hourly(self):
        # empty dataframe
        dataframe = pd.DataFrame()
        # iterate over all variables in the instance
        for result_name in self.reference_model.__dict__.keys():
            if result_name == "scenario" \
                    or result_name == "cp_water":  # do not save scenario and cp_water
                continue
            # exclude the building parameters from the results (too much)
            if result_name in list(self.reference_model.scenario.building_class.__dict__.keys()):
                continue

            result_class = getattr(self.reference_model, result_name)
            # check if result_class is array or single value and if they
            # are single values make them to 8760 array
            result_array = self.check_type_hourly(result_class)
            # save to df
            dataframe[result_name] = result_array
            # add the scenario id to the results
            dataframe["scenario_id"] = self.reference_model.scenario.scenario_id
        return dataframe

    def collect_reference_results_yearly(self):
        # empty dict
        dictionary = {}
        # iterate over all variables in the instance
        for result_name in self.reference_model.__dict__.keys():
            if result_name == "scenario" \
                    or result_name == "cp_water" \
                    or result_name == "day_hour":  # do not save scenario and cp_water and day_hour
                continue
            # exclude the building parameters from the results (too much)
            if result_name in list(self.reference_model.scenario.building_class.__dict__.keys()):
                continue

            result_class = getattr(self.reference_model, result_name)
            # check if result_class is array or single value and if they
            # are array take the sum except if the array only contains the same values
            result_array = self.check_type_yearly(result_class)
            # save to dict
            dictionary[result_name] = result_array
        dataframe = pd.DataFrame(dictionary, index=[0])
        # add the scenario id to the results
        dataframe["scenario_id"] = self.reference_model.scenario.scenario_id
        return dataframe







