from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod
import sqlalchemy
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import sqlalchemy.types
from basics.db import create_db_conn
from basics.kit import get_logger
from models.operation.enums import ResultEnum


if TYPE_CHECKING:
    from basics.config import Config
    from models.operation.abstract import OperationModel

logger = get_logger(__name__)


class DataCollector(ABC):

    def __init__(self,
                 model: Union['OperationModel', 'pyo.ConcreteModel'],
                 scenario_id: int,
                 config: 'Config'):
        self.model = model
        self.scenario_id = scenario_id
        self.db = create_db_conn(config)
        self.hour_result = {}
        self.year_result = {}

    @abstractmethod
    def get_var_values(self, variable_name: str) -> np.array:
        ...

    @abstractmethod
    def get_total_cost(self) -> float:
        ...

    @abstractmethod
    def get_hour_result_table_name(self) -> str:
        ...

    @abstractmethod
    def get_year_result_table_name(self) -> str:
        ...

    def collect_result(self):
        for variable_name, variable_enum in ResultEnum.__members__.items():
            var_values = self.get_var_values(variable_name)
            self.hour_result[variable_name] = var_values
            if variable_enum.value == "year_include":
                self.year_result[variable_name] = var_values.sum()

    def save_hour_result(self):
        result_hour_df = pd.DataFrame(self.hour_result)
        result_hour_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        self.db.write_dataframe(table_name=self.get_hour_result_table_name(),
                                data_frame=result_hour_df)

    def save_year_result(self):
        result_year_df = pd.DataFrame(self.year_result, index=[0])
        result_year_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_year_df.insert(loc=1, column="TotalCost", value=self.get_total_cost())
        self.db.write_dataframe(table_name=self.get_year_result_table_name(),
                                data_frame=result_year_df)

    def run(self):
        self.collect_result()
        self.save_hour_result()
        self.save_year_result()


class OptDataCollector(DataCollector):

    def get_var_values(self, variable_name: str) -> np.array:
        var_values = np.array(list(self.model.__dict__[variable_name].extract_values().values()))
        return var_values

    def get_total_cost(self) -> float:
        total_cost = self.model.total_operation_cost_rule()
        return total_cost

    def get_hour_result_table_name(self) -> str:
        return "Result_OptimizationHour"

    def get_year_result_table_name(self) -> str:
        return "Result_OptimizationYear"


class RefDataCollector(DataCollector):

    def get_var_values(self, variable_name: str) -> np.array:
        var_values = np.array(self.model.__dict__[variable_name])
        return var_values

    def get_total_cost(self) -> float:
        total_cost = self.model.__dict__["TotalCost"].sum()
        return total_cost

    def get_hour_result_table_name(self) -> str:
        return "Result_ReferenceHour"

    def get_year_result_table_name(self) -> str:
        return "Result_ReferenceYear"







class ReferenceDataCollector:

    def __init__(self, reference_model, config: 'Config'):
        self.reference_model = reference_model
        self.db = create_db_conn(config)

    def check_type_hourly(self, value) -> np.array:
        if isinstance(value, float) or isinstance(value, int):  # if its a single float, int return array of same value
            return_value = np.full((8760,), value)
        elif isinstance(value, np.ndarray):
            if len(value) == 1:  # if its array with only one entry return array with 8760 entries
                return_value = np.full((8760,), value)
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

    def collect_reference_results_hourly(self) -> pd.DataFrame:
        # empty dataframe
        dataframe = pd.DataFrame()
        dataframe["scenario_id"] = self.reference_model.scenario.scenario_id
        # iterate over all variables in the instance
        for result_name in self.reference_model.__dict__.keys():
            if result_name == "scenario" \
                    or result_name == "cp_water" \
                    or result_name == "total_operation_cost" \
                    or result_name == "thermal_mass_start_temperature":  # do not save scenario and cp_water, total cost
                continue
            # exclude the building parameters from the results (too much)
            if result_name in list(self.reference_model.scenario.building.__dict__.keys()):
                continue

            result_class = getattr(self.reference_model, result_name)
            # check if result_class is array or single value and if they
            # are single values make them to 8760 array
            result_array = self.check_type_hourly(result_class)
            # save to df
            dataframe[result_name] = result_array
        return dataframe

    def collect_reference_results_yearly(self) -> pd.DataFrame:
        # empty dict
        dictionary = {}
        # iterate over all variables in the instance
        for result_name in self.reference_model.__dict__.keys():
            if result_name == "scenario" \
                    or result_name == "cp_water" \
                    or result_name == "DayHour" \
                    or result_name == "thermal_mass_start_temperature":  # do not save scenario and cp_water and day_hour
                continue
            # exclude the building parameters from the results (too much)
            if result_name in list(self.reference_model.scenario.building.__dict__.keys()):
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

    def save_hourly_results(self, if_exists: str = "append") -> None:

        hourly_results = self.collect_reference_results_hourly()
        # set the datatype to float:
        d_types_dict = {}
        for column_name in hourly_results.columns:
            d_types_dict[column_name] = sqlalchemy.types.Float

        # save the dataframes to the result database
        logger.info(f'ReferenceHourly has {len(hourly_results.columns)} columns.')
        self.db.write_dataframe(table_name="Reference_hourly",
                                data_frame=hourly_results,
                                data_types=d_types_dict,
                                if_exists=if_exists)

    def save_yearly_results(self, if_exists: str = "append") -> None:
        yearly_results = self.collect_reference_results_yearly()
        # set the datatype to float:
        d_types_dict = {}
        for column_name in yearly_results.columns:
            d_types_dict[column_name] = sqlalchemy.types.Float
        # save the dataframes to the result database
        self.db.write_dataframe(table_name="Reference_yearly",
                                data_frame=yearly_results,
                                data_types=d_types_dict,
                                if_exists=if_exists)
