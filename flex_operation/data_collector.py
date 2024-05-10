from typing import TYPE_CHECKING, Union, Optional, List
from abc import ABC, abstractmethod
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import logging

from matplotlib import pyplot as plt

from flex.db import create_db_conn
from flex_operation.constants import OperationResultVar, OperationTable


if TYPE_CHECKING:
    from flex.config import Config
    from flex_operation.model_base import OperationModel



class OperationDataCollector(ABC):
    def __init__(
        self,
        model: Union["OperationModel", "pyo.ConcreteModel"],
        scenario_id: int,
        config: "Config",
        save_hour_results: Optional[bool] = True,
        save_month_results: Optional[bool] = False,
        save_year_results: Optional[bool] = True,
        hourly_save_list: Optional[List[str]] = None
    ):
        """
        :param model: either the ref model or the opt model
        :param scenario_id: id of the scenario that is being saved
        :param config: configuration
        :param save_hour_results: if True hourly results are saved (default = True)
        :param save_month_results: if True monthly results are saved (default = False)
        :param save_year_results: if True yearly results are saved (default = True)
        :param hourly_save_list: if a list of variables is provided only these variables are being saved as hourly
                results. save_hourly_results has to be True. This is to save disc space if only eg. the Load is needed.
        """
        self.model = model
        self.scenario_id = scenario_id
        self.config = config
        self.db = create_db_conn(config)
        self.hour_result = {}
        self.month_result = {}
        self.year_result = {}
        self.save_hour = save_hour_results
        self.save_month = save_month_results
        self.save_year = save_year_results
        self.logger = logging.getLogger(f"{config.project_name}")
        self.hourly_save_list = hourly_save_list

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
    def get_month_result_table_name(self) -> str:
        ...

    @abstractmethod
    def get_year_result_table_name(self) -> str:
        ...

    def convert_hour_to_month(self, values):
        month_results = []
        hours_per_month = {
            1: (1, 744),
            2: (745, 1416),
            3: (1417, 2160),
            4: (2161, 2880),
            5: (2881, 3624),
            6: (3625, 4344),
            7: (4345, 5088),
            8: (5089, 5832),
            9: (5833, 6552),
            10: (6553, 7296),
            11: (7297, 8016),
            12: (8017, 8760)
        }
        for month, hour_range in hours_per_month.items():
            month_sum = values[hour_range[0] - 1:hour_range[1]].sum()
            month_results.append(month_sum)
        return month_results

    def collect_result(self):
        for variable_name, variable_type in OperationResultVar.__dict__.items():
            if not variable_name.startswith("_"):
                var_values = self.get_var_values(variable_name)
                # check if the load in the reference model has outliers which would indicate a problem:
                # if variable_name == "Load" and self.get_hour_result_table_name() == OperationTable.ResultRefHour:
                #     self.check_hourly_results_for_outliers(var_values, variable_name)
                self.hour_result[variable_name] = var_values
                if variable_type == "hour&year":
                    self.month_result[variable_name] = self.convert_hour_to_month(var_values)
                    self.year_result[variable_name] = var_values.sum()

    def check_hourly_results_for_outliers(self, profile: np.array, var_name: str):
        """
        check the result for extreme outliers which indicates that something went wrong in the calculation
        using the IQR method
        """
        Q1 = np.percentile(profile, 25)
        Q3 = np.percentile(profile, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        outlier_indices = np.where((profile < lower_bound) | (profile > upper_bound))[0]
        if len(outlier_indices) > 0:
            print(f"Outlier detected in {var_name} for scenario {self.scenario_id} "
                  f"in table {self.get_hour_result_table_name()}")
            self.logger.warning(f"Outlier detected in profile: {var_name} for scenario {self.scenario_id} "
                                f"in table {self.get_hour_result_table_name()}")
            plt.plot(np.arange(8760), profile)
            ax = plt.gca()
            plt.vlines(x=outlier_indices, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors="red", linestyles="dotted")
            plt.title(f"{var_name} in scenario {self.scenario_id}, \n table: {self.get_hour_result_table_name()}")
            plt.show()

    def reduce_df_size(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        reduce the size of a pd Dataframe by down casting the integer type (automatically) and float type (to float32).
        :param frame: df that should be saved
        :return: DataFrame with reduced size
        """
        float_cols = frame.select_dtypes(include='float').columns
        frame[float_cols] = frame[float_cols].astype('float32')
        frame = frame.apply(pd.to_numeric, downcast='integer')
        return frame

    def save_hour_result(self):
        result_hour_df = pd.DataFrame(self.hour_result)
        result_hour_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_hour_df.insert(loc=1, column="Hour", value=list(range(1, 8761)))
        result_hour_df.insert(
            loc=2, column="DayHour", value=np.tile(np.arange(1, 25), 365)
        )
        if self.hourly_save_list:
            result_hour_df = result_hour_df.loc[:, self.hourly_save_list]
        df_to_save = self.reduce_df_size(result_hour_df)
        self.db.write_to_parquet(table_name=self.get_hour_result_table_name(),
                                 scenario_ID=self.scenario_id,
                                 table=df_to_save,
                                 folder=self.config.output)

    def save_month_result(self):
        result_month_df = pd.DataFrame(self.month_result)
        result_month_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_month_df.insert(loc=1, column="Month", value=list(range(1, 13)))
        df_to_save = self.reduce_df_size(result_month_df)
        self.db.write_dataframe(
            table_name=self.get_month_result_table_name(),
            data_frame=df_to_save
        )

    def save_year_result(self):
        result_year_df = pd.DataFrame(self.year_result, index=[0])
        result_year_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_year_df.insert(loc=1, column="TotalCost", value=self.get_total_cost())
        df_to_save = self.reduce_df_size(result_year_df)

        self.db.write_dataframe(
            table_name=self.get_year_result_table_name(),
            data_frame=df_to_save
        )

    def run(self):
        self.collect_result()
        if self.save_hour:
            self.save_hour_result()
        if self.save_month:
            self.save_month_result()
        if self.save_year:
            self.save_year_result()


class OptDataCollector(OperationDataCollector):
    def get_var_values(self, variable_name: str) -> np.array:
        var_values = np.array(
            list(self.model.__dict__[variable_name].extract_values().values())
        )
        return var_values

    def get_total_cost(self) -> float:
        total_cost = self.model.total_operation_cost_rule()
        return total_cost

    def get_hour_result_table_name(self) -> str:
        return OperationTable.ResultOptHour

    def get_month_result_table_name(self) -> str:
        return OperationTable.ResultOptMonth

    def get_year_result_table_name(self) -> str:
        return OperationTable.ResultOptYear


class RefDataCollector(OperationDataCollector):
    def get_var_values(self, variable_name: str) -> np.array:
        var_values = np.array(self.model.__dict__[variable_name])
        return var_values

    def get_total_cost(self) -> float:
        total_cost = self.model.__dict__["TotalCost"].sum()
        return total_cost

    def get_hour_result_table_name(self) -> str:
        return OperationTable.ResultRefHour

    def get_month_result_table_name(self) -> str:
        return OperationTable.ResultRefMonth

    def get_year_result_table_name(self) -> str:
        return OperationTable.ResultRefYear
