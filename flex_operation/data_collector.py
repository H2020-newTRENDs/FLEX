from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from flex.db import create_db_conn
from flex.kit import get_logger
from flex_operation.constants import OperationResultVar, OperationTable


if TYPE_CHECKING:
    from flex.config import Config
    from flex_operation.model_base import OperationModel

logger = get_logger(__name__)


class OperationDataCollector(ABC):
    def __init__(
        self,
        model: Union["OperationModel", "pyo.ConcreteModel"],
        scenario_id: int,
        config: "Config",
        save_hour_results: bool
    ):
        self.model = model
        self.scenario_id = scenario_id
        self.db = create_db_conn(config)
        self.hour_result = {}
        self.year_result = {}
        self.save_hour = save_hour_results

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
        for variable_name, variable_type in OperationResultVar.__dict__.items():
            if not variable_name.startswith("_"):
                var_values = self.get_var_values(variable_name)
                self.hour_result[variable_name] = var_values
                if variable_type == "hour&year":
                    self.year_result[variable_name] = var_values.sum()

    def save_hour_result(self):
        result_hour_df = pd.DataFrame(self.hour_result)
        result_hour_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_hour_df.insert(loc=1, column="Hour", value=list(range(1, 8761)))
        result_hour_df.insert(
            loc=2, column="DayHour", value=np.tile(np.arange(1, 25), 365)
        )
        self.db.write_dataframe(
            table_name=self.get_hour_result_table_name(), data_frame=result_hour_df
        )

    def save_year_result(self):
        result_year_df = pd.DataFrame(self.year_result, index=[0])
        result_year_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_year_df.insert(loc=1, column="TotalCost", value=self.get_total_cost())
        self.db.write_dataframe(
            table_name=self.get_year_result_table_name(), data_frame=result_year_df
        )

    def run(self):
        self.collect_result()
        if self.save_hour:
            self.save_hour_result()
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

    def get_year_result_table_name(self) -> str:
        return OperationTable.ResultRefYear
