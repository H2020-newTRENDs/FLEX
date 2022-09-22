from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from basics.db import create_db_conn
from basics.kit import get_logger
from models.community.constants import CommunityTable
from models.community.model import CommunityModel

if TYPE_CHECKING:
    from basics.config import Config

logger = get_logger(__name__)


class CommunityDataCollector:
    def __init__(
        self,
        model: "CommunityModel",
        opt_instance: "pyo.ConcreteModel",
        scenario_id: int,
        config: "Config",
        save_hour_results: bool = True
    ):
        self.model = model
        self.opt_instance = opt_instance
        self.scenario_id = scenario_id
        self.db = create_db_conn(config)
        self.hour_result = {}
        self.year_result = {}
        self.save_hour = save_hour_results

    def run(self):
        self.collect_hour_result()
        self.save_hour_result()
        self.collect_year_result()
        self.save_year_result()

    def get_var_values(self, variable_name: str) -> np.array:
        var_values = np.array(list(self.opt_instance.__dict__[variable_name].extract_values().values()))
        return var_values

    def collect_hour_result(self):
        self.hour_result = {
            "p2p_trading": self.model.p2p_trading,
            "battery_charge": self.get_var_values("battery_charge"),
            "battery_discharge": self.get_var_values("battery_discharge"),
            "battery_soc": self.get_var_values("battery_soc"),
            "buy_price": self.get_var_values("buy_price"),
            "sell_price": self.get_var_values("sell_price"),
        }

    def save_hour_result(self):
        result_hour_df = pd.DataFrame(self.hour_result)
        result_hour_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        result_hour_df.insert(loc=1, column="Hour", value=list(range(1, 8761)))
        result_hour_df.insert(loc=2, column="DayHour", value=np.tile(np.arange(1, 25), 365))
        self.db.write_dataframe(table_name=CommunityTable.ResultHour, data_frame=result_hour_df)

    def collect_year_result(self):
        self.year_result = {
            "p2p_profit": self.model.p2p_trading_profit,
            "opt_profit": self.opt_instance.opt_profit_rule(),
            "total_profit": self.model.p2p_trading_profit + self.opt_instance.opt_profit_rule()
        }

    def save_year_result(self):
        result_year_df = pd.DataFrame(self.year_result, index=[0])
        result_year_df.insert(loc=0, column="ID_Scenario", value=self.scenario_id)
        self.db.write_dataframe(table_name=CommunityTable.ResultYear, data_frame=result_year_df)


