from basics.config import Config
from basics.db import create_db_conn
from basics.plotter import Plotter
from models.community.constants import CommunityTable
import numpy as np


class CommunityAnalyzer:

    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)
        self.plotter = Plotter(config)

    def plot_aggregator_profit_against_battery_size(self):
        scenarios = self.db.read_dataframe(CommunityTable.Scenarios)
        result_year = self.db.read_dataframe(CommunityTable.ResultYear)
        values_dict = {
            "P2P_Profit": result_year["p2p_profit"].to_numpy() / 10**5,
            "Opt_Profit": result_year["opt_profit"].to_numpy() / 10**5
        }
        battery_sizes = scenarios["aggregator_battery_size"].to_numpy() / 10**6
        self.plotter.bar_figure(
            values_dict=values_dict,
            fig_name=f"AggregatorProfit-Battery",
            x_label="Battery Size (MWh)",
            y_label="Aggregator Profit (k€)",
            y_lim=(0, 100),
            x_tick_labels=battery_sizes
        )

    def plot_p2p_trading_amount_by_month(self, id_scenario: int):
        result_hour = self.db.read_dataframe(CommunityTable.ResultHour, filter={"ID_Scenario": id_scenario})
        p2p_trading = result_hour["p2p_trading"].to_numpy()
        trading_amount = []
        for month in range(0, 12):
            start_hour = month * 730
            end_hour = month * 730 + 730
            trading_amount.append(np.sum(p2p_trading[start_hour:end_hour])/1000)
        values_dict = {"P2P_trading": np.array(trading_amount)}
        self.plotter.bar_figure(
            values_dict=values_dict,
            fig_name=f"P2P_TradingAmount",
            x_label="Month of the Year",
            y_label="P2P Trading Amount (kWh)",
            add_legend=False,
            y_lim=(0, 16000),
        )

    def plot_battery_operation_by_month(self, id_scenario: int):
        result_hour = self.db.read_dataframe(CommunityTable.ResultHour, filter={"ID_Scenario": id_scenario})
        battery_charge = result_hour["battery_charge"].to_numpy()
        battery_discharge = result_hour["battery_discharge"].to_numpy()
        battery_charge_by_month = []
        battery_discharge_by_month = []
        for month in range(0, 12):
            start_hour = month * 730
            end_hour = month * 730 + 730
            battery_charge_by_month.append(np.sum(battery_charge[start_hour:end_hour]) / 10**6)
            battery_discharge_by_month.append(np.sum(battery_discharge[start_hour:end_hour]) / 10**6)
        values_dict = {
            "BatteryCharge": np.array(battery_charge_by_month),
            "BatteryDischarge": - np.array(battery_discharge_by_month)
        }
        self.plotter.bar_figure(
            values_dict=values_dict,
            fig_name=f"BatteryOperationByMonth_S{id_scenario}",
            x_label="Month of the Year",
            y_label="Battery Charge and Discharge Amount (MWh)",
            y_lim=(-50, 50),
            x_tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
        )

