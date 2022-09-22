from basics.config import Config
from basics.db import create_db_conn
from basics.plotter import Plotter
from models.community.constants import CommunityTable


class CommunityAnalyzer:

    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)
        self.plotter = Plotter(config)

    def plot_aggregator_profit_against_battery_size(self):
        scenarios = self.db.read_dataframe(CommunityTable.Scenarios)
        result_year = self.db.read_dataframe(CommunityTable.ResultYear)
        values_dict = {
            "P2P_Profit": result_year["p2p_profit"].to_numpy() / 100,
            "Opt_Profit": result_year["opt_profit"].to_numpy() / 100
        }
        battery_sizes = scenarios["aggregator_battery_size"].to_numpy() / 10**6
        self.plotter.bar_figure(
            values_dict=values_dict,
            fig_name=f"AggregatorProfit-Battery",
            x_label="Battery Size (MWh)",
            y_label="Aggregator Profit (€)",
            x_tick_labels=battery_sizes
        )

