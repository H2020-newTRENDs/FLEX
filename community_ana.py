from config import config
from models.community.analyzer import CommunityAnalyzer


class ProjectCommunityAnalyzer(CommunityAnalyzer):
    ...


if __name__ == "__main__":
    ana = ProjectCommunityAnalyzer(config)
    ana.plot_aggregator_profit_against_battery_size()
    # ana.plot_p2p_trading_amount(id_scenario=1)
    # ana.plot_battery_operation_by_month(id_scenario=1)
    # ana.plot_battery_operation_by_month(id_scenario=30)
