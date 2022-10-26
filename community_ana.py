from config import config
from models.community.analyzer import CommunityAnalyzer


class ProjectCommunityAnalyzer(CommunityAnalyzer):
    ...


if __name__ == "__main__":
    ana = ProjectCommunityAnalyzer(config)
    ana.plot_aggregator_profit_against_battery_size()
