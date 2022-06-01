from models.investment.analyzer import InvestmentAnalyzer
from config import config


class ProjectInvestmentAnalyzer(InvestmentAnalyzer):
    pass


if __name__ == "__main__":
    ana = ProjectInvestmentAnalyzer(config)

