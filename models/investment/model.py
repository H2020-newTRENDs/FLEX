from basics.kit import get_logger
from models.investment.scenario import InvestmentScenario

logger = get_logger(__name__)


class InvestmentModel:

    def __init__(self, scenario: 'InvestmentScenario'):
        self.scenario = scenario

    def run(self):
        logger.info(f'FlexInvestment is running for Scenario {self.scenario.scenario_id}.')
        pass