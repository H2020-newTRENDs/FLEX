from models.investment.model import InvestmentModel
from models.investment.scenario import InvestmentScenario
from config import config
from basics import kit

logger = kit.get_logger(__name__)

if __name__ == "__main__":
    scenario_ids = list(range(1, 2))
    for scenario_id in scenario_ids:
        logger.info(f'Scenario = {scenario_id}')
        scenario = InvestmentScenario(scenario_id=scenario_id, config=config)
        inv_model = InvestmentModel(scenario).run()


