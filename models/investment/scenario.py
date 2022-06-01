from dataclasses import dataclass

from basics.config import Config


@dataclass
class InvestmentScenario:
    scenario_id: int
    config: 'Config'
