from typing import TYPE_CHECKING
from basics.db import create_db_conn
from basics.kit import get_logger


if TYPE_CHECKING:
    from basics.config import Config
    from models.investment.model import InvestmentModel

logger = get_logger(__name__)


class InvestmentDataCollector:

    def __init__(self,
                 model: 'InvestmentModel',
                 scenario_id: int,
                 config: 'Config'):
        self.model = model
        self.scenario_id = scenario_id
        self.db = create_db_conn(config)

    def run(self):
        pass
