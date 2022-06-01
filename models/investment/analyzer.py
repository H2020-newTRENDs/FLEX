from typing import TYPE_CHECKING
from basics.db import create_db_conn
from basics import kit

if TYPE_CHECKING:
    from basics.config import Config


logger = kit.get_logger(__name__)


class InvestmentAnalyzer:

    def __init__(self, config: 'Config'):
        self.db = create_db_conn(config)
        