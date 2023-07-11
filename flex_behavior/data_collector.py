from typing import TYPE_CHECKING

from flex.db import create_db_conn
from flex.kit import get_logger

if TYPE_CHECKING:
    from flex.config import Config

logger = get_logger(__name__)


class BehaviorDataCollector:
    def __init__(self, config: "Config"):
        self.db = create_db_conn(config)
