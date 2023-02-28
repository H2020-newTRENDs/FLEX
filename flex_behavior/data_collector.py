from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from flex.db import create_db_conn
from flex.kit import get_logger


if TYPE_CHECKING:
    from flex.config import Config

logger = get_logger(__name__)


class BehaviorDataCollector(ABC):
    def __init__(self, scenario_id: int, config: "Config"):
        self.scenario_id = scenario_id
        self.db = create_db_conn(config)
