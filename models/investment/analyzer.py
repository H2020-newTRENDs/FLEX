from typing import TYPE_CHECKING, ClassVar
import numpy as np

from basics.db import create_db_conn
from basics import kit
from basics.enums import InterfaceTable
from basics.plotter import Plotter

if TYPE_CHECKING:
    from basics.config import Config


logger = kit.get_logger(__name__)


class InvestmentAnalyzer:
    def __init__(self, config: "Config", plotter: ClassVar["Plotter"] = Plotter):
        self.db = create_db_conn(config)
        self.plotter = plotter(config)
        self.operation_energy_cost = self.db.read_dataframe(
            InterfaceTable.OperationEnergyCost.value
        )

    def run(self):
        pass
