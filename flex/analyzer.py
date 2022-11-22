from typing import TYPE_CHECKING, Type

from flex import kit
from flex.db import create_db_conn

if TYPE_CHECKING:
    from flex.config import Config
    from flex.plotter import Plotter

logger = kit.get_logger(__name__)


class FlexAnalyzer:

    def __init__(self, config: "Config", plotter_cls: Type["Plotter"]):
        self.db = create_db_conn(config)
        self.output_folder = config.output
        self.plotter = plotter_cls(config)
