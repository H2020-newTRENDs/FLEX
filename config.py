import logging
from flex.config import Config

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
cfg = Config(project_name="FLEX_MA_Yanwei")
