import logging
from basics.config import Config

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
config = Config(project_name="TryBuildingsAgain")
