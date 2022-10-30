import logging
from basics.config import Config

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
config = Config(project_name="Flex_5R1C_validation")


def get_config(project_name: str):
    return Config(project_name=project_name)

