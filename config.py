import logging
from basics.config import Config

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
config = Config(project_name="5R1C_6R2C")


def get_config(project_name: str):
    return Config(project_name=project_name)

