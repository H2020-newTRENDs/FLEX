
# TODO this has to be changed for each project
import _Refactor.projects.PhilippTest.config as configurations
from _Refactor.basic.config import Config
from _Refactor.data.table_generator import InputDataGenerator


# create list of all configurations defined in configurations
config_list = [{config_name: value} for (config_name, value) in configurations.__dict__.items()
               if not config_name.startswith("__")]
# define scenario:
configuration = Config(config_list)
# create all the data for the calculations:
InputDataGenerator(configuration).run(skip_region=True)

