

# TODO this has to be changed for each project
import _Refactor.projects.PhilippTest.config as configurations

# these imports are fixed
import numpy as np
from _Refactor.basic.config import Config
from _Refactor.data.table_generator import InputDataGenerator
from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel
from _Refactor.models.operation.ref import RefOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector


# create list of all configurations defined in configurations
config_list = [{config_name: value} for (config_name, value) in configurations.__dict__.items()
               if not config_name.startswith("__")]
# define scenario:
configuration = Config(config_list)
# create all the data for the calculations:
InputDataGenerator(configuration).run()


# create scenario:
scenario = AbstractScenario(scenario_id=0)

optimization_model = OptOperationModel(scenario)
# solve model
solved_instance = optimization_model.run()
# datacollector
OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()

reference_model = RefOperationModel(scenario)
reference_model.run()
# save results to db
ReferenceDataCollector(reference_model).save_yearly_results()
ReferenceDataCollector(reference_model).save_hourly_results()






