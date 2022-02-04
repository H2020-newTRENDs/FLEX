
import os

import numpy as np

from config import project_config
from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel
from _Refactor.models.operation.ref import RefOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector


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






