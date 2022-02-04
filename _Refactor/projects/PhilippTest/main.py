
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
hourly_results_optimization_df = OptimizationDataCollector().collect_optimization_results_hourly(solved_instance)
yearly_results_optimization_df = OptimizationDataCollector().collect_optimization_results_yearly(solved_instance)

reference_model = RefOperationModel(scenario)
reference_model.run()
hourly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_hourly()
yearly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_yearly()




