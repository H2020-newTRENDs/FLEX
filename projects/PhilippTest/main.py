import numpy as np

from core.household.abstract_scenario import AbstractScenario
from models.operation.opt import OptOperationModel
from models.operation.ref import RefOperationModel
from models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector
from basic.db import DB



# for loop over all the scenarios
scenario_ids = len(DB().read_dataframe("Scenarios", *["ID_Scenarios"]).to_numpy())

for id in range(scenario_ids):
    print(f"scenario: {id}")
    # create scenario:
    scenario = AbstractScenario(scenario_id=id)

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


# Only run the reference model:

# for id in range(scenario_ids):
#     print(f"\n scenario: {id}")
#     id = 106
#     # create scenario:
#     scenario = AbstractScenario(scenario_id=id)
#
#     reference_model = RefOperationModel(scenario)
#     reference_model.run()
#     # save results to db
#     ReferenceDataCollector(reference_model).save_yearly_results()
#     ReferenceDataCollector(reference_model).save_hourly_results()



# Only run the optimization model:

# for id in range(scenario_ids):
#     print(f"\n scenario: {id}")
#     # create scenario:
#     scenario = AbstractScenario(scenario_id=id)
#
#     optimization_model = OptOperationModel(scenario)
#     # solve model
#     solved_instance = optimization_model.run()
#     # datacollector
#     OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
#     OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()




