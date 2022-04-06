from core.household.abstract_scenario import AbstractScenario
from flex.flex_operation.opt import OptOperationModel
#from flex.flex_operation_old.opt import RefOperationModel
#from flex.flex_operation_old.opt import OptimizationDataCollector, ReferenceDataCollector
from flex.flex_operation.opt import OptimizationDataCollector
from basic.db import DB


# for loop over all the scenarios
scenario_ids = len(DB().read_dataframe("Scenarios", *["ID_Scenarios"]).to_numpy())

for scenario_id in range(scenario_ids):
    print(f"scenario: {scenario_id}")
    # create scenario:
    scenario = AbstractScenario(scenario_id=scenario_id)

    optimization_model = OptOperationModel(scenario)
    # solve model
    solved_instance = optimization_model.run()
    # datacollector
    OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
    OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()

#   reference_model = RefOperationModel(scenario)
#    reference_model.run()
    # save results to db
#    ReferenceDataCollector(reference_model).save_yearly_results()
#    ReferenceDataCollector(reference_model).save_hourly_results()

# Total Operation Cost: 54590.28
# Total Operation Cost reference: 67816.73

# test








