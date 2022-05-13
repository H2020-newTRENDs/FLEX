from models.operation.scenario import OperationScenario
from models.operation.opt import OptOperationModel
from models.operation.data_collector import OptimizationDataCollector
from config import config

scenario = OperationScenario(scenario_id=1, config=config)

optimization_model = OptOperationModel(scenario)
solved_instance = optimization_model.run()
OptimizationDataCollector(solved_instance, scenario.scenario_id, config).save_yearly_results()
OptimizationDataCollector(solved_instance, scenario.scenario_id, config).save_hourly_results()





# scenario = OperationScenario(scenario_id=2, config=config)
# reference_model = RefOperationModel(scenario)
# solved_instance = reference_model.run()
# ReferenceDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
# ReferenceDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()
