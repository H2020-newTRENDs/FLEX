from models.operation.scenario import OperationScenario
from models.operation.opt import OptOperationModel
from models.operation.ref import RefOperationModel
from models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector
from config import config

scenario = OperationScenario(scenario_id=1, config=config)

optimization_model = OptOperationModel(scenario).run()
OptimizationDataCollector(optimization_model, scenario.scenario_id, config).save_yearly_results()
OptimizationDataCollector(optimization_model, scenario.scenario_id, config).save_hourly_results()

reference_model = RefOperationModel(scenario).run()
ReferenceDataCollector(reference_model, config).save_yearly_results()
ReferenceDataCollector(reference_model, config).save_hourly_results()
