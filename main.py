from models.operation.scenario import OperationScenario
from models.operation.opt import OptOperationModel
from models.operation.ref import RefOperationModel
from models.operation.data_collector import OptDataCollector, RefDataCollector
from config import config

scenario = OperationScenario(scenario_id=1, config=config)

opt_model = OptOperationModel(scenario).run()
# OptDataCollector(opt_model, scenario.scenario_id, config).run()

# ref_model = RefOperationModel(scenario).run()
# RefDataCollector(ref_model, scenario.scenario_id, config).run()
