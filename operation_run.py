from basics.kit import get_logger
from config import config
from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import FuelBoilerOptInstance
from models.operation.model_opt import FuelBoilerOptOperationModel
from models.operation.model_opt import HeatPumpOptInstance
from models.operation.model_opt import HeatPumpOptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario

logger = get_logger(__name__)

if __name__ == "__main__":
    hp_instance = HeatPumpOptInstance().create_instance()
    fb_instance = FuelBoilerOptInstance().create_instance()
    scenario_ids = [1, 2]
    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")
        scenario = OperationScenario(scenario_id=scenario_id, config=config)
        ref_model = RefOperationModel(scenario).solve()
        if scenario.boiler.type in ["Air_HP", "Ground_HP"]:
            opt_model = HeatPumpOptOperationModel(scenario).solve(hp_instance)
        else:
            opt_model = FuelBoilerOptOperationModel(scenario).solve(fb_instance)
        RefDataCollector(ref_model, scenario.scenario_id, config).run()
        OptDataCollector(opt_model, scenario.scenario_id, config).run()

