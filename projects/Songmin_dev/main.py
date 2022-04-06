from flex.flex_operation.scenario import OperationScenario
from config import config
from flex.flex_operation_old.opt import OptOperationModel
from flex.flex_operation_old.opt import OptimizationDataCollector



scenario = OperationScenario(scenario_id=1, config=config)

print(scenario.behavior.vehicle_distance)
x = scenario.behavior.target_temperature_array_min

print(scenario.boiler.thermal_power_max)
y = scenario.energy_price.electricity_consumption
y= scenario.boiler.db_name
print(scenario.building.internal_gains)

optimization_model = OptOperationModel(scenario)
solved_instance = optimization_model.run()


OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()




