from flex.flex_operation.scenario import OperationScenario
from config import config
from flex.flex_operation_old.opt import OptOperationModel


scenario = OperationScenario(scenario_id=1, config=config)

print(scenario.behavior.vehicle_distance)
x = scenario.behavior.target_temperature_array_min

print(scenario.boiler.thermal_power_max)
y = scenario.energy_price.electricity_consumption
y= scenario.boiler.db_name






optimization_model = OptOperationModel(scenario)
solved_instance = optimization_model.run()




