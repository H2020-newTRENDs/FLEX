from flex.flex_operation.scenario import OperationScenario
from config import config
from flex.flex_operation.opt import OptOperationModel
from flex.flex_operation.opt import OptimizationDataCollector

scenario = OperationScenario(scenario_id=1, config=config)

#print(scenario.behavior.vehicle_distance)
#x = scenario.behavior.target_temperature_array_min

#print(scenario.boiler.thermal_power_max)
#y= scenario.boiler.db_name
#print(scenario.building.internal_gains)

b = scenario.energy_price.electricity_consumption
x = scenario.behavior.hot_water_demand
a = scenario.behavior.vehicle_at_home
c = scenario.behavior.vehicle_demand

print(scenario.vehicle.charge_bidirectional)

optimization_model = OptOperationModel(scenario)
solved_instance = optimization_model.run()

# OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()
# OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()
