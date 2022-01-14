import numpy as np

from _Refactor.basic.db import DB
from _Refactor.basic.reg import Table

pv_power = DB().read_dataframe(Table().pv_generation)["power"].to_numpy()
outside_temperature = DB().read_dataframe(Table().temperature)["temperature"].to_numpy()
mean_24_outside_temperature = np.add.reduceat(outside_temperature, np.arange(0, len(outside_temperature), 24)) / 24
pv_generation_in_24_hours = np.add.reduceat(pv_power, np.arange(0, len(pv_power), 24))

daily_electricty_consumption = DB().read_dataframe(Table().electricity_demand)["electricity_demand"].to_numpy()
electricity_consumption_24_hours = np.add.reduceat(daily_electricty_consumption, np.arange(0, len(daily_electricty_consumption), 24))

for i, mean_temp in enumerate(mean_24_outside_temperature):
    if mean_temp >= 12:
        pv_generation_in_24_hours = np.delete(pv_generation_in_24_hours, i)

maximum_pv_generation_in_24_hours_cold_days = np.max(pv_generation_in_24_hours)
print(f"maximum_pv_generation_in_24_hours: {maximum_pv_generation_in_24_hours_cold_days / 1000} kWh")
estimated_COP = 3
print(f"maximum generated heat with pv and HP: {(maximum_pv_generation_in_24_hours_cold_days - np.mean(electricity_consumption_24_hours)) * 3 / 1000} kWh")

tanksize = 1500
min_temp = 28
max_temp = 45
tank_capacity = tanksize * 4200 * (max_temp - min_temp) / 3600 / 1_000
print(f"maximum tank capacity: {tank_capacity} kWh")
