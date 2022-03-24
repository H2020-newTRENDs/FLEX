

# All Power inputs have to be provided in W !!!!
__country = "AT"
__year = 2019

air_conditioner_config = {
    "efficiency": 3,
    "power": [0, 10_000],
}

appliance_config = {
    None: None
}

battery_config = {
    "capacity": [0, 7_000],
    "charge_efficiency":  0.95,
    "discharge_efficiency": 0.95,
    "charge_power_max": 4_500,
    "discharge_power_max": 4_500
}

behavior_config = {
    "temperature_min": 20,
    "temperature_max": 27,  # TODO make this function from outside temp
    "night_reduction": 2
}

boiler_config = {
    "name": ["Air_HP", "Ground_HP"],
    "thermal_power_max": 15_000,   # W
    "heating_element_power": 7_500,
    "carnot_efficiency_factor": [0.4, 0.35],  # for the heat pumps respectively
    "heating_supply_temperature": 35,
    "hot_water_supply_temperature":  55
}

building_config = {
    "building_mass_temperature_max": 60,  # °C
    "grid_power_max": 21_000,  # W
}

demand_config = {
    "base_load_year": str(__year),  # [2018, 2019, ... 2024]
}

electricity_price_config = {
    # variable price
    "api_key": 'c06ee579-f827-486d-bc1f-8fa0d7ccd3da',
    "start": f"{__year}0101",
    "end": f"{__year+1}0101",
    "country_code": __country,
    "grid_fee": 20,  # ct/kWh

    # fixed price
    "fixed_price": 20,  # ct/kWh

    # extra price scenarios:
    "extra_price_scenario": True  # False or True (if True, prices have to be provided in an extra json)
}

feed_in_tariff_config = {
    "fixed_feed_in_tariff": [7.67],  # ct/kWh ,more feed in tarifs can be provided in list
    "variable_feed_in_tariff_path": [None]  # for variable feed ins provide path(s)
}

hot_water_tank_config = {
    "size": [0, 400],
    "loss": 0.2,  # W/m2K
    "temperature_start": 28,  # °C
    "temperature_max": 55,  # °C
    "temperature_min": 28,
    "temperature_surrounding": 20  # °C
}

person_config = {
    None: None
}

region_config = {
    "nuts_level": 3,
    "country_code": __country,
    "start_year": __year,
    "end_year": __year,
    "pv_size": [0, 5, 10]  # kWp
}

space_heating_tank_config = {
    "size": [0, 750],
    "loss": 0.2,  # W/m2K
    "temperature_start": 28,  # °C
    "temperature_max": 45,  # °C
    "temperature_min": 28,
    "temperature_surrounding": 20  # °C
}

vehicle_config = {
    None: None
}

