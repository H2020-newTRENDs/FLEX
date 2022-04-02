import sqlalchemy

source_data_types = {
    'region': sqlalchemy.types.Unicode,
    'year': sqlalchemy.types.BigInteger,
    'id_hour': sqlalchemy.types.BigInteger,
    'pv_generation': sqlalchemy.types.Float,
    'pv_generation_unit': sqlalchemy.types.Unicode,
    'temperature': sqlalchemy.types.Float,
    'temperature_unit': sqlalchemy.types.Unicode,
    'radiation_south': sqlalchemy.types.Float,
    'radiation_east': sqlalchemy.types.Float,
    'radiation_west': sqlalchemy.types.Float,
    'radiation_north': sqlalchemy.types.Float,
    'radiation_unit': sqlalchemy.types.Unicode,
    'people_at_home': sqlalchemy.types.BigInteger,
    'vehicle_at_home': sqlalchemy.types.BigInteger,
    'vehicle_distance': sqlalchemy.types.Float,
    'vehicle_distance_unit': sqlalchemy.types.Unicode,
    'hot_water_demand': sqlalchemy.types.Float,
    'hot_water_demand_unit': sqlalchemy.types.Unicode,
    'appliance_electricity': sqlalchemy.types.Float,
    'appliance_electricity_unit': sqlalchemy.types.Unicode,
}
