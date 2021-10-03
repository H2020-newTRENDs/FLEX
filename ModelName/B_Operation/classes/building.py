import pandas

class Building:

    def __init__(self, building_params: pandas.Series, building_technology_params: pandas.Series):

        # building

        # heating
        self.id_heating_energy_carrier = building_technology_params["id_heating_energy_carrier"]
        self.heating_boiler_type = building_technology_params["heating_boiler_type"]
        self.heating_boiler_COP = None
        self.heating_tank_size = building_technology_params["heating_tank_size"]

        # cooling
        self.id_cooling_energy_carrier = building_technology_params["id_cooling_energy_carrier"]
        self.cooling_tech_type = building_technology_params["cooling_tech_type"]
        self.cooling_COP = None

        # hot water
        self.id_hot_water_energy_carrier = building_technology_params["id_hot_water_energy_carrier"]
        self.hot_water_tech_type = building_technology_params["hot_water_tech_type"]
        self.hot_water_COP = None

        # pv
        self.pv_adoption = building_technology_params["pv_adoption"]
        self.pv_size = building_technology_params["pv_size"]

        # battery
        self.battery_adoption = building_technology_params["battery_adoption"]
        self.battery_size = building_technology_params["battery_size"]

    def setup_heating_COP(self, temperature):
        # hourly COP --> based on self.heating_boiler_type and temperature
        # self.heating_boiler_COP = 0
        pass

    def setup_cooling_COP(self, temperature):
        # hourly COP --> based on self.cooling_tech_type and temperature
        # self.cooling_COP = 0
        pass

    def setup_hot_water_COP(self, temperature):
        # hourly COP --> based on self.hot_water_tech_type and temperature
        # self.cooling_COP = 0
        pass

    def setup_pv_generation(self, radiation):
        # hourly PV generation --> based on self.hot_water_tech_type and temperature
        # self.pv_generation = 0
        pass






