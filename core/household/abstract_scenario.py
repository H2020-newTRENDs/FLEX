
import core.household.components as components
from core.elements.component_setup import GeneralComponent
from basic.reg import Table


class AbstractScenario:

    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        self.region = None
        self.person_class = None
        self.personlist_class = None
        self.building = None
        self.appliance_class = None
        self.appliancelist_class = None
        self.boiler = None
        self.space_heating_tank = None
        self.space_cooling_technology = None
        self.hot_water_tank = None
        self.pv = None
        self.battery = None
        self.vehicle_class = None
        self.behavior = None
        self.hotwaterdemand_class = None
        self.electricitydemand_class = None
        self.electricityprice_class = None
        self.feedintariff_class = None

        # get all scenario ids from the scenarios table:
        self.scenario_ids = GeneralComponent(
            table_name=Table().scenarios,
            row_id=scenario_id
        ).get_complete_data()  # is a dictionary  with the ids
        self.add_component_classes()

    def add_component_classes(self):
        # iterate through the scenario dict which hols all IDs for each component:
        for component_name, component_id in self.scenario_ids.items():
            name = component_name.lower().replace("id_", "")
            for key in components.__dict__.keys():
                if name == key.lower():  # check if they class exists and use its name
                    class_name = key
                    # get the class
                    # print(f"{class_name} exists")
            instance = getattr(components, class_name)(component_id=component_id)  # all classes take the component_id as parameter to initialize their values
            # create self variable of the class with the filled class
            setattr(self, name + "_class", instance)





if __name__ == "__main__":
    household_class = AbstractScenario(scenario_id=0)
    household_class.add_component_classes()

