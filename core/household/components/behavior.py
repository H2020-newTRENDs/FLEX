from typing import Dict, Any

from core.elements.component import Component


class Behavior(Component):

    # indoor_hours: np.array
    # driving_behavior: np.array
    # appliance_behavior: np.array
    # hot_water_behavior: np.array

    def __init__(self, component_id):
        self.ID_Behavior = None
        self.indoor_set_temperature_min = None
        self.indoor_set_temperature_max = None

        self.set_parameters(component_id)

