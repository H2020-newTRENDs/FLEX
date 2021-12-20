
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class Demand(Component):

    # indoor_temperature_min: np.array
    # indoor_temperature_max: np.array
    # driving_load_profile: np.array
    # appliance_load_profile: np.array
    # hot_water_load_profile: np.array

    def __init__(self, params_dict: Dict[str, Any]):
        self.indoor_temperature_min = None
        self.indoor_temperature_max = None
        self.driving_load_profile = None
        self.appliance_load_profile = None
        self.hot_water_load_profile = None
        self.set_params(params_dict)

