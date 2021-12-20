
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class Behavior(Component):

    # indoor_hours: np.array
    # driving_behavior: np.array
    # appliance_behavior: np.array
    # hot_water_behavior: np.array

    def __init__(self, params_dict: Dict[str, Any]):
        self.indoor_hours = None
        self.driving_behavior = None
        self.appliance_behavior = None
        self.hot_water_behavior = None
        self.set_params(params_dict)
