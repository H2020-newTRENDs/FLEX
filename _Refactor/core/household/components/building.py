
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Building(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.ID_Building: int
        self.invert_index: int
        self.name = str
        self.construction_period_start: int
        self.construction_period_end: int
        self.number_of_dwellings_per_building: float
        self.number_of_persons_per_dwelling: float
        self.horizontal_shading: float
        self.area_windows: float
        self.area_suitable_for_solar: float
        self.hwb_norm: float
        self.effective_window_area_west_east: float
        self.effective_window_area_north: float
        self.effective_window_area_south: float
        self.internal_gains: float
        self.Af: float
        self.H_op: float
        self.Htr_w: float
        self.Hve: float
        self.CM_factor: float
        self.Am_factor: float
        self.building_mass_temperature_start: float
        self.building_mass_temperature_max: float
        self.grid_power_max: float
        self.grid_power_max_unit: float

        self.set_params(params_dict)

