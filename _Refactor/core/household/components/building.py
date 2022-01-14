
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Building(Component):

    def __init__(self, component_id: int):
        self.ID_Building: int = None
        self.invert_index: int = None
        self.name: str = None
        self.construction_period_start: int = None
        self.construction_period_end: int = None
        self.number_of_dwellings_per_building: float = None
        self.number_of_persons_per_dwelling: float = None
        self.horizontal_shading: float = None
        self.area_windows: float = None
        self.area_suitable_for_solar: float = None
        self.hwb_norm: float = None
        self.effective_window_area_west_east: float = None
        self.effective_window_area_north: float = None
        self.effective_window_area_south: float = None
        self.internal_gains: float = None
        self.Af: float = None
        self.Hop: float = None
        self.Htr_w: float = None
        self.Hve: float = None
        self.CM_factor: float = None
        self.Am_factor: float = None
        self.building_mass_temperature_start: float = None
        self.building_mass_temperature_max: float = None
        self.grid_power_max: float = None
        self.grid_power_max_unit: float = None

        self.set_parameters(component_id)

