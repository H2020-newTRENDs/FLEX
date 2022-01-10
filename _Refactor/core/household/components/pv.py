from typing import Dict, Any
import sqlalchemy.types
from _Refactor.core.elements.component import Component


class PV(Component):
    def __init__(self):
        self.ID_PV: int = None
        self.peak_power: float = None
        self.peak_power_unit: str = None


class RadiationData:
    def __init__(self):
        # columns
        self.nuts_id: str = sqlalchemy.types.String
        self.id_hour: int = sqlalchemy.types.Integer
        self.south: float = sqlalchemy.types.Float
        self.east: float = sqlalchemy.types.Float
        self.west: float = sqlalchemy.types.Float
        self.north: float = sqlalchemy.types.Float


class TemperatureData:
    def __init__(self):
        self.nuts_id: str = sqlalchemy.types.String
        self.id_hour: int = sqlalchemy.types.Integer
        self.temperature: float = sqlalchemy.types.Float


class PVGeneration:
    def __init__(self):
        self.nuts_id: str = sqlalchemy.types.String
        self.ID_PV: int = sqlalchemy.types.Integer
        self.id_hour: int = sqlalchemy.types.Integer
        self.power: float = sqlalchemy.types.Float
        self.unit: str = sqlalchemy.types.String
