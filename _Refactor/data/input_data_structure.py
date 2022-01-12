import sqlalchemy.types


class RadiationData:
    def __init__(self):
        # columns
        self.nuts_id = sqlalchemy.types.String
        self.id_hour = sqlalchemy.types.Integer
        self.south = sqlalchemy.types.Float
        self.east = sqlalchemy.types.Float
        self.west = sqlalchemy.types.Float
        self.north = sqlalchemy.types.Float


class TemperatureData:
    def __init__(self):
        self.nuts_id = sqlalchemy.types.String
        self.id_hour = sqlalchemy.types.Integer
        self.temperature = sqlalchemy.types.Float


class PVGenerationData:
    def __init__(self):
        self.nuts_id = sqlalchemy.types.String
        self.ID_PV = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.power = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class ElectricityPriceData:
    def __init__(self):
        self.ID_ElectricityPrice = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.electricity_price = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class FeedInTariffData:
    def __init__(self):
        self.ID_FeedInTariff = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.feed_in_tariff = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String
