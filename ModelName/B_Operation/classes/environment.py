import pandas

class Environment:

    def __init__(self, id_country, params):

        self.id_country = id_country
        self.id_environment = params["id_environment"]
        self.temperature = self.setup_temperature()
        self.radiation = self.setup_radiation()
        self.electricity_price = self.setup_electricity_price()
        self.feedin_tariff = self.setup_feedin_tariff()

    def setup_temperature(self):
        pass

    def setup_radiation(self):
        pass

    def setup_electricity_price(self):
        pass

    def setup_feedin_tariff(self):
        pass
