import pandas

class Environment:

    def __init__(self, params: pandas.Series):

        self.id_environment = params["id_environment"]
        self.id_country = params["id_country"]
        self.annual_holiday = params["annual_holiday"]
        self.time_structure = None
        self.monte_carlo_generation_number = params["monte_carlo_generation_number"]

    def setup_time_structure(self):
        # self.time_structure =
        # (depending on id_country
        pass