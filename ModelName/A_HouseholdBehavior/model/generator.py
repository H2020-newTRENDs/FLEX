
from ..classes.person import Person
from ..classes.household import Household
from ..classes.environment import Environment

class Generator:

    def __init__(self, environment: Environment, household: Household):

        self.environment = environment
        self.household = household

    def save_household_life_plan(self):
        # save also self.environment.id_country and self.environment.monte_carlo_generation_number
        pass

    def run(self):
        self.household.calc_household_life_plan(self.environment.time_structure, self.environment.annual_holiday)
        self.save_household_life_plan()