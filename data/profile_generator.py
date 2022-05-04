import numpy as np

from core.elements.rc_model import R5C1Model
from core.household.abstract_scenario import AbstractScenario


class ProfileGenerator:
    def __init__(self,
                 scenario: AbstractScenario):
        self.scenario = scenario
        # calculate the heating demand in reference mode:
        self.heating_demand, _, self.T_room, _ = R5C1Model(scenario).calculate_heating_and_cooling_demand()


    @staticmethod
    def generate_target_indoor_temperature_fixed(temperature_min: int,
                                                 temperature_max: int,
                                                 night_reduction: int) -> (np.array, np.array):
        """
        generates the target indoor temperature table from input lists which contain max and min temperature.
        Night reduction is done from 22:00 until 06:00 where the minimum temperature is reduced by the value of
        the night reduction value.

        Returns the arrays for minimum and maximum temperature.
        """
        minimum_temperature = np.array([])
        maximum_temperature = np.array([])
        id_day_hour = np.tile(np.arange(1, 25), 365)
        for daytime in id_day_hour:
            if daytime <= 6 or daytime >= 22:
                minimum_temperature = np.append(minimum_temperature, temperature_min - night_reduction)
                maximum_temperature = np.append(maximum_temperature, temperature_max)
            else:
                minimum_temperature = np.append(minimum_temperature, temperature_min)
                maximum_temperature = np.append(maximum_temperature, temperature_max)
        return minimum_temperature, maximum_temperature

    def generate_maximum_target_indoor_temperature_no_cooling(self,
                                                              temperature_max: int) -> np.array:
        """
        calculates an array of the maximum temperature that will be equal to the provided max temperature if
        heating is needed at the current hour, but it will be "unlimited" (60°C) if no heating is needed. This
        way it is ensured that the model does not pre-heat the building above the maximum temperature and at the same
        time the model will not be infeasible when the household is heated above maximum temperature by external
        radiation.
        Args:
            temperature_max: int

        Returns: array of the maximum temperature

        """
        # create max temperature array:
        max_temperature_list = []
        for i, heat_demand in enumerate(self.heating_demand):
            # if heat demand == 0 and the heat demand in the following 8 hours is also 0 and the heat demand of 3
            # hours before that is also 0, then the max temperature is raised so model does not become infeasible:
            if heat_demand == 0 and \
                   self.heating_demand[i - 3] == 0 and \
                   self.heating_demand[i - 2] == 0 and \
                   self.heating_demand[i - 1] == 0 and \
                   self.heating_demand[i + 1] == 0 and \
                   self.heating_demand[i + 2] == 0 and \
                   self.heating_demand[i + 3] == 0 and \
                   self.heating_demand[i + 4] == 0 and \
                   self.heating_demand[i + 5] == 0 and \
                   self.heating_demand[i + 6] == 0 and \
                   self.heating_demand[i + 7] == 0 and \
                   self.heating_demand[i + 8] == 0:
                # append the temperature of the reference model + 0.5°C to make sure it is feasible
                max_temperature_list.append(self.T_room[i] + 1)
            else:
                max_temperature_list.append(temperature_max)
        return np.array(max_temperature_list)



    # def generate_target_indoor_temperature_with_outside_temperature_dependency(self,
    #                                                                            temperature_min: int,
    #                                                                            temperature_max: int
    #                                                                            ) -> (np.array, np.array):
    #     """
    #     set temperature with smart system that adapts according to outside temperature to save energy:
    #     the heating temperature will be calculated by a function, if the heating temperature is above temperature_min,
    #     it is capped. The maximum temperature will be the minimum temperature + 5°C in winter and when the outside
    #     temperature is above temperature_min  the maximum temperature is set to temperature_max for cooling.
    #     Fct: Tset = 20°C + (T_outside - 20°C + 12)/8
    #     returns minimum and maximum indoor set temperatures as array
    #     """
    #     engine = config.root_connection.connect()
    #     assert engine.dialect.has_table(engine, Table().temperature)  # check if outside temperature table exists
    #
    #     outside_temperature = DB().read_dataframe(Table().temperature)["temperature"]
    #     minimum_temperature = np.array([])
    #     maximum_temperature = np.array([])
    #     for T_out in outside_temperature:
    #         if 20 + (T_out - 20 + 12) / 8 <= temperature_min:
    #             minimum_temperature = np.append(minimum_temperature, 20 + (T_out - 20 + 12) / 8)
    #             maximum_temperature = np.append(maximum_temperature, 20 + (T_out - 20 + 12) / 8 + 5)
    #         else:
    #             minimum_temperature = np.append(minimum_temperature, temperature_min)
    #             maximum_temperature = np.append(maximum_temperature, temperature_max)
    #     return minimum_temperature, maximum_temperature

    def run(self):
        self.generate_target_indoor_temperature_fixed(temperature_min=20,
                                                      temperature_max=27,
                                                      night_reduction=2)

        scenario = AbstractScenario(scenario_id=0)
        self.generate_maximum_target_indoor_temperature_no_cooling(scenario=scenario,
                                                                   temperature_max=23)

        self.generate_maximum_heat_pump_power(scenario=scenario)


if __name__ == "__main__":
    ProfileGenerator().run()
