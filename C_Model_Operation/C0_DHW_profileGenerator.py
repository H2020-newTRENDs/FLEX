# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
from A_Infrastructure.A1_CONS import CONS


class DHW_probility():

    def __init__(self):
        self.Conn = DB().create_Connection(CONS().RootDB)
        self.showerWater_mass = 3  # liter (ist noch annahme)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        # number of times for hot water activities:  TODO make it dependent on household
        self.bathing_cycles = 101
        self.showering_cycles = 468
        self.sink_cycles = 6885
        # amount of water used per cycle

        self.probability_function_bathing = {1: 0.00772200772200772,
                                             2: 0.00386100386100386,
                                             3: 0.00386100386100386,
                                             4: 0.00386100386100386,
                                             5: 0.00772200772200772,
                                             6: 0.0193050193050193,
                                             7: 0.0463320463320463,
                                             8: 0.0579150579150579,
                                             9: 0.0656370656370656,
                                             10: 0.0579150579150579,
                                             11: 0.0463320463320463,
                                             12: 0.0347490347490347,
                                             13: 0.0308880308880309,
                                             14: 0.0231660231660232,
                                             15: 0.0231660231660232,
                                             16: 0.0231660231660232,
                                             17: 0.0386100386100386,
                                             18: 0.0463320463320463,
                                             19: 0.0772200772200772,
                                             20: 0.1003861003861,
                                             21: 0.1003861003861,
                                             22: 0.0772200772200772,
                                             23: 0.0656370656370656,
                                             24: 0.0386100386100386}

        self.probability_function_showering = {1: 0.010689990281827,
                                               2: 0.00485908649173955,
                                               3: 0.00340136054421769,
                                               4: 0.00485908649173955,
                                               5: 0.0136054421768708,
                                               6: 0.0515063168124393,
                                               7: 0.117589893100097,
                                               8: 0.116618075801749,
                                               9: 0.0947521865889213,
                                               10: 0.0743440233236152,
                                               11: 0.0597667638483965,
                                               12: 0.0471331389698737,
                                               13: 0.0340136054421769,
                                               14: 0.0291545189504373,
                                               15: 0.0252672497570457,
                                               16: 0.0262390670553936,
                                               17: 0.0301263362487852,
                                               18: 0.0388726919339164,
                                               19: 0.0422740524781341,
                                               20: 0.0422740524781341,
                                               21: 0.0417881438289602,
                                               22: 0.0408163265306122,
                                               23: 0.0291545189504373,
                                               24: 0.0208940719144801}

        self.probability_function_sink = {1: 0.0141643059490085,
                                          2: 0.00679886685552408,
                                          3: 0.00509915014164306,
                                          4: 0.00509915014164306,
                                          5: 0.00679886685552408,
                                          6: 0.0181303116147309,
                                          7: 0.0424929178470255,
                                          8: 0.0623229461756374,
                                          9: 0.0657223796033994,
                                          10: 0.0617563739376771,
                                          11: 0.0543909348441926,
                                          12: 0.0498583569405099,
                                          13: 0.0487252124645892,
                                          14: 0.0453257790368272,
                                          15: 0.0413597733711048,
                                          16: 0.0430594900849858,
                                          17: 0.0481586402266289,
                                          18: 0.0651558073654391,
                                          19: 0.0747875354107649,
                                          20: 0.0691218130311615,
                                          21: 0.056657223796034,
                                          22: 0.0481586402266289,
                                          23: 0.0396600566572238,
                                          24: 0.0271954674220963}

    def create_random_starting_times(self, probability_function):
        # plot the probability function
        plt.plot(np.arange(24), probability_function.values())
        plt.title("probability function throughout the day for bathing")
        plt.grid()
        plt.show()

        # create yearly probability distribution
        df_probability_distribution = self.TimeStructure.loc[:, ["ID_DayHour", "ID_Hour"]]
        # map daily probability to each hour in the year
        df_probability_distribution.loc[:, "daily probability"] = df_probability_distribution.loc[:, "ID_DayHour"].map(
            probability_function)
        # create column with cumulative probability
        for i in range(len(df_probability_distribution)):
            if i == 0:
                df_probability_distribution.loc[i, "cumulative probability"] = df_probability_distribution.loc[
                                                                                   i, "daily probability"] / 365
            else:
                df_probability_distribution.loc[i, "cumulative probability"] = df_probability_distribution.loc[
                                                                                   i, "daily probability"] / 365 + \
                                                                               df_probability_distribution.loc[
                                                                                   i - 1, "cumulative probability"]

        # create vector with random starting times for taking a bath:
        random_vector = np.random.rand(self.bathing_cycles)
        # finding the index which has the closest probability to the random probability value:
        start_hours = np.zeros(shape=random_vector.shape)
        for i, number in enumerate(random_vector):
            start_hours[i] = df_probability_distribution.iloc[
                             (df_probability_distribution["cumulative probability"] - number).abs().argsort()[
                                 0], :].ID_Hour

        return start_hours

    def run(self):
        bathing_start_hours = self.create_random_starting_times(self.probability_function_bathing)
        showering_start_hours = self.create_random_starting_times(self.probability_function_showering)
        sink_start_hours = self.create_random_starting_times(self.probability_function_sink)


if __name__ == "__main__":
    DHW_probility().run()

    pass
