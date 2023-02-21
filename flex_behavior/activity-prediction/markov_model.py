import itertools
from collections import Counter
import numpy as np
import pandas as pd
import os


class MarkovModel:
    def __init__(self, n_activities):
        self.n_activities = n_activities
        self.dirname = os.path.dirname(__file__)
        self.datadir = os.path.join(self.dirname, '..', '..', 'data')

    def compute_activity_changes(
            self,
            activities_before: list[int],
            activities_now: list[int],
    ):
        comparing_list = list(zip(activities_before, activities_now))
        # delete tuples from list, where activity has not changed
        comparing_list[:] = itertools.filterfalse(lambda x: x[0] == x[1], comparing_list)

        changes = dict(Counter(comparing_list))
        changes = {**{t: 0 for t in itertools.product(range(1, self.n_activities + 1), repeat=2)}, **changes}

        return changes

    def compute_change_probability(self, changes) -> "Dict[Tuple[int, int], double]":
        probabilities = {}
        for before in range(1, self.n_activities + 1):
            n_changes_from_before = sum(v for k, v in changes.items() if k[0] == before)
            if n_changes_from_before == 0:
                probabilities[(before, 1)] = 0
            else:
                probabilities[(before, 1)] = changes[(before, 1)] / n_changes_from_before
            for now in range(2, self.n_activities + 1):
                if n_changes_from_before == 0:
                    probabilities[(before, now)] = 0
                else:
                    probabilities[(before, now)] = probabilities[(before, now - 1)] + changes[
                        (before, now)] / n_changes_from_before
        return probabilities

    def print_probabilities_as_matrix(self, probabilities):
        matrix = np.zeros((self.n_activities, self.n_activities))
        for key in probabilities:
            matrix[key[0] - 1, key[1] - 1] = probabilities[key]
        print(matrix)

    def save_probabilities(self, prob_dict, filename: str, column_name):
        data = []
        for t in prob_dict:
            for key, value in prob_dict[t].items():
                data.append([t, key[0], key[1], value])
        df = pd.DataFrame(data, columns=column_name)
        filepath = os.path.join(self.datadir, 'output', filename)
        df.to_excel(filepath, index=False)

    def get_activity_list(self, df, timestep: int):
        activities_before = df[f't{timestep - 1}']
        activities_now = df[f't{timestep}']
        return activities_before, activities_now

    def build_change_matrix(self):
        filepath = os.path.join(self.datadir, 'input_behavior', 'BehaviorScenario_ActivityProfile.xlsx')
        df = pd.read_excel(filepath)
        # Filter ToD and Person Classification before or use already filtered data

        prob_dict = {}
        for t in range(2, 145):
            before, now = self.get_activity_list(df, t)
            changes = model.compute_activity_changes(before, now)
            probability = model.compute_change_probability(changes)
            prob_dict[t] = probability

        column_name = ['t', 'activity at t-1', 'activity at t', 'probability']
        model.save_probabilities(prob_dict, 'change-probability.xlsx', column_name)

    def build_duration_matrix(self):
        filepath = os.path.join(self.datadir, 'input_behavior', 'BehaviorScenario_ActivityProfile.xlsx')
        df = pd.read_excel(filepath)
        # Filter ToD and Person Classification before or use already filtered data
        df = df.drop(columns=['id_hhx', 'id_persx', 'id_tagx', 'monat', 'wtagfei'])
        values = df.values.tolist()

        durations = []
        for row in values:
            act_duration = self.count_occurences_in_list(row)
            durations.extend(act_duration)

        prob_dict = {}
        for t in range(1,145):
            time_filtered = filter(lambda x: x[0] == t, durations)
            prob_dict[t] = self.compute_duration_probability(list(time_filtered))

        column_name = ['t', 'activity', 'duration', 'probability']
        model.save_probabilities(prob_dict, 'duration-probability.xlsx', column_name)

    def compute_duration_probability(
            self,
            act_duration: "List[Tuple[int, int, int]]",  # (timestep, activity, duration)
    ) -> "Dict[Tuple[int, int], double]":  # (activity, duration): accumulated probability
        probabilities = {}  # keys = (activity, duration)
        for act in range(1, self.n_activities + 1):
            act_list = [x[2] for x in act_duration if x[1] == act]
            n_durations = len(act_list)
            if n_durations == 0:
                probabilities[(act, 1)] = 0
            else:
                probabilities[(act, 1)] = act_list.count(1) / n_durations
            for duration in range(2, 145):
                if n_durations == 0:
                    probabilities[(act, duration)] = 0
                else:
                    probabilities[(act, duration)] = probabilities[(act, duration-1)] + act_list.count(duration) / n_durations
        return probabilities

    def count_occurences_in_list(
            self,
            activities: "List[int]"
    ) -> "List[Tuple[int, int, int]]":  # Start Position, Item, Duration
        act_duration = []
        pos = 0
        for key, iter in itertools.groupby(activities):
            l = len(list(iter))
            act_duration.append((pos, key, l))
            pos += l
        return act_duration


if __name__ == '__main__':
    model = MarkovModel(17)
    model.build_change_matrix()
    model.build_duration_matrix()
