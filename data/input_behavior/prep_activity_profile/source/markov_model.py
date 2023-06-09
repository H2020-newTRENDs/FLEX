import itertools
import os
from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd


class MarkovActivityProfileGenerator:
    def __init__(self):
        self.n_activities = 17  # There are 17 activities in Info_Activity.xlsx
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.input_dir = os.path.join(self.path, '../input')
        self.output_dir = os.path.join(self.path, '../output')
        self.data = pd.read_excel(os.path.join(self.output_dir, "Processed_ActivityProfile.xlsx"), engine="openpyxl")
        self.Output_ChangeProb = "BehaviorScenario_ActivityChangeProb.xlsx"
        self.Output_DurationProb = "BehaviorScenario_ActivityDurationProb.xlsx"

    def generate_activity_change_prob_matrix(self):
        person_types = self.data["id_person_type"].unique()
        day_types = self.data["id_day_type"].unique()
        change_prob_matrix = []
        for id_person_type in person_types:
            for id_day_type in day_types:
                df = self.data.loc[(self.data["id_person_type"] == id_person_type) & (self.data["id_day_type"] == id_day_type)]
                for t in range(2, 145):
                    activities_before, activities_now = self.get_activity_list(df, t)
                    changes = self.compute_activity_changes(activities_before, activities_now)
                    change_probability = self.compute_change_probability(changes)
                    for (id_activity_before, id_activity_now), (prob, n_case) in change_probability.items():
                        change_prob_matrix.append({
                            'id_person_type': id_person_type,
                            'id_day_type': id_day_type,
                            't': t,
                            'id_activity_before': id_activity_before,
                            'id_activity_now': id_activity_now,
                            'probability': prob,
                            'n_cases': n_case,
                        })
        pd.DataFrame(change_prob_matrix).to_excel(os.path.join(self.output_dir, self.Output_ChangeProb), index=False)

    @staticmethod
    def get_activity_list(df, timestep: int):
        activities_before = df[f't{timestep - 1}']
        activities_now = df[f't{timestep}']
        return activities_before, activities_now

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

    def compute_change_probability(self, changes) -> "Dict[Tuple[int, int], Tuple[float, int]]":
        change_probability = {}
        for id_activity_before in range(1, self.n_activities + 1):
            n_changes_from_before = sum(v for k, v in changes.items() if k[0] == id_activity_before)
            if n_changes_from_before == 0:
                continue
            for id_activity_now in range(1, self.n_activities + 1):
                prob = changes[(id_activity_before, id_activity_now)] / n_changes_from_before
                n_case = changes[(id_activity_before, id_activity_now)]
                if prob != 0:
                    change_probability[(id_activity_before, id_activity_now)] = (prob, n_case)
        return change_probability

    def generate_activity_duration_prob_matrix(self):
        person_types = self.data["id_person_type"].unique()
        day_types = self.data["id_day_type"].unique()
        duration_prob_matrix = []
        for id_person_type in person_types:
            for id_day_type in day_types:
                df = self.data.loc[(self.data["id_person_type"] == id_person_type) & (self.data["id_day_type"] == id_day_type)]
                values = df.drop(columns=["id_person_type", "id_day_type"]).values.tolist()
                activity_durations = []
                for row in values:
                    activity_durations.extend(self.count_occurrence_in_list(row))
                for t in range(1, 145):
                    activity_durations_filtered = filter(lambda x: x[0] == t, activity_durations)
                    duration_probability = self.compute_duration_probability(list(activity_durations_filtered))
                    for (id_activity, duration), (prob, n_case) in duration_probability.items():
                        duration_prob_matrix.append({
                            'id_person_type': id_person_type,
                            'id_day_type': id_day_type,
                            't': t,
                            'id_activity': id_activity,
                            'duration': duration,
                            'probability': prob,
                            'n_cases': n_case,
                        })
        pd.DataFrame(duration_prob_matrix).to_excel(os.path.join(self.output_dir, self.Output_DurationProb), index=False)

    @staticmethod
    def count_occurrence_in_list(activities: "List[int]") -> "List[Tuple[int, int, int]]":
        activity_durations = []
        pos = 1
        for key, activity_series in itertools.groupby(activities):
            duration = len(list(activity_series))
            activity_durations.append((pos, key, duration))  # starting_period, id_activity, duration
            pos += duration
        return activity_durations

    def compute_duration_probability(self, activity_durations: "List[Tuple[int, int, int]]") -> "Dict[Tuple[int, int], Tuple[float, int]]":
        probability = {}
        for id_activity in range(1, self.n_activities + 1):
            durations = [x[2] for x in activity_durations if x[1] == id_activity]
            n_durations = len(durations)
            if n_durations != 0:
                for duration in range(1, 145):
                    prob = durations.count(duration) / n_durations
                    n_cases = durations.count(duration)
                    if prob != 0:
                        probability[(id_activity, duration)] = (prob, n_cases)
        return probability
