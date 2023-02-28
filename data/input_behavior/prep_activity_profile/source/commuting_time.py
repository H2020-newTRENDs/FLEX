import os
from typing import List
import numpy as np
import pandas as pd
from flex import kit


class CommutingTimer:
    def __init__(self):
        self.id_commuting = 15  # id_activity of commuting to work or school
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.input_dir = os.path.join(self.path, '../input')
        self.output_dir = os.path.join(self.path, '../output')
        self.data = pd.read_excel(os.path.join(self.output_dir, "Processed_ActivityProfile.xlsx"), engine="openpyxl")
        self.Output_CommutingTimer = "BehaviorScenario_CommutingTime.xlsx"

    def get_commuting_start_timeslot(self, activity_list: List[int], commuting_start_count: np.array):
        for index, id_activity in enumerate(activity_list):
            if id_activity == self.id_commuting:
                commuting_start_count[index] += 1
                break
        return commuting_start_count

    def get_commuting_end_timeslot(self, activity_list: List[int], commuting_end_count):
        for index, id_activity in reversed(list(enumerate(activity_list))):
            if id_activity == self.id_commuting:
                commuting_end_count[len(activity_list) - 1 - index] += 1
                break
        return commuting_end_count

    def commuting_time(self):
        person_types = self.data["id_person_type"].unique()
        day_types = self.data["id_day_type"].unique()
        commuting_count = []
        for id_person_type in person_types:
            for id_day_type in day_types:
                df = kit.filter_df(
                    self.data,
                    filter_dict={"id_person_type": id_person_type, "id_day_type": id_day_type}
                )
                commuting_start_count = np.zeros(144, )
                commuting_end_count = np.zeros(144, )
                for index, row in df.iterrows():
                    commuting_start_count = self.get_commuting_start_timeslot(row.copy().tolist()[2:], commuting_start_count)
                    commuting_end_count = self.get_commuting_end_timeslot(row.copy().tolist()[2:], commuting_end_count)
                commuting_count.append(pd.DataFrame({
                    'id_person_type': [id_person_type] * 144,
                    'id_day_type': [id_day_type] * 144,
                    't': range(1, 145),
                    'commuting_start_count': commuting_start_count,
                    'commuting_end_count': commuting_end_count,
                }))
        pd.concat(commuting_count, axis=0).to_excel(os.path.join(self.output_dir, self.Output_CommutingTimer), index=False)
