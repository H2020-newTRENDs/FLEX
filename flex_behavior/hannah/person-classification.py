import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PersonClassification:

    def __init__(self):
        self.dirname = os.path.dirname(__file__)
        self.datadir = os.path.join(self.dirname, '../..', 'data')

    def person_filtering(
            self,
            person_filter: {str, list[int]},
            household_filter: {str, list[int]},
            eval_household: bool = False,
            tod_filter: {str, list[int]} = {'wtagfei': range(1, 8)},
    ):
        person_list = self.filter_person_type(person_filter, household_filter, eval_household)

        # get activity profile for filtered persons
        filepath = os.path.join(self.datadir, 'input_behavior', 'BehaviorScenario_ActivityProfile.xlsx')
        df = pd.read_excel(filepath)
        df = df[df.id_persx.isin(person_list)]

        for key, value in tod_filter.items():
            df = df[df[key].isin(value)]

        return df

    def save_excel(self, df, output_filename: str, ):
        filepath = os.path.join(self.datadir, 'output', output_filename)
        df.to_excel(filepath, index=False)

    def filter_household_type(self, household_filter: {str, list[int]}) -> List[int]:
        filepath = os.path.join(self.datadir, 'input_behavior', 'household_info.xlsx')
        df_household = pd.read_excel(filepath)

        for key, value in household_filter.items():
            df_household = df_household[df_household[key].isin(value)]

        household_list = df_household['id_hhx'].values.tolist()
        return household_list

    def filter_person_type(self, person_filter, household_filter, eval_household) -> List[int]:
        filepath = os.path.join(self.datadir, 'input_behavior', 'person_info.xlsx')
        df_person = pd.read_excel(filepath)

        if eval_household:
            household_list = self.filter_household_type(household_filter)
            df_person = df_person[df_person.id_hhx.isin(household_list)]

        # filter unemployed
        df_person = df_person[~df_person.pb3.isin([11, 10, 9, 8, 7, 5, 4])]
        df_person = df_person[~df_person.soz.eq(4)]

        for key, value in person_filter.items():
            df_person = df_person[df_person[key].isin(value)]
        person_list = df_person['id_persx'].values.tolist()
        return person_list

    def plot_activity_share(
            self,
            data,  # 2-d numpy array
            labels,
            fig_name,
    ):
        data_len = np.shape(data)[0]
        occ = []  # count occurrences in column of np array
        for i in range(len(labels)):
            occ.append(np.count_nonzero(data == i + 1, axis=0))
            occ[i] = [x / data_len for x in occ[i]]

        colors = sns.color_palette("Spectral", len(labels)).as_hex()
        fig = plt.figure(figsize=(20, 7.417))
        plt.stackplot(range(len(data[0])), occ, labels=labels, colors=colors)
        plt.title(f'Activity share in {fig_name}', fontsize=20)
        plt.xlim(0, 143)
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        filename = os.path.join(self.datadir, 'figure', f'activity_share_{fig_name}.png')
        fig.savefig(filename, bbox_inches="tight")

    def calculate_activity_share(
            self,
            filename: str,
    ):
        filepath = os.path.join(self.datadir, 'output', filename)
        df = pd.read_excel(filepath)
        df = df.drop(columns=['id_hhx', 'id_persx', 'id_tagx', 'monat', 'wtagfei'])
        arr = df.to_numpy()

        filepath = os.path.join(self.datadir, 'input_behavior', 'BehaviorScenario_ActivityInfo.xlsx')
        activities = pd.read_excel(filepath)['name'].values.tolist()

        return arr, activities


if __name__ == '__main__':
    PC = PersonClassification()

    person_types = {
        'p_1': {'alterx': range(20, 39), 'pc7': [1]},  # employed full time
        'p_2': {'alterx': range(20, 66), 'pc7': [1]},  # employed full time
        'p_3_1': {'alterx': range(39, 60), 'pc7': [1], 'ha6x': [1]},  # employed full time, main income
        'p_3_2': {'alterx': range(39, 60), 'pc7': [2]},  # employed part-time
        'p_3_34': {'alterx': range(0, 20), 'ha6x': [3]}  # child
    }

    household_types = {
        'h_1': {'ha1x': [1]},
        'h_2': {'ha1x': [2]},
        'h_3': {'ha1x': [4]}
    }

    type_of_days = {'week': {'wtagfei': range(1, 5)},
                    'friday': {'wtagfei': [5]},
                    'saturday': {'wtagfei': [6]},
                    'sunday': {'wtagfei': [7]}}

    person_filter = person_types['p_1']
    household_filter = household_types['h_1']
    tod_filter = type_of_days['week']

    df = PC.person_filtering(
        person_filter=person_filter,
        household_filter=household_filter,
        eval_household=True,
        tod_filter=tod_filter,
    )

    PC.save_excel(df, output_filename='filtered_activity_profile.xlsx', )

    # data, labels = PC.calculate_activity_share('ActivityProfile_HH_1.xlsx')
    # PC.plot_activity_share(data, labels, 'HH_1')
