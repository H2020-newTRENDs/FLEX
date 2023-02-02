import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PersonClassification:

    def person_filtering(
            self,
            person_filter,
            output_filename: str,
            household_filter: {str, list[int]},
            eval_household: bool = False,
    ):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, '..', 'data', 'input_behavior', 'person_info.xlsx')
        df = pd.read_excel(filepath)

        # filter unemployed
        df = df[~df.pb3.isin([11, 10, 9, 8, 7, 5, 4])]
        df = df[~df.soz.eq(4)]

        # filter household type
        if eval_household:
            filepath = os.path.join(dirname, '..', 'data', 'input_behavior', 'household_info.xlsx')
            df_household = pd.read_excel(filepath)

            for key, value in household_filter.items():
                df_household = df_household[df_household[key].isin(value)]

            household_list = df_household['id_hhx'].values.tolist()
            df = df[df.id_hhx.isin(household_list)]

        # filter person type
        df_person = df
        for key, value in person_filter.items():
            df_person = df_person[df_person[key].isin(value)]

        person_list = df_person['id_persx'].values.tolist()

        # get activity profile for filtered persons
        filepath = os.path.join(dirname, '..', 'data', 'input_behavior', 'BehaviorScenario_ActivityProfile.xlsx')
        df = pd.read_excel(filepath)
        df_person = df[df.id_persx.isin(person_list)]

        filepath = os.path.join(dirname, '..', 'data', 'output', output_filename)
        df_person.to_excel(filepath, index=False)

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

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '..', 'data', 'figure', f'activity_share_{fig_name}.png')
        fig.savefig(filename, bbox_inches="tight")

    def plot_activity_share_data_prep(
            self,
            filename: str,
            plot_title: str
    ):
        dirname = os.path.dirname(__file__)

        filepath = os.path.join(dirname, '..', 'data', 'output', filename)
        df = pd.read_excel(filepath)
        df = df.drop(columns=['id_hhx', 'id_persx', 'id_tagx', 'monat', 'wtagfei'])
        arr = df.to_numpy()

        filepath = os.path.join(dirname, '..', 'data', 'input_behavior', 'BehaviorScenario_ActivityInfo.xlsx')
        activities = pd.read_excel(filepath)['name'].values.tolist()

        self.plot_activity_share(arr, activities, plot_title)


if __name__ == '__main__':
    PC = PersonClassification()

    person_filter = {'alterx': range(20, 39), 'pc7': [1]}  # pc7: Employment type, 'ha6x': [1] role in household
    household_filer = {'ha1x': [1]}
    PC.person_filtering(
        person_filter=person_filter,
        output_filename='test.xlsx',
        household_filter=household_filer,
        eval_household=True
    )

    PC.plot_activity_share_data_prep('ActivityProfile_HH_1.xlsx', 'HH_1')
