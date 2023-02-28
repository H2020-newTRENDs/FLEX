import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TouActivityProfileProcessor:

    def __init__(self):
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.input_dir = os.path.join(self.path, '../input')
        self.output_dir = os.path.join(self.path, '../output')
        # input tables
        self.Info_Activity = "Info_Activity.xlsx"
        self.Info_DayType = "Info_DayType.xlsx"
        self.Info_PersonType = "Info_PersonType.xlsx"
        self.ActivityRelation = "Info_ActivityRelation.xlsx"
        self.TOU_ActivityProfile = "ToU_ActivityProfile.xlsx"
        self.TOU_ActivityProfile_Converted = "ToU_ActivityProfile_Converted.xlsx"
        self.TOU_HouseholdInfo = "TOU_Households.xlsx"
        self.TOU_PersonInfo = "TOU_Persons.xlsx"
        # output tables
        self.Output_ActivityProfile = "Processed_ActivityProfile.xlsx"
        # columns
        self.ID_COLS = ["id_hhx", "id_persx", "id_tagx", "monat", "wtagfei"]
        self.TIME_COLS = [f'tb1_{i}' for i in range(1, 145)]
        self.prepare_activity_relation()
        self.prepare_filters()
        self.prepare_tables()

    def prepare_activity_relation(self):
        df = pd.read_excel(os.path.join(self.input_dir, self.ActivityRelation), engine="openpyxl")
        self.activity_relation = {}
        for index, row in df.iterrows():
            self.activity_relation[row["id_survey_activity"]] = row["id_activity"]

    def prepare_filters(self):
        """
        alterx: age range
        pc7: employment type: 1: full-time employed; 2: part-time employed
        ha1x: household size
        wtagfei: day of the week
        """
        self.person_types = {
            1: {'alterx': range(20, 66), 'pc7': [1]},
            2: {'alterx': range(20, 66), 'pc7': [2], 'ha1x': [4]},
            3: {'alterx': range(0, 20), 'ha6x': [3]}
        }

        self.day_types = {
            1: {'wtagfei': [1, 2, 3, 4]},
            2: {'wtagfei': [5]},
            3: {'wtagfei': [6]},
            4: {'wtagfei': [7]}
        }

    def prepare_tables(self):
        self.households = pd.read_excel(os.path.join(self.input_dir, self.TOU_HouseholdInfo), engine="openpyxl")
        self.persons = pd.read_excel(os.path.join(self.input_dir, self.TOU_PersonInfo), engine="openpyxl")
        self.activities = pd.read_excel(os.path.join(self.input_dir, self.Info_Activity), engine="openpyxl")

    def convert_tou_activity_profile(self):
        df = pd.read_excel(os.path.join(self.input_dir, self.TOU_ActivityProfile), engine="openpyxl")
        cols = self.ID_COLS + self.TIME_COLS
        df = df[cols].dropna(axis="index")
        for i, row in df.iterrows():
            for col in self.TIME_COLS:
                df.at[i, col] = self.activity_relation[row[col]]
        for col in self.TIME_COLS:
            new_col = f't{col.split("_")[1]}'
            df.rename(columns={col: new_col}, inplace=True)
        df.to_excel(os.path.join(self.output_dir, self.TOU_ActivityProfile_Converted), index=False)

    def filter_person(self, person_filter: dict) -> List[int]:
        df_person = self.persons[~self.persons.pb3.isin([11, 10, 9, 8, 7, 5, 4])]
        df_person = df_person[~df_person.soz.eq(4)]
        for key, value in person_filter.items():
            if key in df_person.columns:
                df_person = df_person[df_person[key].isin(value)]
        if 'ha1x' in list(person_filter.keys()):
            df_household = self.households[~self.households.ha1x.isin(person_filter["ha1x"])]
            df_person = df_person[df_person.id_hhx.isin(df_household['id_hhx'].tolist())]
        return df_person['id_persx'].tolist()

    def generate_input_activity_profile(self):
        df = pd.read_excel(os.path.join(self.output_dir, self.TOU_ActivityProfile_Converted), engine="openpyxl")
        df_out = pd.DataFrame()
        for person_key, person_filter in self.person_types.items():
            for day_key, day_filter in self.day_types.items():
                df_filtered = df[df["id_persx"].isin(self.filter_person(person_filter))]
                df_filtered = df_filtered[df_filtered["wtagfei"].isin(day_filter["wtagfei"])]
                df_filtered.drop(columns=['id_hhx', 'id_persx', 'id_tagx', 'monat', "wtagfei"], inplace=True)
                df_filtered.insert(loc=0, column='id_person_type', value=person_key)
                df_filtered.insert(loc=1, column='id_day_type', value=day_key)
                df_out = pd.concat([df_out, df_filtered], ignore_index=True)
        df_out.to_excel(os.path.join(self.output_dir, self.Output_ActivityProfile), index=False)

    def plot_activity_share(self):
        df = pd.read_excel(os.path.join(self.output_dir, self.Output_ActivityProfile), engine="openpyxl")
        for id_person_type in self.person_types.keys():
            for id_day_type in self.day_types.keys():
                df_pd = df.loc[(df["id_person_type"] == id_person_type) & (df["id_day_type"] == id_day_type)]
                fig_name = f'P{id_person_type}D{id_day_type}'
                data = df_pd.to_numpy()
                labels = self.activities['name'].tolist()

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
                filename = os.path.join(self.output_dir, 'figure', f'activity_share_{fig_name}.png')
                fig.savefig(filename, bbox_inches="tight")


