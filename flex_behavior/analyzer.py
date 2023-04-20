from typing import TYPE_CHECKING, Type
import os
from datetime import datetime
import numpy as np
import pandas as pd

from flex import kit
from flex.db import create_db_conn
from flex_behavior.plotter import BehaviorPlotter
from flex_behavior.constants import BehaviorTable
from flex_behavior.scenario import BehaviorScenario

from matplotlib import pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from flex.config import Config
    from flex.plotter import Plotter

logger = kit.get_logger(__name__)


class BehaviorAnalyzer:

    def __init__(self, config: "Config", plotter_cls: Type["Plotter"] = BehaviorPlotter, scenario_id: int = 1):
        self.db = create_db_conn(config)
        self.output_folder = config.output
        self.output_folder_figures = config.fig
        self.plotter = plotter_cls(config)
        self.scenario = BehaviorScenario(scenario_id, config=config)
        self.config = config

    def get_household_generated_profile_average(self):
        df = self.db.read_dataframe(BehaviorTable.HouseholdProfiles)
        profiles = df.loc[df["id_scenario"] == self.scenario.scenario_id]
        obj = profiles.groupby(['daytype', 'time']).mean().reset_index()
        obj.to_csv("test.csv", index=False)
        return obj

    def plot_household_profiles(self):
        df = self.get_household_generated_profile_average()
        for daytype in df['daytype'].unique():
            electricity = df[df['daytype'] == daytype]['electricity_demand'].to_list()
            hot_water = df[df['daytype'] == daytype]['hotwater_demand'].to_list()
            plt.plot(range(24), electricity, label='electricity_demand')
            plt.plot(range(24), hot_water, label='hot_water')
            plt.legend()
            plt.title('daytype: ' + str(daytype))
            plt.savefig(os.path.join(self.config.fig, f"Behavior_Household_Profiles_daytype_{daytype}.png"))
            plt.close()

    @staticmethod
    def get_daytype_from_str(date_string):
        day_type = {
            1: 1,  # Monday
            2: 1,  # Tuesday
            3: 1,  # Wednesday
            4: 1,  # Thursday
            5: 2,  # Friday
            6: 3,  # Saturday
            0: 4,  # Sunday --> weekday % 7 = 0
        }

        date_time_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return day_type[date_time_obj.isoweekday() % 7]

    @staticmethod
    def get_time_from_str(date_string):
        date_time_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return date_time_obj.hour + 1

    def plot_electricity_profile_comparison(self):
        hh_profiles = pd.read_csv(os.path.join(self.config.input_behavior, f'household_profiles.csv'))
        hh_info = pd.read_csv(os.path.join(self.config.input_behavior, f'household_information.csv'))
        single_hh = list(hh_info.loc[
                             (hh_info['numberOfPeople'] == '1') &
                             (hh_info['ELECTRIC_VEHICLE'] == 0) &
                             (hh_info['heatingTypes'] != "HEAT_PUMP")
                         ]['userId'])
        real_profiles = hh_profiles[hh_profiles['userId'].isin(single_hh)]
        real_profiles['daytype'] = real_profiles['date'].apply(lambda x: self.get_daytype_from_str(x))
        real_profiles['time'] = real_profiles['date'].apply(lambda x: self.get_time_from_str(x))
        generated_profile_average = self.get_household_generated_profile_average()
        for daytype in generated_profile_average['daytype'].unique():
            real_profiles_mat = np.reshape(
                real_profiles[real_profiles['daytype'] == daytype]['power'].to_numpy(),  # TODO: bug found --> not exactly number of 24-hours
                (-1, 24)
            )
            for index, electricity_profile in enumerate(real_profiles_mat):
                plt.plot(range(1, 25), electricity_profile)
            # TODO: also plot the mean of the empirical profiles
            plt.plot(
                range(24),
                generated_profile_average[generated_profile_average['daytype'] == daytype]['electricity_demand'],
                label=f'generated_profile_average'
            )
            plt.legend()
            plt.title('daytype: ' + str(daytype))
            filename = os.path.join(self.output_folder_figures, f'profile_comparison_D{daytype}.png')
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_activity_share(self):
        df = self.db.read_dataframe(BehaviorTable.PersonProfiles)

        """
        person_type selection
        """
        person_type = 1
        # df.drop(['activity_p1s0', 'id_technology_p1s0', 'electricity_p1s0', 'hotwater_p1s0'], axis=1)
        df.drop(['activity_p2s0', 'id_technology_p2s0', 'electricity_p2s0', 'hotwater_p2s0'], axis=1)
        df.drop(['activity_p3s0', 'id_technology_p3s0', 'electricity_p3s0', 'hotwater_p3s0'], axis=1)

        """
        time selection
        """
        df['daytype'] = [self.scenario.get_daytype_from_10_min(index) for index in range(len(df))]
        df['time'] = [self.scenario.get_time_from_10_min(index) for index in range(len(df))]

        labels_activity = self.scenario.activities
        labels_technology = self.scenario.technologies
        occ_dict_activity = self.count_occurences_in_df(df, f'activity_p{person_type}s0', labels_activity)
        occ_dict_technology = self.count_occurences_in_df(df, f'id_technology_p{person_type}s0', labels_technology)

        for daytype in df['daytype'].unique():
            occ_activity = occ_dict_activity[daytype]
            self.plot_stackplot(occ_activity, labels_activity, figname=f'generated_activity_share_p{person_type}d{daytype}')
            occ_technology = occ_dict_technology[daytype]
            self.plot_stackplot(occ_technology, labels_technology, figname=f'generated_technology_share_p{person_type}d{daytype}')

    def plot_stackplot(self, occ, label, figname):
        colors = sns.color_palette("Spectral", len(label)).as_hex()
        colors.reverse()  # reverse the order of the colors
        fig = plt.figure(figsize=(20, 7.417))
        plt.stackplot(range(144), occ, labels=label, colors=colors)
        plt.title(figname, fontsize=20)
        plt.xlim(0, 143)
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        filename = os.path.join(self.output_folder_figures, f'{figname}.png')
        fig.savefig(filename, bbox_inches="tight")

    def count_occurences_in_df(self, df, column_name, identities):
        df_occ = df.groupby(['daytype', 'time', column_name]).size().reset_index()
        occ_dict = {}
        for daytype in df['daytype'].unique():
            df_occ_daytype = df_occ.loc[df_occ['daytype'] == daytype]
            num_daytype_days = len(df.loc[df['daytype'] == daytype]) / 144
            occ = []  # count occurrences in column of np array
            for i in range(len(identities)):
                occ.append(144 * [0])

            for index, row in df_occ_daytype.iterrows():
                occ[row[column_name] - 1][row['time'] - 1] = row[0] / num_daytype_days
                # divide by number of days in daytype
            occ_dict[daytype] = occ
        return occ_dict

    def run(self):
        self.plot_household_profiles()
        # self.plot_activity_share()
        # self.plot_electricity_profile_comparison()
