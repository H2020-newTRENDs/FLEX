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

    def plot_household_profiles(self):
        df = self.db.read_dataframe(BehaviorTable.HouseholdProfiles)
        df = df.groupby(['id_scenario', 'daytype', 'time']).mean().reset_index()

        for day_type in df['daytype'].unique():
            fig_electricity = plt.figure()
            fig_hot_water = plt.figure()
            fig_occupancy = plt.figure()

            ax_electricity = fig_electricity.add_subplot(1, 1, 1)
            ax_hot_water = fig_hot_water.add_subplot(1, 1, 1)
            ax_occupancy = fig_occupancy.add_subplot(1, 1, 1)

            for scenario_id in range(1,10):
                profiles = df.loc[df["id_scenario"] == scenario_id]

                electricity = profiles[profiles['daytype'] == day_type]['electricity_demand'].to_list()
                hot_water = profiles[profiles['daytype'] == day_type]['hotwater_demand'].to_list()
                occupancy = profiles[profiles['daytype'] == day_type]['building_occupancy'].to_list()

                ax_electricity.plot(range(24), electricity, label=f'scenario{scenario_id}')
                ax_hot_water.plot(range(24), hot_water, label=f'scenario{scenario_id}')
                ax_occupancy.plot(range(24), occupancy, label=f'scenario{scenario_id}')

            self.save_plots_scenario_profiles(fig_electricity,
                                              ax_electricity,
                                              figtile=f'Electricity consumption at Daytype {day_type}',
                                              filename=f'electricity_consumption_d{day_type}.png')

            self.save_plots_scenario_profiles(fig_hot_water,
                                              ax_hot_water,
                                              figtile=f'Hot Water consumption at Daytype {day_type}',
                                              filename=f'hot_water_consumption_d{day_type}.png')

            self.save_plots_scenario_profiles(fig_occupancy,
                                              ax_occupancy,
                                              figtile=f'Building Occupancy at Daytype {day_type}',
                                              filename=f'building_occupandy_d{day_type}.png')

            plt.close()

    def save_plots_scenario_profiles(self, fig, axis, figtile: str, filename: str):
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0))
        fig.suptitle(figtile, fontsize=18)
        filename = os.path.join(self.output_folder_figures, filename)
        fig.savefig(filename, bbox_inches="tight")

    @staticmethod
    def get_daytype_from_str(date_string):
        day_type = {
            1: 1,  # Monday
            2: 1,  # Tuesday
            3: 1,  # Wednesday
            4: 1,  # Thursday
            5: 1,  # Friday
            6: 2,  # Saturday
            0: 2,  # Sunday --> weekday % 7 = 0
        }

        date_time_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return day_type[date_time_obj.isoweekday() % 7]

    @staticmethod
    def get_daytype_from_time(date_time):
        day_type = {
            1: 1,  # Monday
            2: 1,  # Tuesday
            3: 1,  # Wednesday
            4: 1,  # Thursday
            5: 1,  # Friday
            6: 2,  # Saturday
            7: 2,  # Sunday --> weekday % 7 = 0
        }
        return day_type[date_time.isoweekday()]

    @staticmethod
    def get_time_from_str(date_string):
        date_time_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return date_time_obj.hour + 1

    def get_real_profiles(self):
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
        group_by_obj = real_profiles.groupby(['daytype', 'time'])
        return group_by_obj

    def get_validating_profiles(self):
        hh_profile = pd.read_excel(os.path.join(self.config.input_behavior, f'WPuQ_household_profile.xlsx'))
        hh_profile['daytype'] = hh_profile['time'].apply(lambda x: self.get_daytype_from_time(x))
        hh_profile['daytime'] = hh_profile['time'].apply(lambda x: x.time())
        group_by_obj = hh_profile.groupby(['daytype', 'daytime'])
        return group_by_obj

    def plot_electricity_profile_comparison(self):
        generated_profile_average = self.get_household_generated_profile_average()
        real_profiles = self.get_validating_profiles()
        real_profiles_mean = real_profiles.mean().reset_index()
        real_profiles_std = real_profiles.std().reset_index()
        for daytype in real_profiles_mean['daytype'].unique():
            electricity_mean = real_profiles_mean[real_profiles_mean['daytype'] == daytype]['P_TOT'].to_list()
            electricity_std = real_profiles_std[real_profiles_std['daytype'] == daytype]['P_TOT'].to_list()
            fig, ax = plt.subplots(facecolor='white')  #

            ax.errorbar(range(24),
                        electricity_mean,
                        yerr=electricity_std, fmt='-o',
                        label=f'mean electirity_demand', color='blue')
            ax.set_xlabel("time", fontsize=14)
            ax.set_ylabel("Real Data Electricity Demand", color="blue", fontsize=14)

            ax2 = ax.twinx()
            ax2.plot(
                range(24),
                generated_profile_average[generated_profile_average['daytype'] == daytype]['electricity_demand'],
                label=f'generated_profile_average', color='red'
            )
            ax2.set_ylabel("Generated Electricity Demand", color="red", fontsize=14)
            fig.legend()
            fig.suptitle('daytype: ' + str(daytype))
            filename = os.path.join(self.output_folder_figures, f'profile_comparison_D{daytype}.png')
            fig.savefig(filename, bbox_inches="tight")
            plt.close()

    def plot_stackplot(self, axis, occ, label, axis_title):
        colors = sns.color_palette("Spectral", len(label)).as_hex()
        colors.reverse()  # reverse the order of the colors
        axis.stackplot(range(144), occ, labels=label, colors=colors)
        axis.set_xlim(0, 143)
        axis.set_ylim(0, 1)
        axis.set_title(axis_title)
        return axis

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

    def prepare_data_for_generated_activity_share(self, person_type):
        data = self.db.read_dataframe(BehaviorTable.PersonProfiles)
        data['daytype'] = [self.scenario.get_daytype_from_10_min(index) for index in range(len(data))]
        data['time'] = [self.scenario.get_time_from_10_min(index) for index in range(len(data))]

        labels_activity = self.scenario.activities

        occ_dict_activity = self.count_occurences_in_df(data, f'activity_p{person_type}s0', labels_activity)
        return occ_dict_activity, data['daytype'].unique()

    def prepare_data_for_original_activity_share(self, df, person_type, day_type):
        df_pd = df.loc[(df["id_person_type"] == person_type) & (df["id_day_type"] == day_type)]
        df_pd = df_pd.drop(["id_person_type", "id_day_type"], axis=1)
        data = df_pd.to_numpy()
        labels = self.scenario.activities

        data_len = np.shape(data)[0]
        occ = []  # count occurrences in column of np array
        for i in range(len(labels)):
            occ.append(np.count_nonzero(data == i + 1, axis=0))
            occ[i] = [x / data_len for x in occ[i]]

        return occ, labels

    def plot_activity_share(self, axis, occ_dict_activity, day_type: int):
        labels_activity = self.scenario.activities
        occ_activity = occ_dict_activity[day_type]
        return self.plot_stackplot(axis, occ_activity, labels_activity, axis_title='Generated Activity Share')

    def plot_technology_share(self):
        data = self.db.read_dataframe(BehaviorTable.PersonProfiles)
        data['daytype'] = [self.scenario.get_daytype_from_10_min(index) for index in range(len(data))]
        data['time'] = [self.scenario.get_time_from_10_min(index) for index in range(len(data))]

        labels_technology = self.scenario.technologies
        daytypes = data['daytype'].unique()
        for person_type in [1, 2, 3]:
            occ_dict_technology = self.count_occurences_in_df(data, f'id_technology_p{person_type}s0', labels_technology)
            for day_type in daytypes:
                occ_technology = occ_dict_technology[day_type]

                fig = plt.figure(figsize=(20, 7.417))
                ax = plt.subplot(1, 1, 1)
                ax = self.plot_stackplot(ax, occ_technology[0:34], labels_technology[0:34], axis_title=f'Technology Share for Person {person_type} and Daytype {day_type}')
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
                plt.tight_layout(rect=[0, 0, 0.75, 1])
                filename = os.path.join(self.output_folder_figures, f'technologyshare_p{person_type}_d{day_type}.png')
                fig.savefig(filename, bbox_inches="tight")
                plt.close()

    def compare_activity_share_generated_data(self):
        path = os.path.join(self.config.input_behavior, 'prep_activity_profile', 'output',
                            'Processed_ActivityProfile.xlsx')
        df_original = pd.read_excel(path, engine="openpyxl")

        for person_type in [1, 2, 3]:
            occ_dict_activity, daytypes = self.prepare_data_for_generated_activity_share(person_type)
            for day_type in daytypes:
                occ_original, labels = self.prepare_data_for_original_activity_share(df_original, person_type, day_type)
                fig = plt.figure(figsize=(20, 5))
                ax_generated = plt.subplot(1, 2, 1)
                ax_generated = self.plot_activity_share(ax_generated, occ_dict_activity, day_type)

                ax_original = plt.subplot(1, 2, 2)
                ax_original = self.plot_stackplot(ax_original, occ_original, labels, axis_title='Original Activity Share')

                handles, labels = ax_original.get_legend_handles_labels()
                plt.figlegend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0))
                plt.suptitle(f'Activity Share for Person {person_type} and Daytype {day_type}', fontsize= 18)
                filename = os.path.join(self.output_folder_figures, f'activityshare_p{person_type}_d{day_type}.png')
                fig.savefig(filename, bbox_inches="tight")
                plt.close()

    def run(self):
        self.plot_household_profiles()
        # self.compare_activity_share_generated_data()
        # self.plot_technology_share()

        #self.plot_electricity_profile_comparison()
