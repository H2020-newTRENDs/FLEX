from typing import TYPE_CHECKING, Type
import os

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

    def __init__(self, config: "Config", plotter_cls: Type["Plotter"] = BehaviorPlotter, scenario_id = 1):
        self.db = create_db_conn(config)
        self.output_folder = config.output
        self.output_folder_figures = config.fig
        self.plotter = plotter_cls(config)
        self.scenario = BehaviorScenario(scenario_id, config=config)
        self.config = config

    def plot_household_profiles(self):
        df = self.db.read_dataframe(BehaviorTable.HouseholdProfiles)
        profiles = df.loc[df["id_scenario"] == self.scenario.scenario_id]
        #profiles['daytype'] = profiles['hour'].apply(lambda x: self.scenario.get_daytype_from_hour(x))
        #profiles['time'] = profiles['hour'].apply(lambda x: self.scenario.get_time_from_hour(x))

        obj = profiles.groupby(['daytype', 'time']).mean().reset_index()
        for daytype in obj['daytype'].unique():
            electricity = obj[obj['daytype'] == daytype]['electricity_demand'].to_list()
            hot_water = obj[obj['daytype'] == daytype]['hotwater_demand'].to_list()
            plt.plot(range(24), electricity, label='electirity_demand')
            plt.plot(range(24), hot_water, label='hot_water')
            plt.legend()
            plt.title('daytype: ' + str(daytype))
            plt.savefig(os.path.join(self.config.fig, f"Behavior_Household_Profiles_daytype_{daytype}.png"))
            plt.close()

    def plot_activity_share(self):
        df = self.db.read_dataframe(BehaviorTable.PersonProfiles)
        df.drop(['activity_p2s0', 'id_technology_p2s0', 'electricity_p2s0', 'hotwater_p2s0'], axis=1)
        df.drop(['activity_p3s0', 'id_technology_p3s0', 'electricity_p3s0', 'hotwater_p3s0'], axis=1)
        df['daytype'] = [self.scenario.get_daytype_from_10_min(index) for index in range(len(df))]
        df['time'] = [self.scenario.get_time_from_10_min(index) for index in range(len(df))]

        labels_activity = self.scenario.activities
        labels_technology = self.scenario.technologies
        occ_dict_activity = self.count_occurences_in_df(df, 'activity_p1s0', labels_activity)
        occ_dict_technology = self.count_occurences_in_df(df, 'id_technology_p1s0', labels_technology)

        for daytype in df['daytype'].unique():
            occ_activity = occ_dict_activity[daytype]
            self.plot_stackplot(occ_activity, labels_activity, figname=f'generated_activity_share_d_{daytype}')
            occ_technology = occ_dict_technology[daytype]
            self.plot_stackplot(occ_technology, labels_technology, figname=f'generated_technology_share_d_{daytype}')

    def plot_stackplot(self, occ, label, figname):
        colors = sns.color_palette("Spectral", len(label)).as_hex()
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
        self.plot_activity_share()
