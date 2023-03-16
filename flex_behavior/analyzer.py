from typing import TYPE_CHECKING, Type
import os

from flex import kit
from flex.db import create_db_conn
from flex_behavior.plotter import BehaviorPlotter
from flex_behavior.constants import BehaviorTable
from flex_behavior.scenario import BehaviorScenario

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from flex.config import Config
    from flex.plotter import Plotter

logger = kit.get_logger(__name__)


class BehaviorAnalyzer:

    def __init__(self, config: "Config", plotter_cls: Type["Plotter"] = BehaviorPlotter, scenario_id = 1):
        self.db = create_db_conn(config)
        self.output_folder = config.output
        self.plotter = plotter_cls(config)
        self.scenario = BehaviorScenario(scenario_id, config=config)
        self.config = config

    def plot_household_profiles(self):
        df = self.db.read_dataframe(BehaviorTable.HouseholdProfiles)
        profiles = df.loc[df["id_scenario"] == self.scenario.scenario_id]
        profiles['daytype'] = profiles['hour'].apply(lambda x: self.scenario.get_daytype_from_hour(x))
        profiles['time'] = profiles['hour'].apply(lambda x: self.scenario.get_time_from_hour(x))

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

    def run(self):
        self.plot_household_profiles()

