import pandas as pd
import sqlalchemy.exc
import sqlite3
import os
import sys

# Get the absolute path of the directory two levels up
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this directory to sys.path
sys.path.insert(0, two_levels_up)
from flex_operation.scenario import OperationScenario
from flex_operation.model_opt import OptInstance, OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.data_collector import RefDataCollector, OptDataCollector
from flex.db import create_db_conn

from flex_operation.constants import OperationTable


class MotherVisualization:
    def __init__(self, scenario: OperationScenario):
        self.scenario = scenario
        (
            self.hourly_results_reference_df,
            self.yearly_results_reference_df,
            self.hourly_results_optimization_df,
            self.yearly_results_optimization_df,
        ) = self.fetch_results_for_specific_scenario_id()

    def create_header(self) -> str:
        return (
            f"Scenario: {self.scenario.scenario_id}; \n "
            f"AC: {int(self.scenario.space_cooling_technology.power)} W; \n "
            f"Battery: {int(self.scenario.battery.capacity)} W; \n "
            f"Building id: {self.scenario.component_scenario_ids['ID_Building']}; \n "
            f"Boiler: {self.scenario.boiler.type}; \n "
            f"DHW Tank: {self.scenario.hot_water_tank.size} l; \n "
            f"PV: {self.scenario.pv.size} kWp; \n "
            f"Heating Tank: {self.scenario.space_heating_tank.size} l"
        )

    def calculate_single_results(self):
        # list of source tables:
        result_table_names = [
            "Reference_yearly",
            "Optimization_yearly",
        ]
        # delete the rows in case one of them is saved (eg. optimization is not here but reference is)
        for table_name in result_table_names:
            try:
                create_db_conn(self.scenario.config).delete_row_from_table(
                    table_name=table_name,
                    column_name_plus_value={"ID_Scenario": self.scenario.scenario_id},
                )
            except sqlalchemy.exc.OperationalError:
                continue

        # calculate the results and save them
        hp_instance = OptInstance().create_instance()
        # solve model
        opt_model = OptOperationModel(self.scenario).solve(hp_instance)
        # datacollector save results to db
        OptDataCollector(model=opt_model, scenario_id=self.scenario.scenario_id, config=self.scenario.config).run()

        ref_model = RefOperationModel(self.scenario).solve()

        # save results to db
        RefDataCollector(model=ref_model, scenario_id=self.scenario.scenario_id, config=self.scenario.config).run()

    def read_hourly_results(
        self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        db = create_db_conn(self.scenario.config)
        # check if scenario id is in results, if yes, load them instead of calculating them:
        hourly_results_reference_df = db.read_parquet(
            folder=self.scenario.config.output,
            table_name=OperationTable.ResultRefHour,
            scenario_ID=self.scenario.scenario_id
        )
        yearly_results_reference_df = db.read_dataframe(
            table_name=OperationTable.ResultRefYear,
            filter={"ID_Scenario": self.scenario.scenario_id}
        )

        hourly_results_optimization_df = db.read_parquet(
            folder=self.scenario.config.output,
            table_name=OperationTable.ResultOptHour,
            scenario_ID=self.scenario.scenario_id,
        )
        yearly_results_optimization_df = db.read_dataframe(
            table_name=OperationTable.ResultOptYear,
            filter={"ID_Scenario": self.scenario.scenario_id},
        )
        return (
            hourly_results_reference_df,
            yearly_results_reference_df,
            hourly_results_optimization_df,
            yearly_results_optimization_df,
        )

    def fetch_results_for_specific_scenario_id(
        self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # create scenario:
        try:
            # check if scenario id is in results, if yes, load them instead of calculating them:
            (
                hourly_results_reference_df,
                yearly_results_reference_df,
                hourly_results_optimization_df,
                yearly_results_optimization_df,
            ) = self.read_hourly_results()
            # check if the tables are empty:
            if len(hourly_results_reference_df) == 0:
                print("creating the tables...")
                self.calculate_single_results()

        # if table doesn't exist:
        except:
            print("creating the tables...")
            self.calculate_single_results()
        # ---------------------------------------------------------------------------------------------------------
        (hourly_results_reference_df,
            yearly_results_reference_df,
            hourly_results_optimization_df,
            yearly_results_optimization_df,
            ) = self.read_hourly_results()

        return (
            hourly_results_reference_df,
            yearly_results_reference_df,
            hourly_results_optimization_df,
            yearly_results_optimization_df,
        )
