import pandas as pd
import sqlalchemy.exc
import sqlite3
from models.operation.scenario import OperationScenario
from models.operation.model_opt import OptOperationModel, OptimizationDataCollector
from models.operation.model_ref import RefOperationModel, ReferenceDataCollector
from basic.db import DB
import basic.config as config


class MotherVisualization:
    def __init__(self, scenario: OperationScenario):
        self.scenario = scenario
        self.hourly_results_reference_df, self.yearly_results_reference_df, self.hourly_results_optimization_df, self.yearly_results_optimization_df = self.fetch_results_for_specific_scenario_id()

    def create_header(self) -> str:
        return f"Scenario: {self.scenario.scenario_id}; \n " \
               f"AC: {int(self.scenario.space_cooling_technology.power)} W; \n " \
               f"Battery: {int(self.scenario.battery.capacity)} W; \n " \
               f"Building id: {self.scenario.building.ID_Building}; \n " \
               f"Boiler: {self.scenario.boiler.project_name}; \n " \
               f"DHW Tank: {self.scenario.hot_water_tank.size} l; \n " \
               f"PV: {self.scenario.pv.peak_power[0]} kWp; \n " \
               f"Heating Tank: {self.scenario.space_heating_tank.size} l"

    def calculate_single_results(self):
        # list of source tables:
        result_table_names = ["Reference_hourly", "Reference_yearly", "Optimization_hourly", "Optimization_yearly"]
        # delete the rows in case one of them is saved (eg. optimization is not here but reference is)
        for table_name in result_table_names:
            try:
                DB(connection=config.results_connection).delete_row_from_table(table_name=table_name,
                                                                               column_name_plus_value={
                                                                                   "scenario_id": self.scenario.scenario_id})
            except sqlalchemy.exc.OperationalError:
                continue

        # calculate the results and save them
        optimization_model = OptOperationModel(self.scenario)
        # solve model
        solved_instance = optimization_model.run()
        # datacollector save results to db
        OptimizationDataCollector(solved_instance, self.scenario.scenario_id).save_hourly_results()
        OptimizationDataCollector(solved_instance, self.scenario.scenario_id).save_yearly_results()

        reference_model = RefOperationModel(self.scenario)
        reference_model.run()

        # save results to db
        ReferenceDataCollector(reference_model).save_yearly_results()
        ReferenceDataCollector(reference_model).save_hourly_results()

    def read_hourly_results(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # check if scenario id is in results, if yes, load them instead of calculating them:
        hourly_results_reference_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Reference_hourly",
            **{"scenario_id": self.scenario.scenario_id})
        yearly_results_reference_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Reference_yearly",
            **{"scenario_id": self.scenario.scenario_id})

        hourly_results_optimization_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Optimization_hourly",
            **{"scenario_id": self.scenario.scenario_id})
        yearly_results_optimization_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Optimization_yearly",
            **{"scenario_id": self.scenario.scenario_id})
        return hourly_results_reference_df, yearly_results_reference_df, \
               hourly_results_optimization_df, yearly_results_optimization_df

    def fetch_results_for_specific_scenario_id(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # create scenario:
        try:
            # check if scenario id is in results, if yes, load them instead of calculating them:
            hourly_results_reference_df, yearly_results_reference_df, \
            hourly_results_optimization_df, yearly_results_optimization_df = self.read_hourly_results()
            # check if the tables are empty:
            if len(hourly_results_reference_df) == 0:
                print("creating the tables...")
                self.calculate_single_results()
            else:
                return hourly_results_reference_df, yearly_results_reference_df, \
                       hourly_results_optimization_df, yearly_results_optimization_df

        # if table doesn't exist:
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as error:
            print(error)
            print("creating the tables...")
            self.calculate_single_results()
        # ---------------------------------------------------------------------------------------------------------
        hourly_results_reference_df, yearly_results_reference_df, \
        hourly_results_optimization_df, yearly_results_optimization_df = self.read_hourly_results()
        return hourly_results_reference_df, yearly_results_reference_df, \
               hourly_results_optimization_df, yearly_results_optimization_df
