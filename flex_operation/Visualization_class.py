import pandas as pd
import sqlalchemy.exc
import sqlite3
from flex_operation.scenario import OperationScenario
from flex_operation.model_opt import OptInstance, OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.data_collector import RefDataCollector, OptDataCollector
from flex.db import DB
from config import cfg
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
            "Reference_hourly",
            "Reference_yearly",
            "Optimization_hourly",
            "Optimization_yearly",
        ]
        # delete the rows in case one of them is saved (eg. optimization is not here but reference is)
        for table_name in result_table_names:
            try:
                DB(config=cfg).delete_row_from_table(
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
        OptDataCollector(opt_model, self.scenario.scenario_id, cfg).run()

        ref_model = RefOperationModel(self.scenario).solve()

        # save results to db
        RefDataCollector(ref_model, self.scenario.scenario_id, cfg).run()

    def read_hourly_results(
        self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # check if scenario id is in results, if yes, load them instead of calculating them:
        hourly_results_reference_df = DB(cfg).read_parquet(
            table_name=OperationTable.ResultRefHour,
            scenario_ID=self.scenario.scenario_id
        )
        yearly_results_reference_df = DB(cfg).read_dataframe(
            table_name=OperationTable.ResultRefYear,
            filter={"ID_Scenario": self.scenario.scenario_id}
        )

        hourly_results_optimization_df = DB(cfg).read_parquet(
            table_name=OperationTable.ResultOptHour,
            scenario_ID=self.scenario.scenario_id,
        )
        yearly_results_optimization_df = DB(cfg).read_dataframe(
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
            else:
                return (
                    hourly_results_reference_df,
                    yearly_results_reference_df,
                    hourly_results_optimization_df,
                    yearly_results_optimization_df,
                )

        # if table doesn't exist:
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as error:
            print(error)
            print("creating the tables...")
            self.calculate_single_results()
        # ---------------------------------------------------------------------------------------------------------
        (
            hourly_results_reference_df,
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
