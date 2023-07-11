import pandas as pd
import sqlalchemy.exc
import sqlite3
from flex_operation.scenario import OperationScenario, MotherOperationScenario
from flex_operation.model_opt import OptInstance, OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.data_collector import RefDataCollector, OptDataCollector
from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable


class MotherVisualization:
    def __init__(self, scenario: OperationScenario, cfg: "Config"):
        self.scenario = scenario
        self.cfg = cfg
        self.db = DB(config=self.cfg)
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
            f"Boiler: {self.scenario.boiler.power_max}; \n "
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
                self.db.delete_row_from_table(
                    table_name=table_name,
                    column_name_plus_value={"ID_Scenario": self.scenario.scenario_id},
                )
            except sqlalchemy.exc.OperationalError:
                continue

        # calculate the results and save them
        hp_instance = OptInstance().create_instance()
        # solve model
        opt_model = OptOperationModel(self.scenario).solve(hp_instance)
        # data collector save results to db
        OptDataCollector(model=opt_model, scenario_id=self.scenario.scenario_id,
                         config=self.cfg, save_hour_results=True).run()
        ref_model = RefOperationModel(self.scenario).solve()

        # save results to db
        RefDataCollector(model=ref_model, scenario_id=self.scenario.scenario_id,
                         config=self.cfg, save_hour_results=True).run()

    def read_hourly_results(
            self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # check if scenario id is in results, if yes, load them instead of calculating them:
        hourly_results_reference_df = self.db.read_parquet(table_name=OperationTable.ResultRefHour,
                                                           scenario_ID=self.scenario.scenario_id)
        yearly_results_reference_df = self.db.read_dataframe(table_name=OperationTable.ResultRefYear,
                                                             filter={"ID_Scenario": self.scenario.scenario_id})

        hourly_results_optimization_df = self.db.read_parquet(table_name=OperationTable.ResultOptHour,
                                                              scenario_ID=self.scenario.scenario_id)
        yearly_results_optimization_df = self.db.read_dataframe(table_name=OperationTable.ResultOptYear,
                                                                filter={"ID_Scenario": self.scenario.scenario_id})
        return (
            hourly_results_reference_df,
            yearly_results_reference_df,
            hourly_results_optimization_df,
            yearly_results_optimization_df,
        )

    def fetch_results_for_specific_scenario_id(
            self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # check if results exist:
        if self.db.if_parquet_exists(table_name=OperationTable.ResultRefHour,
                                     scenario_ID=self.scenario.scenario_id) \
                and self.db.if_parquet_exists(table_name=OperationTable.ResultOptHour,
                                              scenario_ID=self.scenario.scenario_id):
            pass  # results are there and will get returned
        else:
            print("creating the tables...")
            self.calculate_single_results()

        # load results:
        return self.read_hourly_results()
