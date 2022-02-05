import sqlite3

import sqlalchemy.exc

from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel
from _Refactor.models.operation.ref import RefOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector
from _Refactor.basic.db import DB
import _Refactor.basic.config as config

import pandas as pd
import numpy as np
import plotly.express as px


# -----------------------------------------------------------------------------------------------------------
def show_yearly_comparison_of_SEMS_reference(yearly_results_optimization_df: pd.DataFrame,
                                             yearly_results_reference_df: pd.DataFrame):
    """
    this function creates a plotly bar chart that instantly shows the differences in the yearly results. When bars
    have value = 1 they are the same in both modes, if they are both 0 they are not being used.
    """
    yearly_df = pd.concat([yearly_results_optimization_df, yearly_results_reference_df], axis=0)
    yearly_df.index = ["SEMS", "reference"]
    # normalize (min max normalization) the data so differences can be seen in the bar plot: min = Zero
    # columns not used will be nan and columns of equal value will be 1
    yearly_df_normalized = (yearly_df - 0) / (yearly_df.max() - 0)
    yearly_df_normalized = yearly_df_normalized.fillna(0)  # make nan to zero
    # melt the dataframe so it is easily usable for plotly
    yearly_df_plotly = pd.melt(yearly_df_normalized.T.reset_index(), id_vars="index", value_vars=["SEMS", "reference"])
    # this plot is supposed to instantly show differences in input as well as output parameters:
    fig = px.bar(data_frame=yearly_df_plotly, x="index", y="value", color="variable", barmode="group")
    fig.show()


# ----------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # create scenario:
    scenario = AbstractScenario(scenario_id=0)
    try:
        # check if scenario id is in results, if yes, load them instead of calculating them:
        hourly_results_reference_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Reference_hourly",
            **{"scenario_id": 0})
        yearly_results_reference_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Reference_yearly",
            **{"scenario_id": 0})

        hourly_results_optimization_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Optimization_hourly",
            **{"scenario_id": 0})
        yearly_results_optimization_df = DB(connection=config.results_connection).read_dataframe(
            table_name="Optimization_yearly",
            **{"scenario_id": 0})
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as error:
        print(error)
        # list of result tables:
        result_table_names = ["Reference_hourly", "Reference_yearly", "Optimization_hourly", "Optimization_yearly"]
        # delete the rows in case one of them is saved (eg. optimization is not here but reference is)
        for table_name in result_table_names:
            try:
                DB(connection=config.results_connection).delete_row_from_table(table_name=table_name,
                                                                           column_name_plus_value={"scenario_id": 0})
            except sqlalchemy.exc.OperationalError:
                continue

        # calculate the results and save them
        optimization_model = OptOperationModel(scenario)
        # solve model
        solved_instance = optimization_model.run()
        # datacollector
        hourly_results_optimization_df = OptimizationDataCollector(solved_instance,
                                                                   scenario.scenario_id).collect_optimization_results_hourly()
        yearly_results_optimization_df = OptimizationDataCollector(solved_instance,
                                                                   scenario.scenario_id).collect_optimization_results_yearly()
        # save results to db
        OptimizationDataCollector(solved_instance, scenario.scenario_id).save_hourly_results()
        OptimizationDataCollector(solved_instance, scenario.scenario_id).save_yearly_results()

        reference_model = RefOperationModel(scenario)
        reference_model.run()
        hourly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_hourly()
        yearly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_yearly()
        # save results to db
        ReferenceDataCollector(reference_model).save_yearly_results()
        ReferenceDataCollector(reference_model).save_hourly_results()
    # ---------------------------------------------------------------------------------------------------------

    show_yearly_comparison_of_SEMS_reference(yearly_results_optimization_df, yearly_results_reference_df)
