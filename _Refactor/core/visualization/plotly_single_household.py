from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel
from _Refactor.models.operation.ref import RefOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector

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

    optimization_model = OptOperationModel(scenario)
    # solve model
    solved_instance = optimization_model.run()
    # datacollector
    hourly_results_optimization_df = OptimizationDataCollector().collect_optimization_results_hourly(solved_instance)
    yearly_results_optimization_df = OptimizationDataCollector().collect_optimization_results_yearly(solved_instance)

    reference_model = RefOperationModel(scenario)
    reference_model.run()
    hourly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_hourly()
    yearly_results_reference_df = ReferenceDataCollector(reference_model).collect_reference_results_yearly()
    # ---------------------------------------------------------------------------------------------------------

    show_yearly_comparison_of_SEMS_reference(yearly_results_optimization_df, yearly_results_reference_df)


