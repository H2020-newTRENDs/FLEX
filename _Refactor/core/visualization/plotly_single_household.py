import sqlite3

import sqlalchemy.exc

from _Refactor.core.household.abstract_scenario import AbstractScenario
from _Refactor.models.operation.opt import OptOperationModel
from _Refactor.models.operation.ref import RefOperationModel
from _Refactor.models.operation.data_collector import OptimizationDataCollector, ReferenceDataCollector
from _Refactor.core.visualization.Visualization_class import MotherVisualization
from _Refactor.basic.db import DB
import _Refactor.basic.config as config

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# -----------------------------------------------------------------------------------------------------------
class PlotlyVisualize(MotherVisualization):

    def show_yearly_comparison_of_SEMS_reference(self) -> None:
        """
        this function creates a plotly bar chart that instantly shows the differences in the yearly results. When bars
        have value = 1 they are the same in both modes, if they are both 0 they are not being used.
        """
        yearly_df = pd.concat([self.yearly_results_optimization_df, self.yearly_results_reference_df], axis=0)
        yearly_df.index = ["SEMS", "reference"]
        # normalize (min max normalization) the data so differences can be seen in the bar plot: min = Zero
        # columns not used will be nan and columns of equal value will be 1
        yearly_df_normalized = (yearly_df - 0) / (yearly_df.max() - 0)
        yearly_df_normalized = yearly_df_normalized.fillna(0)  # make nan to zero
        # melt the dataframe so it is easily usable for plotly
        yearly_df_plotly = pd.melt(yearly_df_normalized.T.reset_index(), id_vars="index", value_vars=["SEMS", "reference"])
        # this plot is supposed to instantly show differences in input as well as output parameters:
        fig = px.bar(data_frame=yearly_df_plotly, x="index", y="value", color="variable", barmode="group")
        fig.update_layout(title_text=self.create_header())
        fig.show()

    def hourly_comparison_SEMS_reference(self) -> None:
        reference_df = self.hourly_results_reference_df
        optimization_df = self.hourly_results_optimization_df
        # check if both tables have same columns
        assert sorted(list(reference_df.columns)) == sorted(list(optimization_df.columns))
        # determine how many subplots are needed by excluding profiles that are zero in both modes
        for column_name in reference_df.columns:
            if (reference_df[column_name] == 0).all() and (optimization_df[column_name] == 0).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])
                continue
            # also exclude columns where all values are static (eg battery size) and the same:
            if (reference_df[column_name].to_numpy()[0] == reference_df[column_name].to_numpy()).all() and \
                    (optimization_df[column_name].to_numpy()[0] == optimization_df[column_name].to_numpy()).all():
                reference_df = reference_df.drop(columns=[column_name])
                optimization_df = optimization_df.drop(columns=[column_name])

        # count the columns which will be the number of subplots:
        column_number = len(list(reference_df.columns))
        # x-axis are the hours
        x_axis = np.arange(8760)
        # create plot
        fig = make_subplots(rows=column_number, cols=1,
                            subplot_titles=sorted(list(reference_df.columns)),
                            shared_xaxes=True)
        for i, column_name in enumerate(sorted(list(reference_df.columns))):
            fig.add_trace(go.Scatter(x=x_axis, y=reference_df[column_name], name="reference"), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=x_axis, y=optimization_df[column_name], name="SEMS"), row=i + 1, col=1)

        fig.update_layout(height=400 * column_number, width=1600, title_text=self.create_header())
        fig.show()


if __name__ == "__main__":

    # create scenario:
    scenario_id = 0
    scenario = AbstractScenario(scenario_id=scenario_id)
    plotly_visualization = PlotlyVisualize(scenario=scenario)
    plotly_visualization.show_yearly_comparison_of_SEMS_reference()
    plotly_visualization.hourly_comparison_SEMS_reference()
    # ---------------------------------------------------------------------------------------------------------

