import pandas as pd
import plotly.express as px
import plotly.express.colors as colors
import plotly.graph_objects as go
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import numpy as np
import math
from dash_visualization import ids, data_extraction, main


def load_fat_df():
    # find all possible column names in the yearly results that are not zero
    fat_df = pd.DataFrame()
    for country in main.COUNTRIES:
        for year in main.YEARS:
            try:
                df = data_extraction.grab_yearly_results(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
                )
                # get number of buildings for each scenario
                numbers_df = data_extraction.create_scenario_selection_table_with_numbers(
                    project_name=f"{main.PROJECT_PREFIX}_{country}_{year}"
                )[["ID_Scenario", "number"]]
                merged = pd.merge(left=df, right=numbers_df, on="ID_Scenario")
                merged["year"] = year
                merged["country"] = country
                fat_df = pd.concat([fat_df, merged], axis=0)

            except:
                print(f"Data for {country} in {year} could not be loaded!")

    return fat_df


# FAT_DF = load_fat_df()


def create_plot_df(x_axis: str, y_axis: str):
    plot_df = FAT_DF.copy()
    if x_axis == "number":
        plot_df[x_axis] = plot_df[x_axis] / 1
    elif x_axis == "TotalCost":
        plot_df[x_axis] = plot_df[x_axis] / 100  # cent in €
    else:
        plot_df[x_axis] = plot_df[x_axis] / 1_000  # Wh in kWh
    if y_axis == "number":
        plot_df[y_axis] = plot_df[y_axis] / 1
    elif y_axis == "TotalCost":
        plot_df[y_axis] = plot_df[y_axis] / 100  # cent in €
    else:
        plot_df[y_axis] = plot_df[y_axis] / 1_000  # Wh in kWh
    return plot_df


def plot_yearly_result_together(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.YEARLY_RESULT_CHART, "children"),
        Input(ids.X_AXIS_DROP_DOWN, "value"),
        Input(ids.Y_AXIS_DROP_DOWN, "value")
    )
    def update_figure(x_axis: str, y_axis: str):
        plot_df = create_plot_df(x_axis, y_axis)
        number_columns = len(plot_df["year"].unique())
        fig = px.scatter(
            data_frame=plot_df,
            y=y_axis,
            x=x_axis,
            color="country",
            facet_col="year",
            size="number",
            facet_col_spacing=0.04,
            width=350 * number_columns,
            height=350,

        )
        fig.update_traces(marker=dict(line=dict(color="black", width=0.1)))
        return html.Div(dcc.Graph(figure=fig), id=ids.YEARLY_RESULT_CHART)

    return html.Div(id=ids.YEARLY_RESULT_CHART)


def plot_yearly_results_all_countries(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.HUGE_SUBPLOT_CHART, "children"),
        Input(ids.X_AXIS_DROP_DOWN, "value"),
        Input(ids.Y_AXIS_DROP_DOWN, "value")
    )
    def update_figure(x_axis: str, y_axis: str):
        plot_df = create_plot_df(x_axis, y_axis)
        number_rows = len(plot_df["country"].unique())
        number_columns = len(plot_df["year"].unique())
        fig = px.scatter(
            data_frame=plot_df,
            y=y_axis,
            x=x_axis,
            color="mode",
            facet_col="year",
            facet_row="country",
            size="number",
            facet_row_spacing=0.01,
            facet_col_spacing=0.04,
            width=350 * number_columns,
            height=250 * number_rows,

        )
        fig.update_traces(marker=dict(symbol="circle", line=dict(color="black", width=0.1)))
        fig.update_yaxes(showticklabels=True)
        fig.update_xaxes(showticklabels=True)
        return html.Div(dcc.Graph(figure=fig), id=ids.HUGE_SUBPLOT_CHART)

    return html.Div(id=ids.HUGE_SUBPLOT_CHART)


if __name__ == "__main__":
    plot_yearly_results_all_countries()
