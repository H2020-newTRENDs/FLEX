import pandas as pd
import plotly.express as px
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from dash_visualization import ids, data_extraction, main
import numpy as np


def load_aggregated_load_profiles():
    all_load_profiles = pd.DataFrame()
    for country in main.COUNTRIES:
        for year in main.YEARS:
            try:
                ref_load, sems_load = main.LoadProfileAnalyser(f"D5.4_{country}_{year}", country, year).load_aggregated_load_profiles()
                ref_load.columns = ["price ID", "reference"]
                sems_load.columns = ["price ID", "optimization"]
                df = pd.concat([ref_load, sems_load.drop(columns=["price ID"])], axis=1)
                melted = df.melt(value_vars=["reference", "optimization"], id_vars="price ID", var_name="mode", value_name="Load (GW)")
                melted["country"] = country
                melted["year"] = year
                melted["hour"] = np.hstack((np.arange(8760), np.arange(8760)))
                all_load_profiles = pd.concat([all_load_profiles, melted], axis=0)
            except:
                print(f"Load profiles for {country} {year} could not be loaded")
    return all_load_profiles.reset_index(drop=True)


ALL_LOAD_PROFILES = load_aggregated_load_profiles()



def plot_aggregated_load_profiles(app: Dash) -> html.Div:
        n_rows = len(list(ALL_LOAD_PROFILES["year"].unique()))
        fig = px.line(
            data_frame=ALL_LOAD_PROFILES,
            x="hour",
            y="Load (GW)",
            color="country",
            facet_row="year",
            line_dash="mode",
            width=1_200,
            height=300*n_rows
            )
        return html.Div(dcc.Graph(figure=fig), id=ids.AGGREGATED_LOAD_PROFILE_CHART)




if __name__ == "__main__":
    load_aggregated_load_profiles()

