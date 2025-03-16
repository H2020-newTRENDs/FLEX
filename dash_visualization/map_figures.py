import pathlib
from typing import List, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import numpy as np
import math
from dash_visualization import ids, data_extraction, main
from flex_operation.Country_Visualization import CountryResultPlots
import PIL
import io


COUNTRY_TRANSLATION_DICT = {
    'AUT': "Austria",
    'BEL': "Belgium",
    'BGR': "Bulgaria",
    'HRV': "Croatia",
    'CYP': "Cyprus",
    'CZE': "Czech Republic",
    'DNK': "Denmark",
    'EST': "Estonia",
    'FIN': "Finland",
    'FRA': "France",
    'DEU': "Germany",
    'GRC': "Greece",
    'HUN': "Hungary",
    'IRL': "Ireland",
    'ITA': "Italy",
    'LVA': "Latvia",
    'LTU': "Lithuania",
    'LUX': "Luxembourg",
    'MLT': "Malta",
    'NLD': "Netherlands",
    'POL': "Poland",
    'PRT': "Portugal",
    'ROU': "Roumania",
    'SVK': "Slovakia",
    'SVN': "Slovenia",
    'ESP': "Spain",
    'SWE': "Sweden"
}


def europe_map_drop_down(app: Dash) -> html.Div:
    options = ["Load Factor (%)",
               "PV self consumption (%)",
               "shifted electricity (%)",
               "electricity price std (cent/kWh)",
               "electricity price mean (cent/kWh)",
               "PV share (%)",
               "shifted electricity (MWh)",
               "MAC electricity price (cent/kWh)",
               ]
    return html.Div(
        children=[
            html.H6("Country"),
            dcc.Dropdown(
                id=ids.EU_MAP_DROP_DOWN,
                options=[{"label": option, "value": f"{option}"} for option in options],
                value="Load Factor (%)",
                multi=False,
            ),
        ]
    )


def europe_map_building_selection_drop_down(app: Dash) -> html.Div:
    options = [
        "All buildings",
        "buildings with PV",
        "buildings without PV",
        "buildings with battery",
        "buildings without battery",
        "buildings with thermal storages",
        "buildings without thermal storages"
    ]
    return html.Div(
        children=[
            html.H6("Specifics"),
            dcc.Dropdown(
                id=ids.EU_MAP_SPECIFICS_DROP_DOWN,
                options=[{"label": option, "value": f"{option}"} for option in options],
                value="All buildings",
                multi="False",
            )
        ]
    )


def load_factor_for_figure() -> pd.DataFrame:
    """ returns the mean of the Load Factor! """
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_load_factor()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    mean_df = df.groupby(["country", "year", "price_id"]).mean().reset_index()
    return mean_df


def percentage_shifted_electricity_for_figure() -> pd.DataFrame:
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_percentage_shifted_electricity()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def total_shifted_electricity_for_figure() -> pd.DataFrame:
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_total_shifted_electricity()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def pv_self_consumption_for_figure() -> pd.DataFrame:
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_self_consumption()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def load_electricity_price() -> pd.DataFrame:
    df = data_extraction.load_electricity_price(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX)
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def standard_deviation_electricity_price_for_figure() -> pd.DataFrame:
    df = load_electricity_price()
    df_std = df.groupby(["country", "year", "price_id"]).std().reset_index()
    return df_std.rename(columns={"electricity price (cent/kWh)": "electricity price std (cent/kWh)"})


def mean_electricity_price_for_figure() -> pd.DataFrame:
    df = load_electricity_price()
    df_mean = df.groupby(["country", "year", "price_id"]).mean().reset_index()
    return df_mean.rename(columns={"electricity price (cent/kWh)": "electricity price mean (cent/kWh)"})


def pv_share_for_figure() -> pd.DataFrame:
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_pv_share()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def mean_absolute_price_change_for_figure() -> pd.DataFrame:
    df = CountryResultPlots(main.COUNTRIES, main.YEARS, main.PROJECT_PREFIX).grab_EU27_mean_absolute_price_change()
    df["country"] = df["country"].map(COUNTRY_TRANSLATION_DICT)
    return df


def select_dataframe(kpi: str) -> pd.DataFrame:
    print(kpi)
    if kpi == "Load Factor (%)":
        df = load_factor_for_figure()
    elif kpi == "PV self consumption (%)":
        df = pv_self_consumption_for_figure()
    elif kpi == "shifted electricity (%)":
        df = percentage_shifted_electricity_for_figure()
    elif kpi == "electricity price std (cent/kWh)":
        df = standard_deviation_electricity_price_for_figure()
    elif kpi == "electricity price mean (cent/kWh)":
        df = mean_electricity_price_for_figure()
    elif kpi == "PV share (%)":
        df = pv_share_for_figure()
    elif kpi == "shifted electricity (MWh)":
        df = total_shifted_electricity_for_figure()
    elif kpi == "MAC electricity price (cent/kWh)":
        df = mean_absolute_price_change_for_figure()
    else:
        ValueError("This value doesn't exist")
    return df


def europe_map(app: Dash) -> html.Div:
    @app.callback(
                  Output(ids.EUROPE_MAP_CHART, "children"),
                  Input(ids.EU_MAP_DROP_DOWN, "value"),)
    def update_figure(kpi: str) -> html.Div:
        df = select_dataframe(kpi)
        number_of_rows = len(df["price_id"].unique())
        fig = px.choropleth(df,
                            locations='country',
                            locationmode='country names',
                            color=kpi,
                            scope='europe',
                            hover_data=[kpi],
                            title=f'{kpi}',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            animation_frame="year",
                            range_color=[
                                np.true_divide(math.floor(min(df[kpi]) * 10 ** 2), 10 ** 2),
                                np.true_divide(math.ceil(max(df[kpi]) * 10 ** 2), 10 ** 2)
                            ],
                            facet_row="price_id",
                            height=600 * number_of_rows,
                            width=1200)
        fig.update_geos(projection_type="natural earth")  # orthographic, mercator, equirectangular, robinson, mollweide
        fig.update_layout(
            {
                "transition": {'duration': 300},
                'legend_title': None,
            })  # slow down the speed of the animation
        return html.Div(dcc.Graph(figure=fig), id=ids.EUROPE_MAP_CHART)

    @app.callback(
        Output("output-state-gif", "children"),
        [Input(ids.SAVE_EUROPE_MAP_GIF_BUTTON, "n_clicks")],
        [Input(ids.EUROPE_MAP_CHART, "children")],
        [State("output-state-gif", "children")],
    )
    def save_europe_map_as_gif(n_clicks: int, figure_div: html.Div, prev_state: int):
        print(n_clicks)
        print(prev_state)
        figure = go.Figure(figure_div["props"]["figure"])
        if n_clicks and n_clicks > int(prev_state):
            frames = []
            if 'frames' not in figure:
                raise ValueError('The provided figure does not have any frames')
            for s, fr in enumerate(figure.frames):
                # set main traces to appropriate traces within plotly frame
                figure.update(data=fr.data)
                # move slider to correct place
                figure.layout.sliders[0].update(active=s)
                # generate image of current state
                frames.append(PIL.Image.open(io.BytesIO(figure.to_image(format="png"))))
                print(frames)

            # create animated GIF
            frames[0].save(
                f"Map_{figure.layout.title.text.replace('/', 'per')}.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=2000,
                loop=0,
            )
            print(f"GIF saved")
            return n_clicks
        else:
            return n_clicks

    @app.callback(
        Output("output-state-csv", "children"),
        [Input(ids.SAVE_EUROPE_MAP_CSV_BUTTON, "n_clicks"),
         Input(ids.EU_MAP_DROP_DOWN, "value")],
        [State("output-state-csv", "children")],
    )
    def save_europe_map_to_csv(n_clicks: int, kpi: str, prev_state: int):
        if prev_state is None:
            prev_state = 0
        if n_clicks and n_clicks > int(prev_state):
            df = select_dataframe(kpi)
            path = pathlib.Path(
                r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects/" + f"NewTrends_{main.PROJECT_PREFIX}/")
            df.to_csv(path / f"{kpi.replace(r'/', 'per')}.csv", sep=";", index=False)
            print(f"{kpi} saved as CSV")
            return n_clicks
        else:
            return n_clicks

    return html.Div(id=ids.EUROPE_MAP_CHART)







if __name__ == "__main__":
    mean_absolute_price_change_for_figure()

