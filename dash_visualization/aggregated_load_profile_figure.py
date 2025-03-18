import pandas as pd
import plotly.express as px
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from dash_visualization import ids, data_extraction, main
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

country_code_correction = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "HRV": "HR",
    "CYP": "CY",
    "CZE": "CZ",
    "DNK": "DK",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "DEU": "DE",
    "GRC": "GR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LVA": "LV",
    "LTU": "LT",
    "LUX": "LU",
    "MLT": "MT",
    "NLD": "NL",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "ESP": "ES",
    "SWE": "SE"
}


def load_aggregated_load_profiles():
    all_load_profiles = pd.DataFrame()
    for country in main.COUNTRIES:
        for year in main.YEARS:
            try:
                ref_load, sems_load = main.LoadProfileAnalyser(f"{main.PROJECT_PREFIX}_{country}_{year}", country,
                                                               year).load_aggregated_load_profiles()
                ref_load.columns = ["price ID", "reference"]
                sems_load.columns = ["price ID", "optimization"]
                df = pd.concat([ref_load, sems_load.drop(columns=["price ID"])], axis=1)
                melted = df.melt(value_vars=["reference", "optimization"], id_vars="price ID", var_name="mode",
                                 value_name="Load (GW)")
                melted["country"] = country
                melted["year"] = year
                melted["hour"] = np.hstack((np.arange(8760), np.arange(8760)))
                all_load_profiles = pd.concat([all_load_profiles, melted], axis=0)
            except:
                print(f"Load profiles for {country} {year} could not be loaded")
    return all_load_profiles.reset_index(drop=True)


# ALL_LOAD_PROFILES = load_aggregated_load_profiles()


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
        height=300 * n_rows
    )
    return html.Div(dcc.Graph(figure=fig), id=ids.AGGREGATED_LOAD_PROFILE_CHART)


def plot_generation_load_shift(app: Dash) -> html.Div:
    residential_demand = load_aggregated_load_profiles()
    if main.PROJECT_PREFIX == "D5.4":
        generation = pd.read_csv(
            Path(
                r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects/NewTrends_D5.4/") / "gen_data_shiny.csv",
        )
    else:
        generation = pd.read_csv(
            Path(
                r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects/NewTrends_D5.4/") / "gen_data_leviathan.csv",
            sep=";"
        )

    prices = data_extraction.load_electricity_price(countries=main.COUNTRIES, years=main.YEARS,
                                                    project_prefix=main.PROJECT_PREFIX)

    @app.callback(
        Output(ids.GENERATION_SHIFT_CHART, "children"),
        Input(ids.COUNTRY_DROP_DOWN_GENERATION_SHIFT, "value")
    )
    def update_generation_load_shift(country: str):
        # create figure
        fig = make_subplots(rows=4,
                            cols=1,
                            shared_xaxes=True,
                            subplot_titles=('Year 2020', 'Year 2030', 'Year 2040', 'Year 2050'),
                            vertical_spacing=0.1,
                            specs=[
                                [{"secondary_y": True}],
                                [{"secondary_y": True}],
                                [{"secondary_y": True}],
                                [{"secondary_y": True}]
                            ],
                            column_widths=[1000],
                            row_heights=[800] * 4,
                            )
        for i, year in enumerate(main.YEARS, start=1):
            # calculate the values:
            ref = residential_demand.query(
                f"year == {year} and country == '{country}' and mode == 'reference'"
            )["Load (GW)"].to_numpy()
            opt = residential_demand.query(
                f"year == {year} and country == '{country}' and mode == 'optimization'"
            )["Load (GW)"].to_numpy()
            shift_away = np.maximum(ref - opt, 0)
            new_peak = np.maximum(opt - ref, 0)
            # total generation:
            gen = generation.query(f"year == {year} and country == '{country_code_correction[country]}'")[
                      "generation"].to_numpy() / 1_000
            if year != 2020:
                gen = np.concatenate((gen, gen[-24:]))  # add the first day because generation data is missing one day
            # electricity price:
            electricity_price = prices.query(
                f"year == {year} and country == '{country}'"
            )["electricity price (cent/kWh)"].to_numpy()

            trace_generation = go.Scatter(
                x=np.arange(len(shift_away)),  # Your time data here
                y=gen,
                fill=None,
                mode='lines',
                name='Generation (GW)',
                line=dict(color="blue")
            )
            # Create the area chart for the demand difference
            trace_shift = go.Scatter(
                x=np.arange(len(shift_away)),
                y=np.where(shift_away > 0, gen - shift_away, gen),
                fill='tonexty',  # This will fill the area below this line
                mode='lines',
                name='Demand shifted',
                fillcolor="green",
                line=dict(color="green"),
            )
            trace_peak = go.Scatter(
                x=np.arange(len(new_peak)),
                y=np.where(new_peak > 0, gen + new_peak, gen),
                fill='tonexty',  # This will fill the area below this line
                mode='lines',
                name='New demand peak',
                fillcolor="red",
                line=dict(color="red"),
            )
            elec_price = go.Scatter(
                x=np.arange(len(new_peak)),
                y=electricity_price,
                mode='lines',
                name='electricity price (€/kWh)',
                line=dict(color="rgb(148, 103, 189)"),
                yaxis="y2"
            )

            # Put all the data into a single figure
            fig.add_trace(trace_generation, row=i, col=1)
            fig.add_trace(trace_shift, row=i, col=1)
            fig.add_trace(trace_generation, row=i, col=1)
            fig.add_trace(trace_peak, row=i, col=1)
            fig.add_trace(elec_price, row=i, col=1, secondary_y=True)

        # Update yaxis settings for the secondary y-axis of each subplot
        for i in range(1, 4):
            fig.update_layout({
                f'yaxis2{i}': dict(
                    title='electricity price (cent/kWh)',
                    titlefont=dict(color='rgb(148, 103, 189)'),
                    tickfont=dict(color='rgb(148, 103, 189)'),
                    overlaying=f'y{i}',
                    side='right',
                    anchor=f'x{i}'
                )
            })
        fig.layout.height = 1_000
        fig.layout.width = 500 * 4

        return html.Div(dcc.Graph(figure=fig), id=ids.GENERATION_SHIFT_CHART)

    @app.callback(
        Output("output-demand-profile-csv", "children"),
        [Input(ids.SAVE_DEMAND_PROFILE_CSV_BUTTON, "n_clicks"),
         Input(ids.COUNTRY_DROP_DOWN_GENERATION_SHIFT, "value"),
         ],
        [State("output-demand-profile-csv", "children")],
    )
    def save_demand_profile_to_csv(n_clicks: int, country: str, prev_state: int):
        if prev_state is None:
            prev_state = 0
        if n_clicks and n_clicks > int(prev_state):
            csv_df = pd.DataFrame()
            for i, year in enumerate(main.YEARS, start=1):
                if main.PROJECT_PREFIX == "D5.4":
                    generation = pd.read_csv(
                        Path(
                            r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects/NewTrends_D5.4/") / "gen_data_shiny.csv",
                    )
                else:
                    generation = pd.read_csv(
                        Path(
                            r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects/NewTrends_D5.4/") / "gen_data_leviathan.csv",
                        sep=";"
                    )
                residential_demand = load_aggregated_load_profiles()
                prices = data_extraction.load_electricity_price(countries=main.COUNTRIES, years=main.YEARS,
                                                                project_prefix=main.PROJECT_PREFIX)
                ref = residential_demand.query(
                    f"year == {year} and country == '{country}' and mode == 'reference'"
                )["Load (GW)"]
                opt = residential_demand.query(
                    f"year == {year} and country == '{country}' and mode == 'optimization'"
                )["Load (GW)"]
                gen = generation.query(f"year == {year} and country == '{country_code_correction[country]}'")[
                          "generation"] / 1_000

                if year != 2020:
                    gen = pd.concat([gen, gen[-24:]], axis=0).reset_index(
                        drop=True)  # add the first day because generation data is missing one day
                # electricity price:
                electricity_price = prices.query(
                    f"year == {year} and country == '{country}'"
                )["electricity price (cent/kWh)"]
                # create a table to export to csv later:
                merged = pd.concat([opt, ref, gen, electricity_price], axis=1)
                merged.columns = [f"prosumager load (GW) {year}",
                                  f"consumer load (GW) {year}",
                                  f"generation (GW) {year}",
                                  "price (cent/kWh)"]
                csv_df = pd.concat([csv_df, merged], axis=1)
            csv_df.to_csv(Path(r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects") /
                          f"NewTrends_{main.PROJECT_PREFIX}" / f"country_level_load_{country}.csv", sep=";",
                          index=False)
            print(f"demand profiles for {country} saved as CSV")
            return n_clicks
        else:
            return n_clicks

    @app.callback(
        Output(ids.SLICE_GENERATION_SHIFT_FIGURE, "children"),
        [Input(ids.COUNTRY_DROP_DOWN_GENERATION_SHIFT, "value"),
         Input(ids.SLICE_GENERATION_SHIFT_YEAR_DROP_DOWN, "value"),
         Input(ids.SLICE_DEMAND_SHIFT_START_INPUT_OUTPUT, "value", ),
         Input(ids.SLICE_DEMAND_SHIFT_END_INPUT_OUTPUT, "value"),
         Input(ids.SAVE_SLICE_GENERATION_SHIFT_BUTTON, "n_clicks")],
        [State(ids.SLICE_GENERATION_SHIFT_FIGURE, "children")]

    )
    def save_slice_load_shift(country: str, year: int, start_value: int, end_value: int, n_clicks, prev_state):
        if prev_state is None:
            prev_state = 0
        if n_clicks and n_clicks > int(prev_state):
            print(f"button pressed, start:{start_value} end:{end_value}")
            # calculate the values:
            ref = residential_demand.query(
                f"year == {year} and country == '{country}' and mode == 'reference'"
            )["Load (GW)"].to_numpy()
            opt = residential_demand.query(
                f"year == {year} and country == '{country}' and mode == 'optimization'"
            )["Load (GW)"].to_numpy()
            # total generation:
            gen = generation.query(f"year == {year} and country == '{country_code_correction[country]}'")[
                      "generation"].to_numpy() / 1_000
            if year != 2020:
                gen = np.concatenate(
                    (gen, gen[-24:]))  # add the first day because generation data is missing one day
            # electricity price:
            electricity_price = prices.query(
                f"year == {year} and country == '{country}'"
            )["electricity price (cent/kWh)"].to_numpy()

            figure = create_slice_load_shift(reference_load=ref[int(start_value):int(end_value)],
                                             prosumager_load=opt[int(start_value):int(end_value)],
                                             generation=gen[int(start_value):int(end_value)],
                                             price=electricity_price[int(start_value):int(end_value)])
            figure_name = f"Load_vs_Generation_{country}_{year}_{int(start_value)}_{int(end_value)}.svg"
            figure.savefig(Path(r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/projects") /
                           f"NewTrends_{main.PROJECT_PREFIX}" / figure_name)
            print(f"saved svg")
        return n_clicks


    return html.Div(id=ids.GENERATION_SHIFT_CHART)


def create_slice_load_shift(reference_load: np.array,
                            prosumager_load: np.array,
                            generation: np.array,
                            price: np.array) -> plt.figure:
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot generation on the primary y-axis
    ax1.plot(generation, label='Grid Demand', color='black', linewidth=2)
    ax1.plot(generation - reference_load + prosumager_load, label='Grid demand with prosumer', color='blue', linewidth=2)

    ax1.fill_between(range(len(generation)), generation, generation+prosumager_load-reference_load, color='red', alpha=0.5,
                     label='Increased Demand', interpolate=True, where=prosumager_load>reference_load)
    ax1.fill_between(range(len(generation)), generation, generation+prosumager_load-reference_load, color='green', alpha=0.5,
                     label='Reduced Demand', interpolate=True, where=reference_load>prosumager_load)

    # Set labels, title, and legend for primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Demand (GW)')
    ax1.set_title('Demand vs Price')
    ax1.legend(loc='upper left')

    # Create secondary y-axis for price
    ax2 = ax1.twinx()
    ax2.plot(price, label='Price', color='purple', linewidth=2, linestyle='--')
    ax2.set_ylabel('Price')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    reference_load = np.array([10, 12, 11, 14, 15, 25, 25])
    prosumager_load = np.array([25, 11, 12, 25, 14, 18, 13])
    generation = np.array([100, 110, 110, 130, 140, 160, 120])
    price = np.array([5, 5.5, 5.2, 5.1, 5.3, 5.6, 5.4])
    create_slice_load_shift(reference_load, prosumager_load, generation, price)
