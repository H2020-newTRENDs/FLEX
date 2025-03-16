from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash_visualization import map_figures, dash_figures, buttons, dropdown, year_figures, \
                                aggregated_load_profile_figure, input_fields


def create_layout(app: Dash) -> html.Div:
    yearly_results_page = dbc.Container(
        [
            html.H1('Yearly FLEX results'),
            html.Hr(),
            dbc.Row([
                dbc.Col(dropdown.x_axis_selection_drop_down(app)),
                dbc.Col(dropdown.y_axis_selection_drop_down(app)),
            ]),
            dbc.Row(dbc.Col(year_figures.plot_yearly_result_together(app))),
            dbc.Row(dbc.Col(year_figures.plot_yearly_results_all_countries(app))),
        ]
    )

    map_page = dbc.Container(
        [
            html.H1('EU27 results'),
            html.Hr(),
            dbc.Row([
                dbc.Col(buttons.save_europe_map_gif_button(app), md=4),
                dbc.Col(buttons.save_europe_map_csv_button(app), md=8),
            ]),
            dbc.Row([
                map_figures.europe_map_drop_down(app),
                map_figures.europe_map(app),
                html.Div(id="output-state-gif"),
                html.Div(id="output-state-csv")
            ]),


        ]
    )

    information_on_invert_input_page = dbc.Container(
        [
            html.H1("Input data to the FLEX model"),
            html.Hr(),
            # Drop down menus:
            dbc.Row(
                [
                    dbc.Col(dropdown.country_drop_down_for_building_numbers(app), md=4),
                ],
                align="center"
            ),
            html.Hr(),
            html.H5("Information on the underlying building stock data: number of buildings"),

            html.Hr(),
            dash_figures.building_age_distribution(app),

            dbc.Row(
                [
                    dbc.Col(dash_figures.share_MFH_SFH(app), ),
                    dbc.Col(dash_figures.share_battery(app), ),
                    dbc.Col(dash_figures.share_dhw(app), ),

                ],
                align="center"
            ),
            dbc.Row(
                [
                    dbc.Col(dash_figures.share_heating_tank(app), ),
                    dbc.Col(dash_figures.share_pv(app), ),
                    dbc.Col(dash_figures.share_boiler(app), ),

                ],
                align="center"
            ),
            dbc.Row(
                [
                    dbc.Col(dash_figures.share_heating_element(app), ),
                    dbc.Col(dash_figures.share_space_cooling(app), ),
                    dbc.Col(dash_figures.share_vehicle(app), ),

                ],
                align="center"
            ),
        ]
    )

    specific_results_page = dbc.Container(
        [
            html.H1(app.title),
            html.Hr(),
            html.H5("The results presented here include the peak to peak power, load factor "
                    ",total amount of shifted electricity by the buildings with heat pumps and the PV self-consumption"
                    "for all the buildings with PV."),
            html.Hr(),

            # Drop down menus:
            dbc.Row(
                [
                    dbc.Col(dropdown.all_countries_dropdown(app), md=4),
                    dbc.Col(dropdown.all_years_dropdown(app), md=8),
                ],
                align="center"
            ),

            # figures:
            html.Hr(),
            html.H5("The peak to peak performance describes the difference of the minimum to maximum Grid demand "
                    "by all buildings with heat pumps within a country."),
            dcc.Markdown('''
                $$
                P2P = P_{t,max} - P_{t,min}    \\,\\,\\,\\,\\,\\,    0 \\leq t \\leq 24
                $$
            ''', mathjax=True),
            dash_figures.EU27_p2p(app),

            html.Hr(),
            html.H5("The Load Factor is the ratio of the average of the load registered over a given period of time "
                    "to the maximum load registered during the same period. The load factor is also considered to be "
                    "a measure of the “peakyness” of the load profile, where a higher load factor indicates a smoother "
                    "load profile. The higher the load factor, the better."),
            dcc.Markdown('''
                    $$
                    𝐿𝐹 = \\frac{mean(𝑃_{𝑡})}{𝑃_{t,𝑚𝑎𝑥}}  \\,\\,\\,\\,\\,\\,   0 \\leq t \\leq 8760
                    $$
                    ''', mathjax=True),
            dash_figures.EU27_load_factor(app),

            html.Hr(),
            html.H5("The pv self-consumption describes the average pv self-consumption of all buildings with a "
                    "PV system in a country. With a higher share of low energy buildings, this share should rise."),
            dcc.Markdown('''
                    $$
                    PV_{self-consumption} = \\frac{\\sum{P_{pv2Load, t}}} {\\sum{P_{pv-generation, t}}}
                    \\,\\,\\,\\,\\,\\,   0 \\leq t \\leq 8760
                    $$
                    ''', mathjax=True),
            dash_figures.EU27_pv_self_consumption(app),

            html.Hr(),
            html.H5("The shifted electricity describes the percentage of the electricity shifted to the "
                    "overall electricity consumption of all heat pump operated buildings within a country. With"
                    "a higher share of high efficient buildings, this percentage should increase."),
            dcc.Markdown('''
               $$
               shifted \\,\\,electricity = \\frac{\\int_{t=0}^{8760}P_{reference}-P_{optimized} \\,dt}
               {\\int_{t=0}^{8760}P_{reference}} \\, dt
               \\,\\,\\,\\,\\,\\,  0 \\leq t \\leq 8760
               $$
               ''', mathjax=True),
            dash_figures.EU27_shifted_electricity(app),

            html.Hr(),
            html.H5("difference in total Grid demand and Load on country level for Prosumagers and Consumers to "
                    "have an Idea how much more total electricity is used by Prosumagers because of storage losses."),
            dash_figures.EU27_electricity_consumption_difference_country_level(app),

        ]
        , fluid=True
    )

    hourly_results_page = dbc.Container(
        [
            html.H1('Hourly FLEX results'),
            html.Hr(),

            # Drop down menus:
            dbc.Row(
                [
                    dbc.Col(dropdown.country_drop_down(app)),
                    dbc.Col(dropdown.year_drop_down(app)),
                    dbc.Col(dropdown.battery_dropdown(app), ),
                    dbc.Col(dropdown.building_dropdown(app)),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dropdown.boiler_dropdown(app)),
                    dbc.Col(dropdown.energy_price_dropdown(app)),
                    dbc.Col(dropdown.heating_element_dropdown(app)),
                    dbc.Col(dropdown.hot_water_tank_dropdown(app)),
                ]
            ),
            dbc.Row([
                dbc.Col(dropdown.pv_dropdown(app)),
                dbc.Col(dropdown.space_cooling_dropdown(app)),
                dbc.Col(dropdown.space_heating_tank_dropdown(app)),
                dbc.Col(dropdown.vehicle_dropdown(app)),

            ], align="center"
            ),
            dbc.Row(
                [
                    dbc.Col(dropdown.hourly_column_dropdown(app), md=3),
                    # figure:
                    dbc.Col(dash_figures.hourly_profiles(app), )

                ]
            )
        ], fluid=True
    ),

    aggregated_load_profile_page = html.Div(
        [
            html.H1('Aggregated load profiles'),
            dbc.Row(aggregated_load_profile_figure.plot_aggregated_load_profiles(app)),
        ]

    ),

    # generation shift page
    generation_shift_page = html.Div(
        [
            html.H1("Shifted Load compared to total generation in the country"),
            dbc.Row([
                dbc.Col(dropdown.generation_shift_drop_down(app), md=3),
                dbc.Col(buttons.save_demand_profiles_csv_button(app), md=6)
            ]),
            html.Div(id="output-demand-profile-csv"),
            aggregated_load_profile_figure.plot_generation_load_shift(app),

            dbc.Row([
                dbc.Col(input_fields.create_generation_slice_start_input_field(), md=3),
                dbc.Col(input_fields.create_generation_slice_end_input_field(), md=6),
                dbc.Col(dropdown.dropdown_generation_slice_year_options(), md=9),
                dbc.Col(buttons.save_slice_demand_shift_profiles_button(app), md=12)

            ]),
            html.Div(id="slice-generation-shift-figure")


        ]
    )

    cost_page = html.Div(
        [
            html.H1("Cost reduction through prosumaging"),
            html.H4("The plot does only show the range of cost savings not the actual distribution of the houses that "
                    "make these savings."),
            html.Hr(),
            dbc.Row([dbc.Col(drop_down) for drop_down in dropdown.create_dropdown_options_for_cost_reduction_plot(app)],
                    align="center"),
            dbc.Row(dbc.Col(dropdown.dropdown_options_for_cost(app), md=4)),
            dash_figures.cost_reduction_box_plot_all_countries(app),

        ]
    )


    sidebar = html.Div(
        [
            html.H2("FLEX", className="flex-sidebar"),
            html.Hr(),
            html.P(
                "Results from the Flex model", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Yearly Results", href="/", active="exact"),
                    dbc.NavLink("Hourly Profiles", href="/hourly-profiles", active="exact"),
                    dbc.NavLink("Building stock numbers", href="/building-stock-numbers", active="exact"),
                    dbc.NavLink("Specific Results", href="/specific-results", active="exact"),
                    dbc.NavLink("Results on European level", href="/results-on-european-level", active="exact"),
                    dbc.NavLink("Aggregated load profiles", href="/aggregated-load-profiles", active="exact"),
                    dbc.NavLink("Generation shift", href="/generation-shift", active="exact"),
                    dbc.NavLink("Costs", href="/costs", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "background-color": "#f8f9fa",
        },
    )
    sidebar_content = html.Div(id="page-content",
                               style={
                                   "margin-left": "18rem",
                                   "margin-right": "2rem",
                                   "padding": "2rem 1rem",
                               })

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return yearly_results_page
        elif pathname == "/hourly-profiles":
            return hourly_results_page
        elif pathname == "/specific-results":
            return specific_results_page
        elif pathname == "/input-data":
            return information_on_invert_input_page
        elif pathname == "/building-stock-numbers":
            return information_on_invert_input_page
        elif pathname == "/results-on-european-level":
            return map_page
        elif pathname == "/aggregated-load-profiles":
            return aggregated_load_profile_page
        elif pathname == "/generation-shift":
            return generation_shift_page
        elif pathname == "/costs":
            return cost_page
        else:
            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )

    layout = html.Div([dcc.Location(id="url", refresh=False), sidebar, sidebar_content])
    return layout
