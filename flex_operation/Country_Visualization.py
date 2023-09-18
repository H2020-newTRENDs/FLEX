import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import multiprocessing
from typing import List, Union, Dict

import plotly.graph_objects
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from flex.config import Config
from flex.db import DB
from flex.db_init import DatabaseInitializer
from flex_operation.constants import OperationTable, OperationScenarioComponent
from flex_operation.LoadProfileAnalysation import LoadProfileAnalyser
from dash_visualization import data_extraction


class CountryData:
    def __init__(self, countries: List[str], years: List[int], project_prefix: str):
        self.countries = countries
        self.years = years
        self.projects = [
            {
                "name": f"{project_prefix}_{country}_{year}",
                "year": year,
                "country": country
            } for country in countries for year in years
        ]
        self.projects_per_year = [[f"{project_prefix}_{country}_{year}" for country in countries] for year in years]

    def grab_EU27_self_consumption(self, filter=None):
        big_df = pd.DataFrame(columns=['PV self consumption (%)', 'price_id', 'year', 'country'])
        for project in self.projects:
            try:
                scenarios = data_extraction.create_scenario_selection_table_with_numbers(project_name=project["name"])
                LPA = LoadProfileAnalyser(project_name=project["name"],
                                          year=project["year"],
                                          country=project["country"])
                ref_consumption, opt_consumption = LPA.calc_country_self_consumption(filter=filter)
                # check if ref and opt consumption exist, if they don't scenarios are all without PV:
                if len(ref_consumption) == 0:
                    ref_consumption, opt_consumption = {1: np.float(0)}, {
                        1: np.float(0)}  # careful if there are other prices
                big_df = self.combine_seasonal_dicts_to_df(ref_dict=ref_consumption,
                                                           opt_dict=opt_consumption,
                                                           year=project["year"],
                                                           country=project["country"],
                                                           big_df=big_df,
                                                           value_name="PV self consumption (%)")
            except:
                print(f"Self consumption for {project['country']} {project['year']} could not be loaded!")
        big_df["PV self consumption (%)"] = big_df["PV self consumption (%)"].astype(float)
        # drop countries where self-consumption is 0 because they distort the scale
        return_df = big_df[big_df["PV self consumption (%)"] != 0]
        return return_df


class CountryResultPlots:

    def __init__(self, countries: List[str], years: List[int], project_prefix: str):
        self.countries = countries
        self.years = years
        self.projects = [{"name": f"{project_prefix}_{country}_{year}",
                          "year": year,
                          "country": country} for country in countries for year in years]
        self.projects_per_year = [[f"{project_prefix}_{country}_{year}" for country in countries] for year in years]

    def load_general_electricity_price_ids(self) -> list:
        source = Path(__file__).parent.parent / "data" / "input_operation" / \
                 f"{OperationScenarioComponent.EnergyPrice.table_name}.xlsx"
        price_table = pd.read_excel(source, engine="openpyxl").dropna(axis=1, how="all").dropna(axis=0, how="all")
        price_ids = list(price_table.loc[:, "ID_EnergyPrice"])
        return price_ids

    def load_general_electricity_price_profiles(self) -> dict:
        """ returns a dict with the price ID as the key and the electricity price profile as the value"""
        source = Path(__file__).parent.parent / "data" / "input_operation" / \
                 f"{OperationTable.EnergyPriceProfile}.xlsx"
        prices = pd.read_excel(source, engine="openpyxl").dropna(axis=1, how="all").dropna(axis=0, how="all")
        price_ids = self.load_general_electricity_price_ids()
        price_profiles = {id_price: prices[f"electricity_{id_price}"] for id_price in price_ids}
        return price_profiles

    def plotly_aggregated_load_profiles(self):
        # determine the number of price IDs for subplots:
        figs = {}
        # load the load profile excels from all projects:
        for country in self.countries:
            country_df = pd.DataFrame(columns=['price ID', 'load', 'mode', 'year'])
            for i, year in enumerate(self.years):
                project = f"D5.4_{country}_{year}"
                LPA = LoadProfileAnalyser(project_name=project, year=year, country=country)
                ref_agg_profiles, sems_agg_profiles = LPA.load_aggregated_load_profiles()
                ref_agg_profiles["mode"] = "sems load (GW)"
                sems_agg_profiles["mode"] = "reference load (GW)"
                df = pd.concat([ref_agg_profiles.rename(columns={"reference load (GW)": "load"}),
                                sems_agg_profiles.rename(columns={"sems load (GW)": "load"})], axis=0)
                df["year"] = year
                country_df = pd.concat([country_df, df], axis=0)

            figs[country] = px.line(
                country_df,
                x=country_df.index,
                y="load",
                color="mode",
                facet_row="year",
                line_dash="price ID"
            )
        # Create menu
        update_menus = [{
            'buttons': [],
            'direction': 'down',
            'showactive': True
        }]
        for key in figs:
            update_menus[0]['buttons'].append({
                'method': 'update',
                'label': str(key),
                'args': [{
                    'customdata': [dat.customdata for dat in figs[key].data],
                    'hovertemplate': [dat.hovertemplate for dat in figs[key].data],
                    'x': [dat.x for dat in figs[key].data],
                    'y': [dat.y for dat in figs[key].data]
                }, {
                    'yaxis': figs[key].layout.yaxis
                }]
            })

        # update_menus[0]['type'] = menu_type
        # Create single figure
        fig = figs[list(figs)[0]].update_layout(
            updatemenus=update_menus,
            title="Aggregated load profiles"
        )

        # Finalize figure
        fig.update_layout({
            'legend_title': None,
            'legend_font_size': 18,
            'legend_tracegroupgap': 20,
        })
        fig.show()

    def plotly_barplot(self, figs: dict, df: pd.DataFrame, y_label: str, title: str, country: str):
        """
        df: needs to have years (x-axis) as column and "reference" as well as "optimized" for the value
        as column.
        figs: is a dictionary that contrains the plotly figures, at initialization it will be empty
        y_label: str which will be the y label
        title: str
        country: str describing the country for which the plot is generated
        """

    def plotly_PV_self_consumption(self):
        """ plot the PV self consumption rate for every year in one barplot grouping the comparison
        between reference and prosumager and having the countries as drop down menu."""
        figs = {}
        for country in self.countries:
            country_df_years = pd.DataFrame(columns=["reference", "optimized", "year", "country"])
            for year in self.years:
                project = f"D5.4_{country}_{year}"
                LPA = LoadProfileAnalyser(project_name=project, year=year, country=country)
                ref_consumption, opt_consumption = LPA.calc_country_self_consumption()
                # Zip the values of the two dictionaries
                values = list(zip(ref_consumption.values(), opt_consumption.values()))
                # Create a DataFrame with the keys as the index and the two columns as the values
                df = pd.DataFrame(values, index=ref_consumption.keys(), columns=['reference', 'optimized'])
                df["year"] = year
                df["country"] = country

                country_df_years = pd.concat([country_df_years, df], axis=0).reset_index(drop=True)

            # now plot as bar figure for this specific country:
            figs[country] = px.bar(
                country_df_years,
                x="year",
                y=["reference", "optimized"],
                barmode="group",
                orientation="v",
                labels={"value": "PV self-consumption (%)"}
            )
        # Create menu
        update_menus = [{
            'buttons': [],
            'direction': 'down',
            'showactive': True
        }]
        for key in figs:
            update_menus[0]['buttons'].append({
                'method': 'update',
                'label': str(key),
                'args': [{
                    'customdata': [dat.customdata for dat in figs[key].data],
                    'hovertemplate': [dat.hovertemplate for dat in figs[key].data],
                    'x': [dat.x for dat in figs[key].data],
                    'y': [dat.y for dat in figs[key].data]
                }, {
                    'yaxis': figs[key].layout.yaxis
                }]
            })

        # update_menus[0]['type'] = menu_type
        # Create single figure
        fig = figs[list(figs)[0]].update_layout(
            updatemenus=update_menus,
            title="PV self-consumption"
        )

        # Finalize figure
        fig.update_layout({
            'legend_title': None,
            'legend_font_size': 18,
            'legend_tracegroupgap': 20,
            'width': 600,
            'height': 400
        })
        fig.show()

    def plotly_peak_to_peak(self):
        figs = {}
        for country in self.countries:
            country_df_years = pd.DataFrame(columns=['winter', 'spring', 'summer', 'autumn', 'mode', 'country', 'year'])
            seasons = ["winter", "spring", "summer", "autumn"]

            for year in self.years:
                project = f"D5.4_{country}_{year}"
                LPA = LoadProfileAnalyser(project_name=project, year=year, country=country)
                ref_p2p, opt_p2p = LPA.analyze_peak_to_peak_difference_total_load_profiles()
                for price_id in opt_p2p.keys():
                    df_opt = pd.DataFrame(opt_p2p[price_id], index=seasons).T
                    df_opt["mode"] = "optimized"
                    df_ref = pd.DataFrame(ref_p2p[price_id], index=seasons).T
                    df_ref["mode"] = "reference"

                    df = pd.concat([df_opt, df_ref])
                    df["country"] = country
                    df["year"] = year
                    df["price_id"] = int(price_id)
                    country_df_years = pd.concat([country_df_years, df]).reset_index(drop=True)

            plot_df = country_df_years.melt(id_vars=["mode", "country", "year", "price_id"],
                                            var_name="season", value_name="Peak to peak load (MW)")

            # now plot as box figure for this specific country:
            figs[country] = make_subplots(rows=1, cols=len(self.years), subplot_titles=self.years,
                                          shared_yaxes=True)
            for i, year in enumerate(self.years):
                year_df = plot_df.query("year==@year")
                figs[country].add_trace(go.Box(y=year_df.query("mode=='reference'")["Peak to peak load (MW)"],
                                               x=year_df.query("mode=='reference'")["season"],
                                               name="reference",
                                               notched=True,
                                               legendgroup="reference",
                                               showlegend=True if i == 0 else False,
                                               line={"color": "steelblue"}),
                                        row=1, col=i + 1)
                figs[country].add_trace(go.Box(y=year_df.query("mode=='optimized'")["Peak to peak load (MW)"],
                                               x=year_df.query("mode=='optimized'")["season"],
                                               name="optimized",
                                               notched=True,
                                               legendgroup="optimized",
                                               showlegend=True if i == 0 else False,
                                               line={"color": "firebrick"}),
                                        row=1, col=i + 1)

        # Create menu
        update_menus = [{
            'buttons': [],
            'direction': 'down',
            'showactive': True
        }]
        for key in figs:
            update_menus[0]['buttons'].append({
                'method': 'update',
                'label': str(key),
                'args': [{
                    'customdata': [dat.customdata for dat in figs[key].data],
                    'hovertemplate': [dat.hovertemplate for dat in figs[key].data],
                    'x': [dat.x for dat in figs[key].data],
                    'y': [dat.y for dat in figs[key].data]
                }, {
                    'yaxis': figs[key].layout.yaxis
                }]
            })

        # update_menus[0]['type'] = menu_type
        # Create single figure
        fig = figs[list(figs)[0]].update_layout(
            updatemenus=update_menus,
            title="Peak to peak load",
            boxmode="group",
            yaxis_title="Peak to peak load (MW)"
        )
        fig.update_xaxes(type='category')

        # Finalize figure
        fig.update_layout({
            'legend_title': None,
            'legend_font_size': 18,
            'legend_tracegroupgap': 20,
            'width': 1000,
            'height': 400
        })
        fig.show()

    def plotly_load_factor(self):
        figs = {}
        for country in self.countries:
            country_df_years = pd.DataFrame(columns=['winter', 'spring', 'summer', 'autumn', 'mode', 'country', 'year'])
            seasons = ["winter", "spring", "summer", "autumn"]

            for year in self.years:
                project = f"D5.4_{country}_{year}"
                LPA = LoadProfileAnalyser(project_name=project, year=year, country=country)
                ref_consumption, opt_consumption = LPA.analyze_load_factor()
                for price_id in opt_consumption.keys():
                    df_opt = pd.DataFrame(opt_consumption[price_id], index=seasons).T
                    df_opt["mode"] = "optimized"
                    df_ref = pd.DataFrame(ref_consumption[price_id], index=seasons).T
                    df_ref["mode"] = "reference"

                    df = pd.concat([df_opt, df_ref])
                    df["country"] = country
                    df["year"] = year
                    df["price_id"] = int(price_id)

                    country_df_years = pd.concat([country_df_years, df]).reset_index(drop=True)

            plot_df = country_df_years.melt(id_vars=["mode", "country", "year", "price_id"],
                                            var_name="season", value_name="Load factor")

            # now plot as bar figure for this specific country:
            figs[country] = make_subplots(rows=1, cols=len(self.years), subplot_titles=self.years,
                                          shared_yaxes=True)
            for i, year in enumerate(self.years):
                year_df = plot_df.query("year==@year")
                figs[country].add_trace(go.Box(y=year_df.query("mode=='reference'")["Load factor"],
                                               x=year_df.query("mode=='reference'")["season"],
                                               name="reference",
                                               notched=True,
                                               legendgroup="reference",
                                               showlegend=True if i == 0 else False,
                                               line={"color": "steelblue"}),
                                        row=1, col=i + 1)
                figs[country].add_trace(go.Box(y=year_df.query("mode=='optimized'")["Load factor"],
                                               x=year_df.query("mode=='optimized'")["season"],
                                               name="optimized",
                                               notched=True,
                                               legendgroup="optimized",
                                               showlegend=True if i == 0 else False,
                                               line={"color": "firebrick"}),
                                        row=1, col=i + 1)

        # Create menu
        update_menus = [{
            'buttons': [],
            'direction': 'down',
            'showactive': True
        }]
        for key in figs:
            update_menus[0]['buttons'].append({
                'method': 'update',
                'label': str(key),
                'args': [{
                    'customdata': [dat.customdata for dat in figs[key].data],
                    'hovertemplate': [dat.hovertemplate for dat in figs[key].data],
                    'x': [dat.x for dat in figs[key].data],
                    'y': [dat.y for dat in figs[key].data]
                }, {
                    'yaxis': figs[key].layout.yaxis
                }]
            })

        # update_menus[0]['type'] = menu_type
        # Create single figure
        fig = figs[list(figs)[0]].update_layout(
            updatemenus=update_menus,
            title="Load Factor",
            boxmode="group",
            yaxis_title="Load Factor (%)"
        )
        fig.update_xaxes(type='category')

        # Finalize figure
        fig.update_layout({
            'legend_title': None,
            'legend_font_size': 18,
            'legend_tracegroupgap': 20,
            'width': 1000,
            'height': 400
        })
        fig.show()

    def seasons_to_single_array(self, season_dict: dict) -> Dict[int, list]:
        """ the dicts with price ID as keys and 4 lists with the values for each day in each season will be reverted
        to a dict with the price ID as keys and one single list as values"""
        for key, values in season_dict.items():
            if isinstance(values, (
                    list, tuple, set, np.ndarray)):  # if only one value (np float) is there, no seasons are present
                season_dict[key] = [item for sublist in values for item in sublist]
            else:
                season_dict[key] = [values]
        return season_dict

    def EU27_fig(self, long_df: pd.DataFrame, y: str, title: str, box=True):
        """ if box=True, box plot is created, else a barplot is created"""
        number_years = len(long_df["year"].unique())
        if box:
            fig = px.box(long_df,
                         x="country",
                         y=y,
                         color="price_id",
                         boxmode="group",
                         facet_row="year",
                         notched=True,
                         title=title,
                         height=400 * number_years,
                         )
        else:
            fig = px.bar(long_df,
                         x="country",
                         y=y,
                         color="price_id",
                         barmode="group",
                         facet_row="year",
                         title=title,
                         height=400 * number_years,
                         )

        fig.update_xaxes(type='category')

        # Finalize figure
        fig.update_layout({
            'legend_title': None,
            'legend_font_size': 18,
            'legend_tracegroupgap': 20,
        })
        return fig

    def combine_seasonal_dicts_to_df(self, ref_dict: dict, opt_dict: dict, year: int, country: str,
                                     big_df: pd.DataFrame,
                                     value_name: str) -> pd.DataFrame:
        # reverse the seasonal arrays into one single array:
        opt = self.seasons_to_single_array(opt_dict)
        opt_df = pd.DataFrame.from_dict(opt)
        column_name = f"optimized price {int(opt_df.columns[0])}"
        opt_df.columns = [value_name]
        opt_df["price_id"] = column_name

        ref = self.seasons_to_single_array(ref_dict)
        ref_df = pd.DataFrame.from_dict(ref)
        ref_df.columns = [value_name]
        ref_df["price_id"] = "reference"

        df = pd.concat([ref_df, opt_df], axis=0)
        df["year"] = year
        df["country"] = country
        return pd.concat([big_df, df], axis=0)

    def grab_EU27_total_shifted_electricity(self) -> pd.DataFrame:
        big_df = pd.DataFrame()
        for project in self.projects:
            try:
                LPA = LoadProfileAnalyser(project_name=project['name'],
                                          year=project['year'],
                                          country=project['country'])
                shifted = LPA.calc_total_shifted_electricity_building_stock()
                df = pd.DataFrame.from_dict(shifted, orient="index", columns=["shifted electricity (MWh)"])
                df["price_id"] = df.index
                df["year"] = project['year']
                df["country"] = project['country']

                big_df = pd.concat([big_df, df], axis=0).reset_index(drop=True)
            except:
                print(f"shifted electricity for {project['country']} {project['year']} could not be loaded.")
        big_df["shifted electricity (MWh)"] = big_df["shifted electricity (MWh)"].astype(float)
        return big_df

    def grab_EU27_percentage_shifted_electricity(self) -> pd.DataFrame:
        big_df = pd.DataFrame()
        for project in self.projects:
                try:
                    LPA = LoadProfileAnalyser(project_name=project['name'],
                                              year=project['year'],
                                              country=project['country'])
                    shifted = LPA.calc_percentage_shifted_electricity_building_stock()
                    df = pd.DataFrame.from_dict(shifted, orient="index", columns=["shifted electricity (%)"])
                    df["price_id"] = df.index
                    df["year"] = project['year']
                    df["country"] = project['country']

                    big_df = pd.concat([big_df, df], axis=0).reset_index(drop=True)
                except:
                    print(f"shifted electricity for {project['country']} {project['year']} could not be loaded.")
        big_df["shifted electricity (%)"] = big_df["shifted electricity (%)"].astype(float)
        return big_df

    def plotly_EU27_shifted_electricity(self) -> plotly.graph_objects.Figure:
        df = self.grab_EU27_percentage_shifted_electricity()
        figure = self.EU27_fig(long_df=df, y="shifted electricity (%)", title="Shifted electricity", box=False)
        return figure

    def plotly_EU27_p2p(self):
        big_df = pd.DataFrame()
        for project in self.projects:
                LPA = LoadProfileAnalyser(project_name=project['name'],
                                          year=project['year'],
                                          country=project['country'])
                ref_p2p, opt_p2p = LPA.analyze_peak_to_peak_difference_total_load_profiles()
                big_df = self.combine_seasonal_dicts_to_df(ref_dict=ref_p2p,
                                                           opt_dict=opt_p2p,
                                                           year=project['year'],
                                                           country=project['country'],
                                                           big_df=big_df,
                                                           value_name="peak to peak demand (MW)")

        figure = self.EU27_fig(long_df=big_df, y="peak to peak demand (MW)", title="Peak to peak load")
        return figure

    def grab_EU27_load_factor(self):
        big_df = pd.DataFrame(columns=['Load Factor (%)', 'price_id', 'year', 'country'])
        for project in self.projects:
                try:
                    LPA = LoadProfileAnalyser(project_name=project['name'],
                                              year=project['year'],
                                              country=project['country'])
                    ref_factor, opt_factor = LPA.analyze_load_factor()
                    big_df = self.combine_seasonal_dicts_to_df(ref_dict=ref_factor,
                                                               opt_dict=opt_factor,
                                                               year=project['year'],
                                                               country=project['country'],
                                                               big_df=big_df,
                                                               value_name="Load Factor (%)")
                except:
                    print(f"Load Factor for {project['country']} {project['year']} could not be loaded!")
        return big_df

    def plotly_EU27_load_factor(self):
        df = self.grab_EU27_load_factor()
        figure = self.EU27_fig(long_df=df, y="Load Factor (%)", title="Load Factor")
        return figure

    def grab_EU27_self_consumption(self, filter=None):
        big_df = pd.DataFrame(columns=['PV self consumption (%)', 'price_id', 'year', 'country'])
        for project in self.projects:
                try:
                    LPA = LoadProfileAnalyser(project_name=project['name'],
                                              year=project['year'],
                                              country=project['country'])
                    ref_consumption, opt_consumption = LPA.calc_country_self_consumption(filter=filter)
                    # check if ref and opt consumption exist, if they don't scenarios are all without PV:
                    if len(ref_consumption) == 0:
                        ref_consumption, opt_consumption = {1: np.float(0)}, {
                            1: np.float(0)}  # careful if there are other prices
                    big_df = self.combine_seasonal_dicts_to_df(ref_dict=ref_consumption, opt_dict=opt_consumption,
                                                               year=project['year'],
                                                               country=project['country'],
                                                               big_df=big_df,
                                                               value_name="PV self consumption (%)")
                except:
                    print(f"Self consumption for {project['country']} {project['year']} could not be loaded!")
        big_df["PV self consumption (%)"] = big_df["PV self consumption (%)"].astype(float)
        # drop countries where self-consumption is 0 because they distort the scale
        return_df = big_df[big_df["PV self consumption (%)"] != 0]
        return return_df

    def plotly_EU27_self_consumption(self):
        df = self.grab_EU27_self_consumption()
        figure = self.EU27_fig(long_df=df, y="PV self consumption (%)", title="PV self-consumption", box=False)
        return figure

    def grab_EU27_pv_share(self):
        big_df = pd.DataFrame()
        for project in self.projects:
                try:
                    LPA = LoadProfileAnalyser(project_name=project['name'], year=project['year'], country=project['country'])
                    pv_share_optimization = LPA.calc_pv_share_building_stock_optimization().reset_index(drop=False).rename(columns={"ID_EnergyPrice": "price_id"})
                    pv_share_ref = LPA.calc_pv_share_building_stock_reference().reset_index(drop=False).rename(columns={"ID_EnergyPrice": "price_id"})
                    pv_share_ref["price_id"] = pv_share_ref["price_id"].apply(lambda x: f"reference_{x}")
                    new_df = pd.concat([pv_share_optimization, pv_share_ref], axis=0).rename(columns={0: "PV share (%)"})
                    new_df["country"] = project['country']
                    new_df["year"] = project['year']
                    big_df = pd.concat([big_df, new_df])

                except:
                    print(f"Self consumption for {project['country']} {project['year']} could not be loaded!")
        big_df["PV share (%)"] = big_df["PV share (%)"].astype(float)
        # drop countries where self-consumption is 0 because they distort the scale
        return_df = big_df[big_df["PV share (%)"] != 0]
        return return_df

    def force_load_analyzation_of_all_scenarios(self):
        """ creates the load analyzation files for all provided scenarios new by calling the
        LoadProfileAnalyzation class in multiprocessing"""

        input_list = [(LoadProfileAnalyser, project["name"], project["country"], project["year"]) for project in
                      self.projects]
        # get physical cores:
        cores = int(multiprocessing.cpu_count() / 2)

        with multiprocessing.Pool(cores) as pool:
            pool.starmap(self.unwrap_self_create_load_analyzation_files, input_list)

    @staticmethod
    def unwrap_self_create_load_analyzation_files(obj, name, country, year):
        obj(name, country, year).create_load_analyzation_files()

    def main(self):
        # self.force_load_analyzation_of_all_scenarios()  # way faster if all the profiles do not exist yet
        # self.plotly_aggregated_load_profiles()
        # self.plotly_PV_self_consumption()
        # self.plotly_load_factor()
        # self.plotly_peak_to_peak()
        # self.plotly_EU27_shifted_electricity()
        self.plotly_EU27_p2p()
        # self.plotly_EU27_load_factor()
        self.plotly_EU27_self_consumption()


if __name__ == "__main__":
    years = [2020, 2030]
    countries = ["DEU", "AUT"]
    CountryResultPlots(countries=countries, years=years, project_prefix="D5.4").grab_EU27_pv_share()
    CountryResultPlots(countries=countries, years=[2050], project_prefix="D5.4").main()
