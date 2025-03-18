from dash import Dash, html
from dash_bootstrap_components.themes import BOOTSTRAP
import multiprocessing
from joblib import Parallel, delayed
from flex_operation.LoadProfileAnalysation import LoadProfileAnalyser
import dash_visualization.layout as layout


YEARS = [2020, 2030, 2040, 2050]
COUNTRIES = [
    'BEL',
    'AUT',
    'HRV',
    "CYP",
    'CZE',
    'DNK',
    'FRA',
    'DEU',
    'LUX',
    'HUN',
    'IRL',
    'ITA',
    'NLD',
    'POL',
    'PRT',
    'SVK',
    'SVN',
    'ESP',
    'SWE',
    'GRC',
    'LVA',
    'LTU',
    'MLT',
    'ROU',
    'BGR',
    'FIN',
    'EST',
]
PROJECT_PREFIX = "D5.4"


def process_inst(country, year):
    try:
        inst = LoadProfileAnalyser(project_name=f"{PROJECT_PREFIX}_{country}_{year}",
                                   country=country,
                                   year=year)
        inst.create_load_analyzation_files(overwrite=False)
    except:
        print(f"Load Analyzing files could not be created for {country} {year}")

def create_files():
    cores = int(multiprocessing.cpu_count() / 2)
    arg_list = [(country, year) for country in COUNTRIES for year in YEARS]
    Parallel(n_jobs=cores)(delayed(process_inst)(*inst) for inst in arg_list)


    # for country in COUNTRIES:
    #     for year in YEARS:
    #         print(f"{country} {year}")
    #         LoadProfileAnalyser(f"D5.4_{country}_{year}", country, year).create_load_analyzation_files()


def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "FLEX Results"
    app.layout = layout.create_layout(app)
    app.run_server(debug=True)


if __name__ == "__main__":
    create_files()
    main()
