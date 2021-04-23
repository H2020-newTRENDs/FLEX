from pathlib import Path
from Heatdemand_rc_model import Heatdemand_rc_model
import os

# define if new data is calculated or just loaded:
load_data = False
print("load data is set to " + str(load_data) + '\n')

# Read data:
# base_results_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "Prosumager/_Philipp/inputdata")
project_directory_path = Path(__file__).parent.resolve()
base_results_path = project_directory_path / "inputdata"

# check if "outputdata folder exists:
output_path = project_directory_path / "outputdata"
try:
    os.makedirs(output_path)
except FileExistsError:
    pass

# define scenarios:
# scenariolist = [r"_scen_aut_cheetah_ref_install_iopt_dh_95"]
scenariolist = [r'_scen_aut_a00_WAM_plus_v6_ren_mue']

# start and end year:
start_year = 2020
end_year = 2050

# Grid reference year:
grid_reference_year = 2010

for scn in scenariolist:
    results_path = base_results_path / scn
    run_number_str = "001"
    results_path_rcm = results_path / "Dynamic_Calc_Input_Data"
    results_path_temperatures = results_path / "Annual_Climate_Data"
    results_path_FWKWK_data = results_path / "FWKWK_REGIONAL_DATA"

    # check if paths exist:
    print("rcm path exists: " + str(results_path_rcm.exists()))
    print("temperatures path exists: " + str(results_path_temperatures.exists()))
    print("FWKWK path exists: " + str(results_path_FWKWK_data.exists()))

    year_vektor = [start_year]#, end_year]

    for year in year_vektor:
        Heatdemand_rc_model(results_path_rcm, results_path_FWKWK_data, results_path_temperatures,
                            run_number_str, year, load_data)
