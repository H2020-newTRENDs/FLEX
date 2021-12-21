import pandas as pd
import os
from pathlib import Path
from Create_set_temp_profile import CREATE_SET_TEMP_PROFILE
from Create_dhw_energyneed_profile import CREATE_DHW_ENERGYDEMAND_PROFILE
from Prosumager._Philipp.Core_rc_model import core_rc_model
import h5py
import timeit
from Simple_plots import *


def read_h5(filename):
    print('reading hf file...')
    dict_ = {}
    hf = h5py.File(filename, "r")
    # save all arrays from hf to dict as np.array:
    for key in hf.keys():
        dict_[key] = np.array(hf.get(key))
    hf.close()
    print('done')
    return dict_


def save_to_h5(outputpath, h5_name, Q_H_LOAD_8760, Q_C_LOAD_8760, Q_DHW_LOAD_8760, Af, bc_num_building_not_Zero_vctr,
               climate_region_index, share_Circulation_DHW, T_e_HSKD_8760_clreg, Tset_heating_8760_up,
               Tset_cooling_8760_up):

    print('hf file is being saved...')
    starttimeh5 = timeit.default_timer()
    # check if folder exists:
    try:
        os.makedirs(outputpath)
    except FileExistsError:
        pass
    # create file object, h5_name is path and name of file
    hf = h5py.File(outputpath / h5_name, "w")
    # files can be compressed with compression="lzf" and compression_opts=0-9 to save space, but then it reads slower
    # renaming the variables from here on!
    hf.create_dataset("Heating_load", data=Q_H_LOAD_8760)
    hf.create_dataset("Cooling_load", data=Q_C_LOAD_8760)
    hf.create_dataset("DHW_load", data=Q_DHW_LOAD_8760)

    hf.create_dataset("Af", data=Af)
    hf.create_dataset("bc_num_building_not_Zero_vctr", data=bc_num_building_not_Zero_vctr)
    hf.create_dataset("climate_region_index", data=climate_region_index)

    hf.create_dataset("share_Circulation_DHW", data=share_Circulation_DHW)
    hf.create_dataset("T_outside", data=T_e_HSKD_8760_clreg)
    hf.create_dataset("T_set_heating", data=Tset_heating_8760_up)
    hf.create_dataset("T_set_cooling", data=Tset_cooling_8760_up)
    # save data
    hf.close()
    print('saving hf completed')
    print("Time for saving to h5: ", timeit.default_timer() - starttimeh5)


# This calculation is done after DIN EN ISO 13790:2008.
# input data is either provided as a single value for each building or as an 8760 array for every hour of the year:
# The function needs the solar radiation, DHW need per day, the outside temperature profile, the indoor set temperature
# for heating and cooling and the data for each building. The solar radiation is a (x, 36) array, where x stands for the
# number of regions for which solar radiation is provided. Each building has a "climate region index" which is between 0
# and x. The first 12 columns of the solar radiation represent the solar radiation from north for each month. Columns 12
# to 23 represent the radiation from east and west and columns 24 to 35 are the southern radiation. The "data" is a
# pandas matrix which contains all relevant building information like climate region index, Floor area, transmission
# coefficient, thermal mass, effective window area in celestial directions and the "user profile". The user profile is
# the index with a user is assigned to each building. The user profile contains  indoor set temperatures and times when
# the building is used.
def prepare_core_input_data(sol_rad, data):
    # function converts all neccesary input data to numpy arrays so loops can be accelerated with numba:
    sol_rad = sol_rad.drop(columns=["climate_region_index"]).to_numpy()
    # climate region index - 1 because python starts to count at zero:
    climate_region_index = data.loc[:, "climate_region_index"].to_numpy().astype(int) - 1
    # user Profile index -1 because python starts indexing at zero compared to matlab
    UserProfile_idx = data.loc[:, "user_profile"].to_numpy().astype(int) - 1
    unique_climate_region_index = np.unique(climate_region_index).astype(int)

    sol_rad_north = np.empty((1, 12), float)
    sol_rad_east_west = np.empty((1, 12))
    sol_rad_south = np.empty((1, 12))
    for climate_region in unique_climate_region_index:
        anzahl = len(np.where(climate_region_index == climate_region)[0])
        sol_rad_north = np.append(sol_rad_north,
                                  np.tile(sol_rad[int(climate_region), np.arange(12)], (anzahl, 1)),
                                  axis=0)  # weil von 0 gezählt
        sol_rad_east_west = np.append(sol_rad_east_west,
                                      np.tile(sol_rad[int(climate_region), np.arange(12, 24)], (anzahl, 1)), axis=0)
        sol_rad_south = np.append(sol_rad_south,
                                  np.tile(sol_rad[int(climate_region), np.arange(24, 36)], (anzahl, 1)), axis=0)

    # delete first row in solar radiation as its just zeros
    sol_rad_north = np.delete(sol_rad_north, 0, axis=0)
    sol_rad_east_west = np.delete(sol_rad_east_west, 0, axis=0)
    sol_rad_south = np.delete(sol_rad_south, 0, axis=0)

    # konditionierte Nutzfläche
    Af = data.loc[:, "Af"].to_numpy()
    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af  # 7.2.2.2
    # Airtransfercoefficient
    Hve = data.loc[:, "Hve"].to_numpy()
    # Transmissioncoefficient wall
    Htr_w = data.loc[:, "Htr_w"].to_numpy()
    # Transmissioncoefficient opake Bauteile
    Hop = data.loc[:, "Hop"].to_numpy()
    # Speicherkapazität J/K
    Cm = data.loc[:, "CM_factor"].to_numpy() * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = data.loc[:, "Am_factor"].to_numpy() * Af
    # internal gains
    Qi = data.loc[:, "spec_int_gains_cool_watt"].to_numpy() * Af
    HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    # Days per Month
    DpM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # hours per day
    h = np.arange(1, 25)
    # sol_hx = min(14, max(0, h + 0.5 - 6.5))
    # Verteilung der Sonneneinstrahlung auf die einzelnen Stunden mit fixem Profil:
    # Minimalwert = 0 und Maximalwert = 14
    sol_hx = []
    for i in h:
        if h[i - 1] + 0.5 - 6.5 < 0:
            sol_hx.append(0)
        elif h[i - 1] + 0.5 - 6.5 > 14:
            sol_hx.append(14)
        else:
            sol_hx.append(h[i - 1] + 0.5 - 6.5)

    # sol_rad_norm = max(0, np.sin(3.1415 * sol_hx / 13)) / 8.2360
    # Sinusprofil für die Sonneneinstrahlung:
    sol_rad_norm = []
    for i in sol_hx:
        if np.sin(3.1415 * i / 13) / 8.2360 < 0:
            sol_rad_norm.append(0)
        else:
            sol_rad_norm.append(np.sin(3.1415 * i / 13) / 8.2360)

    sol_rad_norm = np.array(sol_rad_norm)
    Af = np.tile(Af, (8760, 1)).T
    # Number of building household:
    num_bc = len(Qi)
    print("Number of building household: " + str(num_bc))
    daylist = np.zeros((num_bc, 24))
    monthlist = np.zeros((num_bc, 12))
    yearlist = np.zeros((num_bc, 8760))
    return sol_rad_north, sol_rad_south, sol_rad_east_west, sol_rad_norm, Af, Atot, Hve, Htr_w, Hop, Cm, Am, Qi, \
           Awindows_rad_east_west, Awindows_rad_north, Awindows_rad_south, DpM, UserProfile_idx, num_bc, \
           climate_region_index, daylist, monthlist, yearlist


def Heatdemand_rc_model(OUTPUT_PATH, OUTPUT_PATH_NUM_BUILD, OUTPUT_PATH_TEMP, RN, YEAR, load_data):
    print("start heatdemand_rc_model")
    # TODO für testen:
    YEAR = 2050

    BCAT_1_3 = np.ones((6, 3))
    BCAT_4_8 = np.ones((1, 5))
    NUM_GFA_BEFORE_BCAT_1_3 = np.ones((6, 3))
    NUM_GFA_BEFORE_BCAT_4_8 = np.ones((1, 5))
    NUM_GFA_AFTER_BCAT_1_3 = np.ones((6, 3))
    NUM_GFA_AFTER_BCAT_4_8 = np.ones((1, 5))

    # for k in range(1, 7): weeggeben weil nur relevant für ein spezifisches projekt
    # TODO diese if abfragen in funktion ändern weil blöd für andere länder
    # if k == 1:
    # Weichstätten:
    nominal_gridloss_factor = 0.16
    region_obs_data_file_name = "Austria"
    t_movavg_DHW = 2
    t_movavg_SH = 3
    share_Circulation_DHW = 0
    GE_IDX = 1061

    # Wohngebäude mit je 6 Bauperioden (4 historisch, 5 Leer, 6 Neubau)
    # BCAT_1_3 = [1.0,0.5, 0.2, 0.2, 0.0, 1] *[1, 0.5, 0.3]
    # TODO Andi fragen was es mit den BCAT matrizen auf sich hat ud was share_Circulation_DHW bedeuted!
    BCAT_4_8[0, -2:] = 0

    TGridmom = 75
    TGridmin = 75
    scale_DHW = 1

    # output_file_name combines region and climate data file names
    output_file_name = region_obs_data_file_name + "_" + str(YEAR)

    # load building stock data exportet by invert run:
    datei = str(RN) + "__dynamic_calc_data_bc_" + str(YEAR) + ".npz"
    data_np = np.load(OUTPUT_PATH / datei)
    data = pd.DataFrame(data=data_np["arr_0"], columns=data_np["arr_1"][0])
    data.columns = data.columns.str.decode("utf-8")
    data.columns = data.columns.str.replace(" ", "")

    datei_num_build = "_BC_BUI_GE_" + str(YEAR) + ".npz"
    data_num_build_per_GE_np = np.load(OUTPUT_PATH_NUM_BUILD / datei_num_build)
    data_num_build_per_GE = pd.DataFrame(data=data_num_build_per_GE_np["arr_0"],
                                         columns=data_num_build_per_GE_np["arr_1"][0])
    data_num_build_per_GE.columns = data_num_build_per_GE.columns.str.decode("utf-8")
    data_num_build_per_GE.columns = data_num_build_per_GE.columns.str.replace(" ", "")

    datei_num_gfa_per_GE = "_BC_GFA_GE_" + str(YEAR) + ".npz"
    data_num_gfa_per_GE_np = np.load(OUTPUT_PATH_NUM_BUILD / datei_num_gfa_per_GE)
    data_num_gfa_per_GE = pd.DataFrame(data=data_num_gfa_per_GE_np["arr_0"],
                                       columns=data_num_gfa_per_GE_np["arr_1"][0])
    data_num_gfa_per_GE.columns = data_num_gfa_per_GE.columns.str.decode("utf-8")

    datei_num_gfa_per_GE = "_BC_BUILD_CAT_" + str(YEAR) + ".npz"
    data_Bcat_per_BC_np = np.load(OUTPUT_PATH_NUM_BUILD / datei_num_gfa_per_GE)
    data_Bcat_per_BC = pd.DataFrame(data=data_Bcat_per_BC_np["arr_0"], columns=data_Bcat_per_BC_np["arr_1"][0])
    data_Bcat_per_BC.columns = data_Bcat_per_BC.columns.str.decode("utf-8")

    if str(GE_IDX) in data_num_build_per_GE.columns:
        col = data_num_build_per_GE.columns.get_loc(str(GE_IDX))
    else:
        print("Gemeinde nicht gefunden!")

    if region_obs_data_file_name == "Wien":
        print("da musst du noch irgendeinen Shit machen")
    else:
        data_num_build_per_GE = data_num_build_per_GE.iloc[:, [0, col]]
        data_num_gfa_per_GE = data_num_gfa_per_GE.iloc[:, [0, col]]

    # time loops:
    starttime = timeit.default_timer()

    # Wohngebäude 1 bis 3
    # Bauperioden 1 bis 6
    if load_data == False:
        for i in range(1, 4):
            for j in range(1, 7):
                idx = data_Bcat_per_BC.loc[(data_Bcat_per_BC["bcat_map"] == i) &
                                           (data_Bcat_per_BC["constrp_map"] == j), :].index
                data_num_build_per_GE.iloc[idx, 1] = data_num_build_per_GE.iloc[idx, 1] * BCAT_1_3[
                    j - 1, i - 1]  # weil python bei 0 anfängt zu zählen
                data_num_gfa_per_GE.iloc[idx, 1] = data_num_gfa_per_GE.iloc[idx, 1] * BCAT_1_3[j - 1, i - 1]

                NUM_GFA_BEFORE_BCAT_1_3[j - 1, i - 1] = data_num_gfa_per_GE.iloc[idx, 1].sum()

        # für kategorien 4 bis 8:
        for i in range(1, 6):
            idx = data_Bcat_per_BC.loc[data_Bcat_per_BC["bcat_map"] == (i + 3)].index
            data_num_build_per_GE.iloc[idx, 1] = data_num_build_per_GE.iloc[idx, 1] * BCAT_4_8[0, i - 1]
            data_num_gfa_per_GE.iloc[idx, 1] = data_num_gfa_per_GE.iloc[idx, 1] * BCAT_4_8[0, i - 1]

            NUM_GFA_BEFORE_BCAT_4_8[0, i - 1] = data_num_gfa_per_GE.iloc[idx, 1].sum()

        # TODO Wieso dieses Limit?
        limit = 0.000002

        bc_num_building_not_Zero_vctr = (data_num_gfa_per_GE.iloc[:, 1] / data_num_gfa_per_GE.iloc[:,
                                                                          1].sum() > limit) & \
                                        (data_num_gfa_per_GE.iloc[:, 0] < data.iloc[:, 0].size)
        bc_idx_not_Zero = data_num_build_per_GE[bc_num_building_not_Zero_vctr].iloc[:, 0]
        bc_num_build = data_num_build_per_GE[bc_num_building_not_Zero_vctr].iloc[:, 1]
        print("Number of BC household: " + str(bc_num_build.size))
        bc_gfa = data_num_gfa_per_GE[bc_num_building_not_Zero_vctr].iloc[:, 1]
        Bcat_per_BC = data_Bcat_per_BC[bc_num_building_not_Zero_vctr]
        print("WG")
        for i in range(1, 4):
            print("WG type " + str(i))
            for j in range(1, 7):  # Buildingcategories
                idx = Bcat_per_BC.loc[(Bcat_per_BC["bcat_map"] == i) &
                                      (Bcat_per_BC["constrp_map"] == j), :].index
                NUM_GFA_AFTER_BCAT_1_3[j - 1, i - 1] = bc_gfa[idx].sum()
                ratio = NUM_GFA_BEFORE_BCAT_1_3[j - 1, i - 1] / max(0.000001, NUM_GFA_AFTER_BCAT_1_3[j - 1, i - 1])
                # TODO wieso 8? Ist das das maximale mögliche ratio oder einfach so?
                ratio = min(8, max(1, ratio))
                bc_gfa[idx] = bc_gfa[idx] * ratio
                bc_num_build[idx] = bc_num_build[idx] * ratio

        print("NWG")
        for i in range(1, 6):
            idx = Bcat_per_BC.loc[Bcat_per_BC["bcat_map"] == (i + 3)].index
            NUM_GFA_AFTER_BCAT_4_8[0, i - 1] = bc_gfa[idx].sum()
            ratio = NUM_GFA_BEFORE_BCAT_4_8[0, i - 1] / max(0.000001, NUM_GFA_AFTER_BCAT_4_8[0, i - 1])
            ratio = min(8, max(1, ratio))
            bc_gfa[idx] = bc_gfa[idx] * ratio
            bc_num_build[idx] = bc_num_build[idx] * ratio

        # reducing the data:
        data = data.set_index("bc_index")
        data_red = data.loc[bc_idx_not_Zero.values, :]

        # load temperature data:
        temp_data = pd.read_excel(OUTPUT_PATH_TEMP / "Input_Weather2015_AT.xlsx", engine="openpyxl")
        temp_8760 = temp_data.loc[:, "Temperature"].to_numpy()
        # convert numpy array into matrix:
        temp_8760 = np.tile(temp_8760, (len(data_red), 1))

        HoursPerYear = len(data_red)

        #  CREATE SET TEMPERATURE PROFILE
        Tset_heating_8760_up, Tset_cooling_8760_up = CREATE_SET_TEMP_PROFILE(RN, YEAR, OUTPUT_PATH)

        # CREATE DHW PROFILE
        DHW_need_day_m2_8760_up, DHW_loss_Circulation_040_day_m2_8760_up = \
            CREATE_DHW_ENERGYDEMAND_PROFILE(RN, YEAR, OUTPUT_PATH)

        # TODO implement data of building segments!
        # data of building segments (number of buildings)
        # data_bssh = load([OUTPUT_PATH, RN '_dynamic_calc_data_bssh_' num2str(YEAR) '.mat'])

        # solar radiation
        datei = RN + '__climate_data_solar_rad_' + str(YEAR) + ".csv"
        sol_rad = pd.read_csv(OUTPUT_PATH / datei)

        # inputdata for core is converted to numpy arrays:
        sol_rad_north, sol_rad_south, sol_rad_east_west, sol_rad_norm, Af, Atot, Hve, Htr_w, Hop, Cm, Am, Qi, \
        Awindows_rad_east_west, Awindows_rad_north, Awindows_rad_south, DpM, UserProfile_idx, num_bc, \
        climate_region_index, daylist, monthlist, yearlist = prepare_core_input_data(sol_rad, data_red)

        # core rc model after DIN EN ISO 13790: (only takes numpy arrays)
        starttime_core = timeit.default_timer()
        Q_H_LOAD_8760, Q_C_LOAD_8760, Q_DHW_LOAD_8760, Af = \
            core_rc_model(DHW_need_day_m2_8760_up, DHW_loss_Circulation_040_day_m2_8760_up,
                          share_Circulation_DHW, temp_8760, Tset_heating_8760_up, Tset_cooling_8760_up, sol_rad_north,
                          sol_rad_south, sol_rad_east_west, sol_rad_norm, Af, Atot, Hve, Htr_w, Hop, Cm, Am, Qi,
                          Awindows_rad_east_west, Awindows_rad_north, Awindows_rad_south, DpM, UserProfile_idx, num_bc,
                          daylist, monthlist, yearlist)
        print("Time for core calculation: ", timeit.default_timer() - starttime_core)

        # save data to h5 file for fast accessability later:
        saving_path = Path(__file__).parent.resolve()
        saving_path = saving_path / "outputdata"
        saving_name = 'Building_load_curve_' + output_file_name + '.hdf5'
        save_to_h5(saving_path, saving_name,
                   Q_H_LOAD_8760, Q_C_LOAD_8760, Q_DHW_LOAD_8760, Af, bc_num_building_not_Zero_vctr,
                   climate_region_index, share_Circulation_DHW, temp_8760, Tset_heating_8760_up, Tset_cooling_8760_up)

    # load the data from h5 file:
    saving_path = Path(__file__).parent.resolve()
    saving_path = saving_path / "outputdata"
    filename = 'Building_load_curve_' + output_file_name + '.hdf5'
    dict_ = read_h5(saving_path / filename)

    # print time
    print("Time for execution: ", timeit.default_timer() - starttime)

    # create a dict for a simple plot if the data is loaded as dict:
    dict2 = dict.fromkeys(["Cooling_load", "DHW_load", "Heating_load"], [])
    for key in dict2:
        dict2[key] = dict_[key]
    # create dict for subplot with outside and set temperatures
    dict3 = dict.fromkeys(["T_set_heating", "T_set_cooling", "T_outside"], [])
    for key in dict3:
        dict3[key] = dict_[key]

    # plot only the Heating cooling and dhw loads
    lineplot_plt(dict2)

    # plot heating cooling and DHW loads as well as temperature settings and outside temp:
    overview_core(dict2, dict3)

    one_day(dict2, dict3)

    a = 1
