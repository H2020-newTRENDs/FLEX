import pyomo.environ as pyo
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from Core_rc_model import rc_heating_cooling
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A1_CONS import CONS

def create_building_dataframe():
    #%%
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    endungen =  {"_AUT": 20, "_GER": 5}
    for endung, id_country in endungen.items():
        dateiName_dynamic = "001__dynamic_calc_data_bc_2015" + endung + ".csv"
        dateiName_classes = "001_Building_Classes_2015" + endung + ".csv"
        dynamic_calc = pd.read_csv(base_input_path / dateiName_dynamic,
                                   sep=None,
                                   engine="python")\
                                    .dropna().drop(columns=["average_effective_area_wind_west_east_red_cool",
                                                            "average_effective_area_wind_south_red_cool",
                                                            "average_effective_area_wind_north_red_cool",
                                                            "spec_int_gains_cool_watt"])
        building_classes = pd.read_csv(base_input_path / dateiName_classes,
                                       sep=None,
                                       engine="python").dropna().reset_index(drop=True)
        building_df = pd.concat([building_classes, dynamic_calc], axis=1)
        # drop a lot of columns:
        building_df = building_df.drop(columns=[
                                        "building_life_time_index",
                                        "length_of_building",
                                        "width_of_building",
                                        "number_of_floors",
                                        "room_height",
                                        "building_geometry_special_index",
                                        "percentage_of_building_surface_attached_length",
                                        "percentage_of_building_surface_attached_width",
                                        "share_of_window_area_on_gross_surface_area",
                                        "share_of_windows_oriented_to_south",
                                        "share_of_windows_oriented_to_north",
                                        "gebaudebauweise_fbw",
                                        "renovation_facade_year_start",
                                        "renovation_facade_year_end",
                                        "f_x3_facade",
                                        "f_x2_facade",
                                        "f_x1_facade",
                                        "f_x0_facade",
                                        "renovation_windows_year_start",
                                        "renovation_windows_year_end",
                                        "f_x3_windows",
                                        "f_x2_windows",
                                        "f_x1_windows",
                                        "f_x0_windows",
                                        "envelope_quality_def_index",
                                        "mech_ventilation_system_index",
                                        "user_profiles_index",
                                        "climate_region_index",
                                        "agent_mixes_index",
                                        "bc_specific_userfactor",
                                        "INPUT_DATA_STOP",
                                        "grossfloor_area",  # ist = Af
                                        "heated_area",
                                        "total_vertical_surface_area",
                                        "aewd",
                                        "areafloor",
                                        "areadoors",
                                        "grossvolume",
                                        "heatedvolume",
                                        "heated_norm_volume",
                                        "characteristic_length",
                                        "A_V_ratio",
                                        "LEK",
                                        "hwb",
                                        "hlpb",
                                        "orig_hlpb",
                                        "hlpb_dhw",
                                        "t_indoor_eff_factor",
                                        "effective_indoor_temp_jan",
                                        "prev_index",
                                        "last_action",
                                        "last_action_year",
                                        "mean_construction_period",
                                        "uedh_sh_norm",
                                        "uedh_sh_effective",
                                        "uedh_sh_original",
                                        "climate_change_factor1",
                                        "uedh_sh_climate1",
                                        "climate_change_factor1a",
                                        "uedh_sh_climate1a",
                                        "climate_change_factor1b",
                                        "uedh_sh_climate1b",
                                        "climate_change_factor2",
                                        "uedh_sh_climate2",
                                        "ued_cool",
                                        "ued_cool_climate1",
                                        "ued_cool_climate1a",
                                        "ued_cool_climate1b",
                                        "ued_cool_climate2",
                                        "ued_cool_no_int_gains",
                                        "uedh_sh_savings_mech_vent",
                                        "effective_heating_hours",
                                        "effective_operation_hours_solar",
                                        "boiler_operation_hours_dhw",
                                        "distribution_operation_hours_dhw",
                                        "rand_number_1",
                                        "rand_number_2",
                                        "rand_number_3",
                                        "u_value_ceiling",
                                        "u_value_exterior_walls",
                                        "u_value_windows1",
                                        "u_value_windows2",
                                        "u_value_roof",
                                        "u_value_zangendecke",
                                        "u_value_floor",
                                        "seam_loss_windows",
                                        "n_50",
                                        "envelope_quality_add_data_idx",
                                        "add_building_life_time",
                                        "facade_life_time_factor",
                                        "windows_life_time_factor",
                                        "envelope_lifetime_index",
                                        "trans_loss_walls",
                                        "trans_loss_ceil",
                                        "trans_loss_wind",
                                        "trans_loss_doors",
                                        "trans_loss_floor",
                                        "trans_loss_therm_bridge",
                                        "trans_loss_ventilation",
                                        "total_heat_losses",
                                        "uedh_sh_effective_gains",
                                        "uedh_sh_effective_gains_solar",
                                        "uedh_sh_effective_gains_lighting",
                                        "uedh_sh_effective_gains_internal_other",
                                        "uedh_sh_effective_gains_internal_people",
                                        "uedh_sh_norm_gains_solar_transparent",
                                        "ued_cool_gains_solar_transparent",
                                        "ued_cool_gains_solar_opaque",
                                        "ued_cool_gains_lighting",
                                        "ued_cool_gains_internal_other",
                                        "ued_cool_gains_internal_people",
                                        "internal_cool_gains_lighting",
                                        "internal_gains_lighting",
                                        "internal_cool_gains_other_appliances",
                                        "internal_gains_other_appliances",
                                        "internal_cool_gains_people",
                                        "internal_gains_people",
                                        "uedh_sh_effective_losses_before_gains",
                                        "estimated_electricity_internal_gains",
                                        "estimated_electricity_not_internal_gains",
                                        "uedh_sh_effective_jan",
                                        "uedh_sh_effective_feb",
                                        "uedh_sh_effective_mar",
                                        "uedh_sh_effective_apr",
                                        "uedh_sh_effective_may",
                                        "uedh_sh_effective_jun",
                                        "uedh_sh_effective_jul",
                                        "uedh_sh_effective_aug",
                                        "uedh_sh_effective_sep",
                                        "uedh_sh_effective_oct",
                                        "uedh_sh_effective_nov",
                                        "uedh_sh_effective_dec",
                                        "uedh_sh_effective_gains_solar_jan",
                                        "uedh_sh_effective_gains_solar_feb",
                                        "uedh_sh_effective_gains_solar_mar",
                                        "uedh_sh_effective_gains_solar_apr",
                                        "uedh_sh_effective_gains_solar_may",
                                        "uedh_sh_effective_gains_solar_jun",
                                        "uedh_sh_effective_gains_solar_jul",
                                        "uedh_sh_effective_gains_solar_aug",
                                        "uedh_sh_effective_gains_solar_sep",
                                        "uedh_sh_effective_gains_solar_oct",
                                        "uedh_sh_effective_gains_solar_nov",
                                        "uedh_sh_effective_gains_solar_dec",
                                        "annual_mech_vent_volume_heat_recovery_mode",
                                        "annual_mech_vent_volume_non_heat_recovery_mode",
                                        "ued_cooling_jan",
                                        "ued_cooling_feb",
                                        "ued_cooling_mar",
                                        "ued_cooling_apr",
                                        "ued_cooling_may",
                                        "ued_cooling_jun",
                                        "ued_cooling_jul",
                                        "ued_cooling_aug",
                                        "ued_cooling_sep",
                                        "ued_cooling_oct",
                                        "ued_cooling_nov",
                                        "ued_cooling_dec",
                                        "renovation_inv_costs",
                                        "of_which_maintenance_inv_costs_part",
                                        "target_hwb",
                                        "target_hwb_asterix",
                                        "hwb_lc_factor",
                                        "target_hwb_u_value_adoption_factor",
                                        "hwb_original_u_values_using_days",
                                        "hwb_using_days",
                                        "immissionFlaeche_sommer_nachweis_B8110_3",
                                        "immissionsflaechenbezogenerLuftwechsel_sommer_nachweis_B8110_3",
                                        "erforderliche_speicherwirksameMasse_sommer_nachweis_B8110_3",
                                        "immissionsflaechenbezogene_speicherwirksameMasse_sommer_nachweis_B8110_3",
                                        "heatstoragemass_ratio_sommer_nachweis_B8110_3",
                                        "start_refurb_obligation_year",
                                        "is_unrefurbished",
                                        "deep_refurb_level",
                                        "delta_hwb_refurb",
                                        "average_hwb_unrefurbished_simulation_start",
                                        "average_hwb_unrefurbished_current_period",
                                        "hwb_ref_climate_zone",
                                        "is_unrefurbished_age_not_considered",
                                        "cum_inv_env_exist_build",
                                        "cum_sub_env_exist_build",
                                        "bc_index",
                                        "Tset",
                                        "user_profile",
                                        ])
        building_df["country_ID"] = id_country
        excel_name = "building_dataframe_2015" + endung + ".xlsx"
        building_df.to_excel(base_input_path / excel_name)
    #%%



def create_radiation_gains():
    data = DB().read_DataFrame(REG_Table().Sce_Weather_Radiation, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    return data.loc[:, "Radiation"].to_numpy()
# radiation = create_radiation_gains()

def get_elec_profile():
    data = DB().read_DataFrame(REG_Table().Sce_Demand_BaseElectricityProfile, conn=DB().create_Connection(CONS().RootDB))
    return data.loc[:, "BaseElectricityProfile"].to_numpy()


def get_out_temp():
    data = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, conn=DB().create_Connection(CONS().RootDB), ID_Country=20)
    temperature = data.loc[:, "Temperature"].to_numpy()
    return temperature



def get_invert_data():
    #%%

    project_directory_path = Path(__file__).parent.resolve()
    base_results_path = project_directory_path / "inputdata"

    data = pd.read_excel(base_results_path / "building_dataframe_2015_AUT.xlsx", engine="openpyxl").drop(columns="Unnamed: 0")
    # take only single family houses:
    data = data.loc[data["building_categories_index"]==2]

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
    # HWB_norm = data.loc[:, "hwb_norm"].to_numpy()

    # window areas in celestial directions
    Awindows_rad_east_west = data.loc[:, "average_effective_area_wind_west_east_red_cool"].to_numpy()
    Awindows_rad_south = data.loc[:, "average_effective_area_wind_south_red_cool"].to_numpy()
    Awindows_rad_north = data.loc[:, "average_effective_area_wind_north_red_cool"].to_numpy()

    # solar gains through windows TODO berechne auf Fensterfläche!
    # Qsol = create_radiation_gains()
    Q_sol_all = pd.read_csv(base_results_path / "directRadiation_himmelsrichtung.csv", sep=";")
    Q_sol_north = np.outer(Q_sol_all.loc[:, "RadiationNorth"].to_numpy(), Awindows_rad_north)
    Q_sol_south = np.outer(Q_sol_all.loc[:, "RadiationSouth"].to_numpy(), Awindows_rad_south)
    Q_sol_east_west = np.outer((Q_sol_all.loc[:, "RadiationEast"].to_numpy() +
                                Q_sol_all.loc[:, "RadiationWest"].to_numpy()), Awindows_rad_east_west)
    Q_sol = Q_sol_north + Q_sol_south + Q_sol_east_west



    # electricity price for 24 hours:
    elec_price = pd.read_excel(base_results_path / "Elec_price_per_hour.xlsx", engine="openpyxl")
    elec_price = elec_price.loc[:, "Euro/MWh"].dropna().to_numpy() + 0.1

    # elec_price = DB().read_DataFrame(REG().Sce_Price_HourlyElectricityPrice, conn=DB().create_Connection(CONS().RootDB),
    #                                  ID_ElectricityPriceType=2).loc[:, "HourlyElectricityPrice"].to_numpy()

    # elec_price = np.array([0.1]*8760)


    return Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Q_sol, elec_price, Af

def create_dict(liste):
    dictionary = {}
    for index, value in enumerate(liste, start=1):
        dictionary[index] = value
    return dictionary


Atot, Hve, Htr_w, Hop, Cm, Am, Qi, Q_solar, price, Af = get_invert_data()
temperature = get_out_temp()
temp_outside = temperature


# fixed starting values:
tank_starting_temp = 50
thermal_mass_starting_temp = 16

# constants:
# water mass in storage
m_water = 10_000  # l
# thermal capacity water
cp_water = 4.2  # kJ/kgK
# constant surrounding temp for water tank
T_a = 20

At = 4.5  # 7.2.2.2

# Kopplung Temp Luft mit Temp Surface Knoten s
his = np.float_(3.45)  # 7.2.2.2
# kopplung zwischen Masse und  zentralen Knoten s (surface)
hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
# thermischer Kopplungswerte W/K
Htr_is = his * Atot
# Equ. C.1
PHI_ia = 0.5 * Qi

# Equ. C.6
Htr_1 = 1 / (1 / Hve + 1 / Htr_is)
# Equ. C.7
Htr_2 = Htr_1 + Htr_w
# Equ.C.8
Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)

# minimum room temperature
T_air_min = 20
# maximum room temperature
T_air_max = 25



def create_pyomo_model(elec_price, tout, Qsol, Am, Atot, Cm, Hop, Htr_1, Htr_2, Htr_3, Htr_em, Htr_is, Htr_ms,
                       Htr_w, Hve, PHI_ia, Qi, COP):
    # model
    m = pyo.AbstractModel()

    # parameters
    m.time = pyo.RangeSet(len(elec_price))   # later just len(elec_price) for whole year
    # electricity price
    m.p = pyo.Param(m.time, initialize=create_dict(elec_price))
    # outside temperature
    m.T_outside = pyo.Param(m.time, initialize=create_dict(tout))
    # solar gains
    m.Q_sol = pyo.Param(m.time, initialize=create_dict(Qsol))


    # variables
    # energy used for heating
    m.Q_heating = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # energy used for cooling
    m.Q_cooling = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 15_000))
    # real indoor temperature
    m.T_room = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(T_air_min, T_air_max))
    # thermal mass temperature
    m.Tm_t = pyo.Var(m.time, within=pyo.NonNegativeReals, bounds=(0, 50))


    # objective
    def minimize_cost(m):
        return sum((m.Q_heating[t] + m.Q_cooling[t]) / COP * m.ElectricityPrice[t] for t in m.time)
    m.OBJ = pyo.Objective(rule=minimize_cost)


    # constraints
    def thermal_mass_temperature_rc(m, t):
        if t == 1:
            return m.Tm_t[t] == 15

        else:
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])

            # T_sup = T_outside because incoming air for heating and cooling ist not pre-heated/cooled
            T_sup = m.T_outside[t]
            # Equ. C.5
            PHI_mtot = PHI_m + Htr_em * m.T_outside[t] + Htr_3 * (
                    PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (((PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve) +
                                                               T_sup)) / Htr_2

            # Equ. C.4
            return m.Tm_t[t] == (m.Tm_t[t - 1] * ((Cm/3600) - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot) / (
                                (Cm/3600) + 0.5 * (Htr_3 + Htr_em))
    m.thermal_mass_temperature_rule = pyo.Constraint(m.time, rule=thermal_mass_temperature_rc)


    def room_temperature_rc(m, t):
        if t == 1:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + thermal_mass_starting_temp) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
        else:
            # Equ. C.3
            PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * (0.5 * Qi + m.Q_sol[t])
            # Equ. C.9
            T_m = (m.Tm_t[t] + m.Tm_t[t-1]) / 2
            T_sup = m.T_outside[t]
            # Euq. C.10
            T_s = (Htr_ms * T_m + PHI_st + Htr_w * m.T_outside[t] + Htr_1 * (T_sup + (
                    PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / Hve)) / (Htr_ms + Htr_w + Htr_1)
            # Equ. C.11
            T_air = (Htr_is * T_s + Hve * T_sup + PHI_ia + m.Q_heating[t] - m.Q_cooling[t]) / (Htr_is + Hve)
            # Equ. C.12
            T_op = 0.3 * T_air + 0.7 * T_s
            # T_op is according to norm the inside temperature whereas T_air is the air temperature # TODO which one?
            return m.T_room[t] == T_air
    m.room_temperature_rule = pyo.Constraint(m.time, rule=room_temperature_rc)


    instance = m.create_instance(report_timing=True)
    opt = pyo.SolverFactory("gurobi")
    results = opt.solve(instance)#, tee=True)
    print(results)
    instance.display("./log.txt")
    return instance, m


def calculate_cost_diff(instance, Q_H, Q_C, price, COP):
    total_cost_optimized = instance.OBJ() / 100 / 1_000  #  €  (COP is already in instance)
    # total cost not optimized:
    total_cost_normal = sum((Q_H + Q_C) * price) / 100 / 1_000 / COP  # €

    # difference in energy consumption
    total_energy_optimized = sum(np.array([instance.Q_heating[t]()+instance.Q_cooling[t]() for t in m.time])) / 1_000 / COP  # kWh
    total_energy_normal = sum(Q_H+Q_C) / 1_000 / COP  # kWh

    x_ticks = [1, 2, 3, 4]
    width = 0.5
    fig = plt.figure()
    ax = plt.gca()
    ax2 = ax.twinx()
    colors = ["blue", "skyblue", "darkred", "orangered"]
    labels = ["total cost normal", "total cost optimized", "total energy normal", "total energy optimized"]
    ax.bar(1, total_cost_normal, color="#305496", edgecolor='black', width=width)
    ax.bar(2, total_cost_optimized, color='#8EA9DB', edgecolor='black', width=width)
    ax.bar(2, total_cost_normal-total_cost_optimized, color='#375623', edgecolor='black',
           bottom=total_cost_optimized, hatch="//", width=width)

    ax.text(1.85, total_cost_optimized + (total_cost_normal-total_cost_optimized)/2,
            str(round((total_cost_normal-total_cost_optimized) / total_cost_normal * 100, 2)) + " %")

    ax2.bar(3, total_energy_normal, color='#F47070', edgecolor='black', width=width)
    ax2.bar(4, total_energy_optimized, color="#F4B084", edgecolor='black', width=width)
    ax2.bar(3, total_energy_optimized-total_energy_normal, color="#FA9EFA", edgecolor='black',
            bottom=total_energy_normal, hatch="//", width=width)

    ax2.text(2.85, total_energy_normal + (total_energy_optimized-total_energy_normal)/2,
             str(round((total_energy_optimized-total_energy_normal) / total_energy_normal * 100, 2)) + " %")

    ax.set_xticks(x_ticks)
    ax.set_ylabel("EUR")
    ax2.set_ylabel("kWh")
    plt.xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=15)
    plt.title("Cost and Energy difference")
    plt.tight_layout()
    plt.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Cost_energy_diff_total.png")
    plt.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Cost_energy_diff_total.svg")
    plt.show()

# create plots to visualize results
def show_results(instance, m, Q_H_noDR, Q_C_noDR, Tm_t, T_room_noDR, Q_solar, price, temp_outside, elec_profile, COP):
    # colorcode:
    # temp_outside = darkblue
    # indoor temp DR = darkgreen
    # indoor temp noDR = purple -- #AC0CB0
    # price = pink
    # solar gains = yellow
    # heating DR = red
    # cooling DR = blue
    # heating noDR = orange --
    # cooling noDR = green --
    # Electricity no DR = grey --
    # Electricity DR = turquoise #3BE0ED
    red = '#F47070'
    blue = '#8EA9DB'
    green = '#A9D08E'
    orange = '#F4B084'
    yellow = '#FFD966'
    grey = '#C9C9C9'
    pink = '#FA9EFA'
    dark_green = '#375623'
    dark_blue = '#305496'
    purple = '#AC0CB0'
    turquoise = '#3BE0ED'
    colors = {"temp_outside": dark_blue,
              "T_room": dark_green,
              "T_room_noDR": purple,
              "price": pink,
              "solar_gains": yellow,
              "Q_heating": red,
              "Q_cooling": blue,
              "Q_H_noDR": orange,
              "Q_C_noDR": green,
              "Electricity_noDR": grey,
              "Electricity_DR": turquoise}


    Q_heating = np.array([instance.Q_heating[t]() for t in m.time]) / 1_000  # kW
    Q_cooling = np.array([instance.Q_cooling[t]() for t in m.time]) / 1_000  # kW
    T_room = [instance.T_room[t]() for t in m.time]
    T_mass_mean = [instance.Tm_t[t]() for t in m.time]
    total_cost = instance.OBJ()

    x_achse = np.arange(len(Q_heating))
    # Heating powers and Temperatures
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax2.plot(x_achse, T_mass_mean, label="thermal mass optimized", color=grey)
    # ax2.plot(x_achse, Tm_t, label="thermal mass normal", color="skyblue", linestyle="--")
    ax2.plot(x_achse, temp_outside, label="outside temperature", color=colors["temp_outside"])
    ax2.plot(x_achse, T_room, label="indoor temperature DR", color=colors["T_room"])
    ax2.plot(x_achse, T_room_noDR, label="indoor temperature no DR", color=colors["T_room_noDR"])

    ax1.plot(x_achse, Q_heating, label="heating DR", color=colors["Q_heating"], alpha=0.8)
    ax1.plot(x_achse, Q_cooling, label="cooling DR", color=colors["Q_cooling"], alpha=0.8)
    ax1.plot(x_achse, Q_H_noDR / 1_000, label="heating no DR", color=colors["Q_H_noDR"], linestyle="--")
    ax1.plot(x_achse, Q_C_noDR / 1_000, label="cooling no DR", color=colors["Q_C_noDR"], linestyle="--")
    ax1.plot(x_achse, Q_solar / 1_000, label="solar gains", color=colors["solar_gains"], alpha=0.5)

    ax1.set_title("Heating power and temperatures")
    ax2.set_ylabel("temperature in °C")
    ax1.set_ylabel("heating power in kWh")
    ax2.set_xlabel("time in hours")
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    fig.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Heat_Temp.png")
    fig.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Heat_Temp.svg")
    plt.show()


    # Compare Electricity consumption graph with price
    fig2 = plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    # ax1.plot(x_achse, (Q_heating+Q_cooling)/COP, label="electricity DR", color=colors["Electricity_DR"])
    # ax1.plot(x_achse, (Q_H_noDR+Q_C_noDR)/COP/1_000, label="electricity no DR", linestyle="--", color=colors["Electricity_noDR"])
    ax2.plot(x_achse, price, label="electricity price", color=colors["price"])
    plt.title(f"compare electricity loads with COP = {COP}")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    ax1.set_xlabel("time in hours")
    ax1.set_ylabel("electricity in kW")
    ax2.set_ylabel("price per kWh")
    ax1.grid(axis="x")
    plt.tight_layout()
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Elec_Price.png")
    fig2.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Elec_Price.svg")
    plt.show()


start_number_toplot = 0
number_hours_toplot = 8760
COP = 3

for i in range(1):
    Q_H_noDR, Tm_t, Q_C_noDR, T_air_noDR = rc_heating_cooling(Q_solar[start_number_toplot:number_hours_toplot, i], Atot, Hve, Htr_w, Hop, Cm,
                                                              Am, Qi, Af,
                                         temp_outside[start_number_toplot:number_hours_toplot],
                                                              initial_thermal_mass_temp=thermal_mass_starting_temp, T_air_min=T_air_min,
                                                              T_air_max=T_air_max)

    instance, m = create_pyomo_model(price[start_number_toplot:number_hours_toplot],
                                     temp_outside[start_number_toplot:number_hours_toplot],
                                     Q_solar[start_number_toplot:number_hours_toplot, i], Am[i], Atot[i], Cm[i], Hop[i],
                                     Htr_1[i], Htr_2[i], Htr_3[i], Htr_em[i], Htr_is[i], Htr_ms[i], Htr_w[i], Hve[i],
                                     PHI_ia[i], Qi[i], COP)

    elec_profile = get_elec_profile()[start_number_toplot:number_hours_toplot] * 1_000

    show_results(instance, m, Q_H_noDR[:, i], Q_C_noDR[:, i], Tm_t[:, i], T_air_noDR[:, i], Q_solar[start_number_toplot:number_hours_toplot, i],
                 price[start_number_toplot:number_hours_toplot], temp_outside[start_number_toplot:number_hours_toplot],
                 elec_profile, COP)

    calculate_cost_diff(instance, Q_H_noDR[:, i], Q_C_noDR[:, i], price[start_number_toplot:number_hours_toplot], COP)




