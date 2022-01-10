import numpy as np
import pandas as pd
import numpy as np
import os
from pathlib import Path
from A_Infrastructure.A2_DB import DB
from A_Infrastructure.A1_CONS import CONS


def load_building_data_from_excel():
    base_path = Path().absolute().resolve()
    dynamic_data_path = Path("inputdata/AUT/001__dynamic_calc_data_bc_2017_AUT.csv")
    building_segment_path = Path("inputdata/AUT/040_aut__3__BASE__1_zz_new_bc_seg__b_building_segment_sh.csv")
    building_class_path = Path("inputdata/AUT/001_Building_Classes_2017.csv")

    dynamic_data = pd.read_csv(Path(base_path / dynamic_data_path), sep=None, engine="python")
    building_segment = pd.read_csv(Path(base_path / building_segment_path), sep=None, engine="python").drop(
        index=[0, 1, 2]).reset_index(drop=True)
    building_class = pd.read_csv(Path(base_path / building_class_path), sep=None, engine="python").drop(
        index=[0, 1, 2]).reset_index(drop=True)

    # create the data for calculations
    # for old buildings I use the data of the latest renovation (most likely to have heat pump installed)
    gen4 = building_class.loc[building_class["name"].str.contains("gen4", case=False)]
    sfh_gen4 = gen4.loc[gen4["name"].str.contains("sfh", case=False)]
    rh_gen4 = gen4.loc[gen4["name"].str.contains("rh", case=False)]

    gen3 = building_class.loc[building_class["name"].str.contains("gen3", case=False)]
    sfh_gen3 = gen3.loc[gen3["name"].str.contains("sfh", case=False)].iloc[-1,
               :].to_frame().transpose()  # only last one (F-series because no gen4)
    rh_gen3 = gen3.loc[gen3["name"].str.contains("rh", case=False)].iloc[-1, :].to_frame().transpose()

    gen2 = building_class.loc[building_class["name"].str.contains("gen2", case=False)]
    sfh_gen2 = gen2.loc[gen2["name"].str.contains("sfh", case=False)].iloc[-1,
               :].to_frame().transpose()  # only last one (G-series because no gen3)
    rh_gen2 = gen2.loc[gen2["name"].str.contains("rh", case=False)].iloc[-1, :].to_frame().transpose()

    modern_sfh = building_class.loc[building_class["name"].str.contains("sfh", case=False)].iloc[-2:, :]
    modern_rh = building_class.loc[building_class["name"].str.contains("rh", case=False)].iloc[-2:, :]

    all_classes = pd.concat([modern_sfh, sfh_gen2, sfh_gen3, sfh_gen4, modern_rh,
                             rh_gen2, rh_gen3, rh_gen4]).set_index("index", drop=True).sort_index()
    indices = all_classes.index.to_numpy().astype(int)

    all_dynamic_data = dynamic_data.set_index("bc_index").loc[indices, :]

    df_database = pd.concat([all_classes, all_dynamic_data], axis=1)
    # remove duplicate columns
    df_database = df_database.loc[:, ~df_database.columns.duplicated()]
    df_database = df_database.drop(columns=[
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
        "Tset",
        "user_profile",
    ])
    df_database = df_database.reset_index().rename(columns={"index": "index_invert"})
    df_database["index"] = np.arange(1, len(df_database)+1)

    for index in indices:
        air_heatpump_buildings = building_segment.loc[
            (building_segment.loc[:, "buildingclasscsvid"] == str(index)) &
            (building_segment.loc[:, "heatingsystem"].str.contains("heatpump\(air/water\)", case=False))]
        number_of_air_heatpumps = air_heatpump_buildings.loc[:, "number_of_buildings"].sum()

        water_heatpump_buildings = building_segment.loc[
            (building_segment.loc[:, "buildingclasscsvid"] == str(index)) &
            (building_segment.loc[:, "heatingsystem"].str.contains("heatpump\(water/water\)", case=False))]
        number_of_water_heatpumps = water_heatpump_buildings.loc[:, "number_of_buildings"].sum()
    print(f"number of air HP: {number_of_air_heatpumps} \n number of ground HP: {number_of_water_heatpumps}")
         # TODO noch die anzahl bestimmen! entweder alle heat pumps von einer gebäudekategorie oder nur die die ich auswähle


    return df_database

if __name__=="__main__":
    building_data = load_building_data_from_excel()
    # write to DB
    column_names = {"index": "INTEGER", "index_invert": "INTEGER", "name": "TEXT",
                    "construction_period_start": "INTEGER", "construction_period_end": "INTEGER",
                    "building_categories_index": "INTEGER", "number_of_dwellings_per_building": "REAL",
                    "number_of_persons_per_dwelling": "REAL", "horizontal_shading_building": "REAL",
                    "areawindows": "REAL", "area_suitable_solar": "REAL", "hwb_norm": "REAL",
                    "ued_dhw": "REAL", "average_effective_area_wind_west_east_red_cool": "REAL",
                    "average_effective_area_wind_south_red_cool": "REAL",
                    "average_effective_area_wind_north_red_cool": "REAL",
                    "spec_int_gains_cool_watt": "REAL",
                    "Af": "REAL", "Hop": "REAL", "Htr_w": "REAL", "Hve": "REAL", "CM_factor": "REAL",
                    "Am_factor": "REAL"}
    DB().write_DataFrame(df_database, "ID_BuildingOption",
                         column_names=column_names.keys(),
                         conn=DB().create_Connection(CONS().RootDB),
                         dtype=column_names)






