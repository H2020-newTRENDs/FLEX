from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import shutil
import multiprocessing
from joblib import Parallel, delayed
from flex.config import Config
from config import cfg as project_config
from flex.db import create_db_conn
from flex_operation.analyzer import OperationAnalyzer
from flex_operation.constants import OperationTable
from flex_operation.data_collector import OptDataCollector
from flex_operation.data_collector import RefDataCollector
from flex_operation.model_opt import OptInstance
from flex_operation.model_opt import OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.scenario import OperationScenario, MotherOperationScenario


def determine_number_of_scenarios(cfg: "Config"):
    scenario_df = create_db_conn(config=cfg).read_dataframe(OperationTable.Scenarios)
    number_scenarios = len(scenario_df)
    return number_scenarios


def check_if_results_exist(cfg: "Config", result_table_name: str) -> int:
    """checks if the results already exist, if yes check how many scenarios exist and if some are missing,
    return the scenario ID from where calculation should start"""
    db = create_db_conn(cfg)  # connect to database
    # check if result tables exist
    engine = db.get_engine().connect()
    if engine.dialect.has_table(engine, result_table_name):
        # find the last index where results are still available
        indices = db.read_dataframe(result_table_name, column_names=["ID_Scenario"]).to_numpy()
        start_index = np.max(indices) + 1  # return the last/highest index + 1 for the next scenario
    else:  # table does not exist
        start_index = 1
    return start_index


def find_start_index(cfg: "Config", mother_operation: "MotherOperationScenario") -> int:
    logger = logging.getLogger(f"{cfg.project_name}")
    # before running the model check if results already exist
    start_index_ref = check_if_results_exist(cfg, OperationTable.ResultRefYear)
    start_index_opt = check_if_results_exist(cfg, OperationTable.ResultOptYear)
    if start_index_opt == start_index_ref:
        start_index = start_index_opt
    elif start_index_ref > start_index_opt:  # ref scenario is further ahead by 1 scenario (any other case should not happen)
        # calculate the missing opt scenario and then return the start index of the ref scenario:
        logger.info(f"calculating opt scenario {start_index_opt} to catch up with ref scenarios.")
        opt_instance = OptInstance().create_instance()
        scenario = OperationScenario(scenario_id=start_index_opt, config=cfg, tables=mother_operation)
        opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
        if solve_status:
            OptDataCollector(opt_model, scenario.scenario_id, cfg, save_hour_results=True).run()
        start_index = start_index_ref
    else:
        logger.critical("Ref and Opt Scenario result IDs do not match and neither is ref 1 scenario ahead! \n "
                        f"last Ref scenario ID: {start_index_ref - 1} \n "
                        f"last Opt scenario ID: {start_index_opt - 1}")
        start_index = None
    return start_index


def check_for_infeasible_scenarios(cfg: "Config"):
    logger = logging.getLogger(f"{cfg.project_name}")
    infeasible_list = find_infeasible_scenarios(cfg)
    if infeasible_list:
        logger.warning(f"{cfg.project_name} has infeasible scenarios! \n"
                       f"Infeasible Scenarios: {infeasible_list}.")
    else:
        logger.info(f"All scenarios for {cfg.project_name} went through!")
    return infeasible_list


def re_run_infeasible_scenarios(cfg: "Config", scenario_number: List[int], mother_operation: MotherOperationScenario):
    """
    Run to rerun the optimization for single scenarios specificed in scenario_number list.
    """
    mother_operation = MotherOperationScenario(cfg)
    # read Infeasible scenario table
    opt_instance = OptInstance().create_instance()
    for id_operation_scenario in scenario_number:
        scenario = OperationScenario(scenario_id=id_operation_scenario, config=cfg, tables=mother_operation)
        # Reference model is not run because it is never infeasible and results are already saved!
        # run opt model
        print(f"running infeasible scenario {cfg.project_name}: {id_operation_scenario}")
        opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
        if solve_status:
            OptDataCollector(opt_model, scenario.scenario_id, cfg, save_hour_results=True).run()


def create_intermediate_folders(conf: "Config", folder_names: List[str]):
    # copy the output and input operation folders:
    for sub_project_name in folder_names:
        output_folder_source = conf.output
        output_folder_target = conf.output.parent / sub_project_name
        operation_folder_source = conf.input_operation
        operation_folder_target = conf.input_operation.parent / sub_project_name
        shutil.copytree(src=output_folder_source, dst=output_folder_target, dirs_exist_ok=True)
        # rename the copied database so it will be used:
        path_to_sqlite = output_folder_target / f"{conf.project_name}.sqlite"
        path_to_sqlite.rename(output_folder_target / f"{sub_project_name}.sqlite")
        # copy operation folder
        shutil.copytree(src=operation_folder_source, dst=operation_folder_target, dirs_exist_ok=True)


def create_lists_of_subscenarios(conf: "Config", number: int) -> List[List[int]]:
    number_of_scenarios = determine_number_of_scenarios(conf)
    scenarios_per_subfolder, rest = divmod(number_of_scenarios, number)
    equal_parts = [scenarios_per_subfolder] * number  # determine the length of each list
    # add the rest to the single list lengths
    for i in range(rest):
        equal_parts[i] += 1
    # create lists out of the lengths which contain the IDs of the scenarios:
    result = []
    start = 1
    for part in equal_parts:
        end = start + part
        result.append(list(range(start, end)))
        start = end
    return result


def split_scenario(orig_project_name: str, n_cores: int) -> List[str]:
    # scenario is split into even pieces based on the amount of multiprocessing that can be done:
    config = Config(project_name=orig_project_name)
    copy_names = [f"{config.project_name}__{i}" for i in range(n_cores)]
    create_intermediate_folders(conf=config, folder_names=copy_names)
    # now split the scenario table in each intermediate folder so each folder contains only part of the calculation
    list_of_scenario_ids = create_lists_of_subscenarios(conf=config, number=n_cores)
    # go into each subfolder and trim the scenario table
    for i, proj_name in enumerate(copy_names):
        scenario_list = list_of_scenario_ids[i]
        db = create_db_conn(config=Config(project_name=proj_name))
        scenario_df = db.read_dataframe(OperationTable.Scenarios)
        new_scenario_df = scenario_df.loc[scenario_df.loc[:, "ID_Scenario"].isin(scenario_list)]
        db.write_dataframe(
            table_name=OperationTable.Scenarios,
            data_frame=new_scenario_df,
            if_exists="replace"
        )
    return copy_names


def main(project_name: str,
         use_multiprocessing: bool = True,
         save_hourly_results: bool = True,
         save_monthly_results: bool = False,
         save_yearly_results: bool = True,
         ):
    if use_multiprocessing:
        number_of_physical_cores = int(multiprocessing.cpu_count() / 2)
        new_scenario_names = split_scenario(orig_project_name=project_name, n_cores=number_of_physical_cores)
        input_list = [
            (Config(project_name=name), save_hourly_results, save_monthly_results, save_yearly_results)
            for name in new_scenario_names
        ]
        Parallel(n_jobs=number_of_physical_cores)(delayed(run_operation_model)(*inst) for inst in input_list)
    else:
        run_operation_model(
            cfg=Config(project_name=project_name),
            save_hourly_results=save_hourly_results,
            save_monthly_results=save_monthly_results,
            save_yearly_results=save_yearly_results
        )


def run_operation_model(cfg: "Config",
                        operation_scenario_ids: List[int] = None,
                        save_hourly_results: bool = True,
                        save_monthly_results: bool = False,
                        save_yearly_results: bool = True):
    """ operation_scenario_ids: None means that the scenario IDs are taken from the scenario table an are continued
    if there are already results available. By defining the operation_scneario_ids, the model is forced to run the
    defined IDs."""
    logger = logging.getLogger(f"{cfg.project_name}")
    # define mother operation before looping through scenarios:
    mother_operation = MotherOperationScenario(cfg)
    # define the scenario IDs that still have to be calculated
    if operation_scenario_ids is None:
        # check if the start index for opt and ref model are different (if calculation gets interrupted during optimization)
        start_index = find_start_index(cfg, mother_operation)
        logger.info(f"starting calculating scenario from ID {start_index}")
        total_scenarios = determine_number_of_scenarios(cfg)
        operation_scenario_ids = [id_scenario for id_scenario in
                                  range(start_index, total_scenarios + 1)]
        # check if total number of scenarios is already reached:
        if start_index > total_scenarios:
            infeasible_list = check_for_infeasible_scenarios(cfg)
            # rerun the infeasible scenarios
            re_run_infeasible_scenarios(cfg, infeasible_list, mother_operation)
            logger.info(f"re-calculated scenarios {infeasible_list}")
            new_infeasible_list = check_for_infeasible_scenarios(cfg)
            if len(new_infeasible_list) > 0:
                logger.warning(f"infeasible scenarios still exist! scenarios: {new_infeasible_list}")
                print(f"infeasible scenarios still exist! scenarios: {new_infeasible_list}")

            return

    opt_instance = OptInstance().create_instance()
    for id_operation_scenario in tqdm(operation_scenario_ids, desc=f"{cfg.project_name}"):
        logger.info(f"FlexOperation Model --> Scenario = {id_operation_scenario}.")
        scenario = OperationScenario(scenario_id=id_operation_scenario, config=cfg, tables=mother_operation)
        # run ref model
        ref_model = RefOperationModel(scenario).solve()
        RefDataCollector(model=ref_model,
                         scenario_id=scenario.scenario_id,
                         config=cfg,
                         save_hour_results=save_hourly_results,
                         save_month_results=save_monthly_results,
                         save_year_results=save_yearly_results).run()
        # run opt model

        opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
        if solve_status:
            OptDataCollector(model=opt_model,
                             scenario_id=scenario.scenario_id,
                             config=cfg,
                             save_hour_results=True,
                             save_month_results=save_monthly_results,
                             save_year_results=save_yearly_results).run()

    # check if there were any infeasible scenarios:
    check_for_infeasible_scenarios(cfg)


def run_and_replace_reference_scenario(scen: OperationScenario,
                                       db,
                                       cfg):
    # run ref model
    ref_model = RefOperationModel(scen).solve()
    # delete the results from the sqlite database, the hourly results will be overwritten by default
    db.delete_row_from_table(
        table_name=OperationTable.ResultRefYear,
        column_name_plus_value={
            "ID_Scenario": scen.scenario_id
        }
    )
    RefDataCollector(ref_model, scen.scenario_id, cfg, save_hour_results=True).run()


def run_and_replace_optimization_scenario(scen: OperationScenario,
                                          db,
                                          cfg,
                                          opt_instance):
    # run opt model
    opt_model, solve_status = OptOperationModel(scen).solve(opt_instance)
    # delete the results from the sqlite database, the hourly results will be overwritten by default
    db.delete_row_from_table(
        table_name=OperationTable.ResultOptYear,
        column_name_plus_value={
            "ID_Scenario": scen.scenario_id
        }
    )
    if solve_status:
        OptDataCollector(opt_model, scen.scenario_id, cfg, save_hour_results=True).run()


def replace_scenario_runs(cfg: "Config", operation_scenario_ids: List[int] = None, mode: str = "both"):
    """
    If it turns out that a certain scenario did not run correctly either in the optimization or the reference
    mode, running this function will recalculate this single scenario and replace the results of this single
    scenario with existing ones.
    operation_scenario_ids: list of ids which scenarios should be replaced, if None all scenarios are replaced
    mode: both (optimization and reference runs are replaced), optimization, reference

    """
    logger = logging.getLogger(f"{cfg.project_name}")
    db = create_db_conn(config=cfg)
    # define mother operation before looping through scenarios:
    mother_operation = MotherOperationScenario(cfg)
    if mode == "optimization" or "both":
        opt_instance = OptInstance().create_instance()

    if operation_scenario_ids == None:
        operation_scenario_ids = list(np.arange(1, determine_number_of_scenarios(cfg) + 1))
    for id_operation_scenario in tqdm(operation_scenario_ids, desc=f"{cfg.project_name}"):
        scenario = OperationScenario(scenario_id=id_operation_scenario, config=cfg, tables=mother_operation)
        if mode == "reference":
            logger.info(f"Rerunning Scenario {id_operation_scenario} for reference.")
            run_and_replace_reference_scenario(scenario, db, cfg)
        elif mode == "optimization":
            logger.info(f"Rerunning Scenario {id_operation_scenario} for optimization.")
            run_and_replace_optimization_scenario(scenario, db, cfg, opt_instance)
        elif mode == "both":
            logger.info(f"Rerunning Scenario {id_operation_scenario} for reference and optimization.")
            run_and_replace_reference_scenario(scenario, db, cfg)
            run_and_replace_optimization_scenario(scenario, db, cfg, opt_instance)
        else:
            ValueError("mode must be either 'both', 'optimization' or 'reference'")


def run_operation_analyzer(
        component_changes: List[Tuple[str, int, int]],
        components: List[Tuple[str, int]],
        config: "Config"
):
    ana = OperationAnalyzer(config)
    ana.create_operation_energy_cost_table()
    ana.plot_operation_energy_cost_curve()
    ana.create_component_energy_cost_change_tables(component_changes)
    ana.plot_operation_energy_cost_change_curve(component_changes)
    ana.plot_scenario_electricity_balance(scenario_id=1)
    # # component interaction
    component_change = ("ID_PV", 2, 1)
    other_components = list(filter(lambda item: item[0] != component_change[0], components))
    ana.plot_component_interaction_full(component_change, other_components)
    ana.plot_component_interaction_specific(component_change, ("ID_Battery", 2))
    ana.plot_component_interaction_heatmap(component_change, ("ID_Boiler", 2), ("ID_SpaceHeatingTank", 2), components)


def debug_operation_result(config: "Config"):
    ana = OperationAnalyzer(config)
    ana.compare_opt_ref(scenario_id=1)


def find_infeasible_scenarios(config: "Config") -> list:
    db = create_db_conn(config)
    opt = db.read_dataframe(OperationTable.ResultOptYear)
    ref = db.read_dataframe(OperationTable.ResultRefYear)
    opt_scenarios = list(opt["ID_Scenario"].unique())
    ref_scenarios = list(ref["ID_Scenario"].unique())
    infeasible_scenarios = set(ref_scenarios) - set(opt_scenarios)

    return list(infeasible_scenarios)


if __name__ == "__main__":
    main(project_name=project_config.project_name,
         use_multiprocessing=True,
         save_hourly_results=True,
         save_monthly_results=False,
         save_yearly_results=True,
         )

