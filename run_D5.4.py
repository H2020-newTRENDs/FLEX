import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import shutil
from flex_operation.run import run_operation_model, replace_scenario_runs, main
from flex.config import Config
from flex.kit import get_logger
from config import cfg








# def replace_scenarios(country: str, year: int, mode, scen_ids):
#     cfg = Config(project_name=f"D5.4_{country}_{year}")
#     get_logger(f"{cfg.project_name}", file_name=f"{cfg.output / cfg.project_name}.log")
#     replace_scenario_runs(cfg=cfg,
#                           mode=mode,
#                           operation_scenario_ids=scen_ids
#                           )


if __name__ == "__main__":
    main(project_name=cfg.project_name)

    # input_list = [(country, year, project_prefix) for year in years for country in country_list]
    number_of_physical_cores = int(multiprocessing.cpu_count() / 2) - 12
    # Parallel(n_jobs=number_of_physical_cores)(delayed(main)(*inst) for inst in input_list)



    # for year in years:
    #     for country in country_list:
    #         main(country, year, project_prefix)

    #         # replace_scenarios(country=country, year=year, mode="reference", scen_ids=None)

    # mode = "reference"
    # replace_list = [(country, year, mode, None) for year in years for country in country_list]
    # number_of_physical_cores = int(multiprocessing.cpu_count() / 2)
    # with multiprocessing.Pool(number_of_physical_cores) as pool:
    #     pool.starmap(replace_scenarios, replace_list)

