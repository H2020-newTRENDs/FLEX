from db_init import ProjectDatabaseInit
import pandas as pd
from flex.config import Config
from flex_operation.run import main as operation_main


def delete_result_task_folders(conf):
    for item in Path(conf.output).iterdir():
        if item.is_dir():
            print(f"Deleting directory and all contents: {item}")
            for sub_item in item.iterdir():
                # Check if the sub_item is a file
                if sub_item.is_file():
                    sub_item.unlink()  # Delete the file
            shutil.rmtree(item)

if __name__ == "__main__":

    for year in [ 2030, 2040, 2050]:
        for scen in ["H"]:#, "M"]:
            cfg = Config(f"ECEMF_T4.3_Leeuwarden_{year}_{scen}")
            df_start = pd.read_excel(cfg.input_operation / f'Scenario_start_Leeuwarden_{year}_{scen}.xlsx')
            init = ProjectDatabaseInit(
                config=cfg,
                exclude_from_permutation=["ID_Building",
                                        "ID_PV",
                                        "ID_Battery",
                                        "ID_HotWaterTank",
                                        "ID_SpaceHeatingTank",
                                        "ID_HeatingElement",
                                        "ID_Boiler",
                                        "ID_Behavior",
                                        ],
                load_scenario_table=False,
                start_scenario_table=df_start)
            init.run()

            operation_main(project_name = cfg.project_name,
                            use_multiprocessing = True,
                            save_hourly_results = True,
                            save_monthly_results = False,
                            save_yearly_results = True,
                            hourly_save_list = None,
                            operation_scenario_ids = None,
                            n_cores=20,
                            )
            
            try:
                delete_result_task_folders(cfg)
            except:
                print(f"taskfolders for {country} {year} were not deleted")
            





