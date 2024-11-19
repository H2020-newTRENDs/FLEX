from projects.main import get_config
from projects.main import run_flex_operation_model
from projects.main import generate_summary, process_summary
import shutil
from pathlib import Path


def delete_result_files(conf):
    # Iterate through each item in the directory
    for item in Path(conf.output).iterdir():
        # Check if the item is a file
        if item.is_file():
            # Check the file extension and delete if it's not .csv or .parquet
            if item.suffix in ['.gzip']:
                # print(f"Deleting: {item.name}")
                item.unlink()  # Delete the file


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
    country_list = [
            # "AUT",  #songmin 2040
            # "BEL", #songmin 2040
            # "POL",
            # "CYP",  # songmin 2040
            # "PRT",
            # "DNK",
            # "FRA",
            # "CZE",  #songmin2040
            # "DEU",  #songmin 2040
            # "HRV",
            # "HUN",
            "ITA",  # 2050 fehlt
            "LUX",
            "NLD",
            "SVK",
            "SVN",
            'IRL',
            # 'ESP',  # 2050 fehlt
            # 'SWE',
            # 'GRC',
            # 'LVA',  # 2050 fehlt
            # 'LTU',
            # 'MLT',
            # 'ROU',
            # 'BGR',  # songmin 2040
            # 'FIN',
            # 'EST',
        ]

    years = [2040]#2030, 2040]
    deletion_went_wrong = []
    for country in country_list:
        for year in years:
            # if country == "AUT" and year == 2020:
            #     cfg = get_config(f"{country}_{year}")
            #     generate_summary.gen_summary_year(cfg)
            #     # Iterate through each item in the directory
            #     for item in Path(cfg.output).iterdir():
            #         # Check if the item is a file
            #         if item.is_file():
            #             # Check the file extension and delete if it's not .csv or .parquet
            #             if item.suffix not in ['.csv', '.sqlite']:
            #                 print(f"Deleting: {item.name}")
            #                 item.unlink()  # Delete the file
            #         elif item.is_dir():
            #             print(f"Deleting directory and all contents: {item}")
            #             shutil.rmtree(item)
            #     continue
            cfg = get_config(f"{country}_{year}")

    # run model and plotter
            run_flex_operation_model(cfg, task_number=20)
    # run_flex_operation_plotter(cfg)

    # generate summary
            generate_summary.gen_summary_year(cfg)
            generate_summary.gen_summary_hour(cfg)
            # process_summary.compare_hwb_diff()
            try:
                delete_result_files(cfg)
            except:
                deletion_went_wrong.append(cfg)
                print(f"{country} {year} deletion of files did not work. Delete manually!")
            try:
                delete_result_task_folders(cfg)
            except:
                print(f"taskfolders for {country} {year} were not deleted")

    print(deletion_went_wrong)

    # process summary
    # process_summary.concat_summary()
    # process_summary.process_summary_year()
    # process_summary.process_summary_hour()
