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
            "AUT",  
            "BEL", 
            # "POL",
            # "CYP", 
            # "PRT",
            # "DNK",
            # "FRA",
            # "CZE",  
            # "DEU", 
            # "HRV",
            # "HUN",
            # "ITA",  
            # "LUX",
            # "NLD",
            # "SVK",
            # "SVN",
            # 'IRL',
            # 'ESP',  
            # 'SWE',
            # 'GRC',
            # 'LVA',  
            # 'LTU',
            # 'MLT',
            # 'ROU',
            # 'BGR',  
            # 'FIN',
            # 'EST',
        ]

    years = [2020, 2030, 2040, 2050]#2030, 2040]
    for country in country_list:
        for year in years:
            cfg = get_config(f"{country}_{year}")

            run_flex_operation_model(cfg, task_number=10)

    # cfg = get_config("AUT_2030")
    # run_flex_operation_model(cfg, task_number=1)

          
