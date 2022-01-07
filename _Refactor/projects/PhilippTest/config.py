
from pathlib import Path

from _Refactor.basic.config import Config

project_config = Config(
    project_name="PhilippTest",
    project_root=Path(__file__).parent,
    sqlite_folder='data/sqlite',
    database_name="ProsumagerUpdated_Philipp",
    output_folder='data/output'
)
