
import os

from _Refactor.basic.config import Config

project_config = Config(
    project_name="SongminTest",
    project_root=os.path.dirname(__file__),
    sqlite_folder='data/sqlite',
    output_folder='data/output'
)