import os
from pathlib import Path
from abc import ABC


class Config(ABC):

    def __init__(self, db_name: str,
                 flex_behavior_config: dict = None,
                 flex_operation_config: dict = None,
                 flex_investment_config: dict = None):
        self.db_name: str = db_name
        self.project_root: 'Path' = self.setup_folder_path(os.path.dirname(__file__))
        self.source_data_folder: 'Path' = Path(self.project_root).parent/Path('data')
        self.flex_behavior_config = flex_behavior_config
        self.flex_operation_config = flex_operation_config
        self.flex_investment_config = flex_investment_config

    @property
    def db_folder(self) -> 'Path':
        folder = self.setup_folder_path(os.path.dirname('data/'))
        return folder

    @property
    def fig_folder(self) -> 'Path':
        folder = self.setup_folder_path(os.path.dirname('figure/'))
        return folder

    @staticmethod
    def setup_folder_path(folder_path) -> 'Path':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return Path(folder_path)


if __name__ == "__main__":
    config = Config('test')
    print(config.source_data_folder)
