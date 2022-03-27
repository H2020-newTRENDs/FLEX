import os
from abc import ABC


class Config(ABC):

    def __init__(self, project_name):
        self.project_name: str = project_name
        self.project_root: str = self.setup_folder_path(os.path.dirname(__file__))
        self.database_folder: str = self.setup_folder_path(os.path.dirname('data/'))
        self.figure_folder: str = self.setup_folder_path(os.path.dirname('figure/'))

    @staticmethod
    def setup_folder_path(folder_path) -> str:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

