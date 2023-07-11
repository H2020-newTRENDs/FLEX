import logging
import os
from abc import ABC
from pathlib import Path


class Config(ABC):
    def __init__(self, project_name: str):
        self.project_name: str = project_name
        self.project_root: "Path" = self.setup_path(__file__)
        self.input_behavior: "Path" = self.setup_path(f"data/input_behavior/{project_name}/")
        self.input_operation: "Path" = self.setup_path(f"data/input_operation/{project_name}/")
        self.input_community: "Path" = self.setup_path(f"data/input_community/{project_name}/")
        self.output: "Path" = self.setup_path(f"data/output/{project_name}/")
        self.fig: "Path" = self.setup_path(f"data/figure/{project_name}/")
        self.config_logging()

    @staticmethod
    def setup_path(folder_path: str) -> "Path":
        folder_path = os.path.dirname(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return Path(folder_path)

    @staticmethod
    def config_logging():
        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
