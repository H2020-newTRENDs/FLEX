import logging
import os
from abc import ABC
from pathlib import Path


class Config(ABC):
    def __init__(self, project_name: str):
        self.project_name: str = project_name
        self.project_root: "Path" = self.setup_path(Path(__file__).parent.parent)
        self.input_behavior: "Path" = self.setup_path(self.project_root / f"data/input_behavior/{project_name}/")
        self.input_operation: "Path" = self.setup_path(self.project_root / f"data/input_operation/{project_name}/")
        self.input_community: "Path" = self.setup_path(self.project_root / f"data/input_community/{project_name}/")
        self.output: "Path" = self.setup_path(self.project_root / f"data/output/{project_name}/")
        self.fig: "Path" = self.setup_path(self.project_root / f"data/figure/{project_name}/")
        self.project_folder = self.project_root / "projects" / f"{project_name}"
        # self.config_logging()

    @staticmethod
    def setup_path(folder_path: Path) -> "Path":
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    # def config_logging(self):
    # get_logger(name="FLEX_LOGGER", file_name=f"{self.output / self.project_name}.log")
    # logging.getLogger("pyomo.core").setLevel(logging.ERROR)
