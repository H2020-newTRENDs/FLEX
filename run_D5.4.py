import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import shutil
from flex_operation.run import run_operation_model, replace_scenario_runs, main
from flex.config import Config
from flex.kit import get_logger
from config import cfg



if __name__ == "__main__":
    main(project_name=cfg.project_name, use_multiprocessing=True)


