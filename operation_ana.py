from models.operation.analyzer import OperationAnalyzer
from config import config


class ProjectOperationAnalyzer(OperationAnalyzer):
    pass


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    # ana.compare_opt_ref(1)
    # ana.compare_opt(id1=1, id2=16)
    # ana.compare_ref(id1=1, id2=16)
    ana.create_inv_table()

