from models.operation.analyzer import Analyzer
from config import config


class ProjectAnalyzer(Analyzer):
    pass


if __name__ == "__main__":
    analyzer = ProjectAnalyzer(config)
    # analyzer.compare_opt_ref(1)
    analyzer.compare_opt(id1=1, id2=16)
