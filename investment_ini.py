from basics.database_initializer import DatabaseInitializer
from basics.linkage import InterfaceTable
from config import config


class ProjectInvestmentInit(DatabaseInitializer):

    def load_tables(self):
        self.load_table(InterfaceTable.OperationEnergyCost.value)

    def main(self):
        self.load_tables()


if __name__ == "__main__":
    init = ProjectInvestmentInit(config=config,
                                 input_folder=config.input_investment)
    init.main()
