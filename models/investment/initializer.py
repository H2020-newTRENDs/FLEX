from basics.database_initializer import DatabaseInitializer


class InvestmentDatabaseInitializer(DatabaseInitializer):

    def get_input_folder(self):
        return self.config.input_investment

