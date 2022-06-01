from basics.database_initializer import DatabaseInitializer


class OperationDatabaseInitializer(DatabaseInitializer):

    def get_input_folder(self):
        return self.config.input_operation

