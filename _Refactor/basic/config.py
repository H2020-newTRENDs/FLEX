from pathlib import Path
import sqlalchemy


def create_connection(db_folder: Path = Path(__file__).parent / Path("data"),
                      db_name: str = "root"
                      ) -> sqlalchemy.engine.Engine:
    """if no database name and folder path are provided the root database is selected"""
    return sqlalchemy.create_engine(f'sqlite:///{db_folder / Path(db_name + ".sqlite")}',
                                    pool_pre_ping=True)
    # return sqlalchemy.create_engine(f'sqlite:///{os.path.join(db_folder, db_name + ".sqlite")}')


root_connection = create_connection()  # when multiprocessing the engine has to be created for each process


class Config:
    def __init__(self,
                 project_name: str,
                 project_root: str,
                 sqlite_folder: str,
                 database_name: str,
                 output_folder: str
                 ):
        self.root_db_folder = Path(__file__).parent / Path("data")
        self.database_name = database_name
        self.project_name = project_name
        self.project_root = project_root
        self.sqlite_folder = sqlite_folder
        self.output_folder = output_folder
