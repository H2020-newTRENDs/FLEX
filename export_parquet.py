import pyarrow as pa
import pyarrow.parquet as pq
from config import cfg
from flex.config import Config
from flex.db import create_db_conn
from flex_operation.constants import OperationTable
import pandas as pd


class ParquetExporter:

    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)

    def get_parquet_path(self, table_name: str):
        return self.config.output.joinpath(self.config.project_name + "_" + table_name).with_suffix('.parquet')

    def export_table(self, table_name: str):
        print(f'reading table {table_name} from sqlite file...')
        table = self.db.read_dataframe(table_name)
        print(f'exporting table {table_name} to parquet file...')
        pq.write_table(pa.Table.from_pandas(df=table), self.get_parquet_path(table_name))

    def read_table(self, table_name: str):
        print(f'reading table {table_name} from parquet file...')
        df = pd.read_parquet(self.get_parquet_path(table_name))
        print(df.head())


if __name__ == "__main__":
    pe = ParquetExporter(cfg)
    # pe.export_table(OperationTable.ResultOptHour)
    # pe.read_table(OperationTable.ResultOptHour)
    # pe.export_table(OperationTable.ResultRefHour)

