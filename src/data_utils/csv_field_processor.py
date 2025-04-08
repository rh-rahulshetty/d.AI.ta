import pandas as pd

from src.modals.file_types.csv_data import (
    CSVField,
    CSVFileMetadata
)
from src.utils.logger import get_module_logger

CSV_UNIQUE_COL_LIMIT = 10  # max no. of unique values to be considered to index

logger = get_module_logger(__name__)


def create_csv_metadata(_id: str, file_path: str) -> CSVFileMetadata:
    '''Generate metadata for a CSV file'''
    df = pd.read_csv(file_path).infer_objects()
    fields = []

    for col in df.columns.tolist():
        col_series = df[col]
        uniq_list = [str(x) for x in col_series.unique().tolist()]
        fields.append(
            CSVField(
                name=str(col),
                field_type=str(col_series.dtype),
                null_count=col_series.isna().sum(),
                unique_count=len(uniq_list),
                uniques=uniq_list if len(uniq_list) <= CSV_UNIQUE_COL_LIMIT else []
            )
        )
    logger.debug("Creating CSV metadata for file: '%s'", file_path)

    return CSVFileMetadata(
        id=_id,
        file_path=file_path,
        row_count=len(df),
        col_count=len(fields),
        fields=fields
    )
