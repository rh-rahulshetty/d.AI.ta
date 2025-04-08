import os
from typing import List, Any
from pydantic import BaseModel, computed_field
from src.modals.file_types.base import FileDataFormat, FileMetadata


class CSVField(BaseModel):
    name: str
    field_type: str
    null_count: int = 0
    unique_count: int = 0
    uniques: List[Any] = []


class CSVFileMetadata(FileMetadata):
    '''Metadata for CSV file'''
    file_format: FileDataFormat = FileDataFormat.CSV
    fields: List[CSVField] = []
    row_count: int = 0
    col_count: int = 0

    @computed_field
    @property
    def doc_string(self) -> str:
        filename = os.path.basename(self.file_path)
        fields_list_string = ", ".join(
            [csv_field.name for csv_field in self.fields]
        )
        return f'Metadata for a CSV file "{filename}" that consists of the following fields: {fields_list_string}'
