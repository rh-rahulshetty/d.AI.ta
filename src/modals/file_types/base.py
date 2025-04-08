import os
from enum import Enum
from pydantic import BaseModel, computed_field


class FileDataFormat(str, Enum):
    CSV = 'csv'
    LOG = 'log'
    ZIP = 'zip'  # Collection Data Format
    JSON = 'json'


class FileMetadata(BaseModel):
    id: str
    file_path: str
    file_format: FileDataFormat

    @computed_field
    @property
    def file_name(self) -> str:
        return os.path.basename(self.file_path)
