from enum import StrEnum
import os
from typing import List
from pydantic import BaseModel, computed_field
from src.modals.file_types.base import FileDataFormat, FileMetadata


MAX_KEY_COUNT_FOR_DOCSTRING = 50


def split_last_key(key):
    return key.split(".")[-1]


class JSONType(StrEnum):
    array = 'list'
    number = 'number'
    string = 'str'
    json = 'dict'
    boolean = 'bool'


class JSONKey(BaseModel):
    key: str
    type: JSONType
    
    def __hash__(self) -> int:
        return hash(self.key)


class JSONFileMetadata(FileMetadata):
    '''Metadata for CSV file'''
    file_format: FileDataFormat = FileDataFormat.JSON
    json_keys: List[JSONKey]

    @computed_field
    @property
    def doc_string(self) -> str:
        filename = os.path.basename(self.file_path)
        # Filter keys
        unflatten_json_keys = [
            split_last_key(json_key.key)
            for json_key in self.json_keys
            # Need to filter json/array roots as they could be stored in the json_keys
            if json_key.type not in [JSONType.json, JSONType.array] 
        ]
        fields_list_string = ", ".join(
           list(set(unflatten_json_keys))
        )
        return f'Metadata for a JSON file "{filename}" that consists of the following keys: {fields_list_string}'
