import time
import jq
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from typing import List
from src.modals.file_types.base import FileMetadata, FileDataFormat


def load_df_from_file(file_metadata: FileMetadata):
    file_format = file_metadata.file_format
    if file_format == FileDataFormat.CSV:
        return pd.read_csv(file_metadata.file_path).infer_objects()
    elif file_format == FileDataFormat.LOG:
        return pd.read_csv(file_metadata.csv_file_path).infer_objects()
    suggestion = ""
    if file_format == FileDataFormat.JSON:
        suggestion = "Use fetch_json to load JSON file."
    raise Exception(f"Unable to load given file({file_metadata.id}) with fetch_df. {suggestion}")


def create_local_ns(file_metadatas: List[FileMetadata]):
    '''Create default list of variables, methods and imported module.'''

    def fetch_df(file_id: str) -> pd.DataFrame:
        for metadata in file_metadatas:
            if metadata.id == file_id:
                return load_df_from_file(metadata)

    def fetch_json(file_id: str, jq_query: str) -> list:
        try:
            for metadata in file_metadatas:
                if metadata.id == file_id:
                    with open(metadata.file_path, 'r', encoding='utf-8') as fp:
                        json_data = json.load(fp)
                        compiled_query = jq.compile(jq_query)
                        result = compiled_query.input(json_data).all()
                        return result
        except ValueError as e:
            raise e

    return {
        'matplotlib': matplotlib,
        'np': np,
        'pd': pd,
        'plt': plt,
        'math': math,
        'fetch_df': fetch_df,
        'fetch_json': fetch_json,
        'time': time
    }


def executor(code: str, file_metadatas: List[FileMetadata]):
    local_ns = create_local_ns(file_metadatas)
    exec(code, local_ns, local_ns)
    if 'solver' in local_ns:
        return local_ns["solver"]()

