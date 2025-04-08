import re
from typing import Any, List

from src.modals.llm_data import (
    TasksTag,
    ModelResponse
)
from src.modals.file_types.base import FileMetadata, FileDataFormat
from src.modals.file_types.csv_data import CSVFileMetadata
from src.modals.file_types.json_data import JSONKey

from src.llm.llm_task import LLMTask
from src.utils.logger import get_module_logger


logger = get_module_logger(__name__)


SYS_PROMPT = """Given below is a definition of dataframes, your task is to create a Python method 'solver' to compute the answer to the user's query.

Make sure to follow the below rules while generating the code:
- Do NOT under any circumstances call the solver method.
- Do NOT make random assumptions about the data.
- Use the appropriate data loader functions based on the data source: 
    - To access data from CSV, LOG files, you need to make use of a helper method "fetch_df" that is pre-defined.
    - To access data from JSON files, you need to make use of a helper method "fetch_json" that is pre-defined. It takes second argument called "jq_query" in which you can specify a "jq" utility query.
- Do NOT import any modules. You can make use of the imported modules defined below.
- When the user query involves drawing a plot, make sure to use matplotlib and return the Figure object from "solver" method. Also make sure to add appropriate labels and legends as required.
- Always generate minimalistic answers to the question.

Definition of fetch_df:
```
def fetch_df(file_id: str) -> pd.DataFrame:
    ...
```

Definition of fetch_json:
```
def fetch_json(file_id: str, jq_query: str) -> list:
    try:
        json_data: dict | list = load_json_data(file_id)
        compiled_query = jq.compile(jq_query)
        result = compiled_query.input(json_data).all()
        return result
    except Exception as e:
        raise e
```

Input Dataframes:
{file_infos}


Imported Modules:
```
import time
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

Output Format:
```python
def solver() -> str | int | pd.DataFrame | matplotlib.figure.Figure:
    <Code Here>
```
"""

USER_PROMPT = """### Query
{query}
"""

REGEX_PATTERN = r"```python([\s\S]*)```"


CSV_INFO_TEMPLATE = '''{idx}. File ID: {file_id}
 Filename: {file_name}
 Fields:
{fields}
'''

CSV_FIELD_INFO_TEMPLATE = '''  - "{name}" (type: {type}, unique count: {unique_count}{uniques})'''


JSON_INFO_TEMPLATE = '''{idx}. File ID: {file_id}
 Filename: {file_name}
 Keys:
{keys}
'''
JSON_KEY_INFO_TEMPLATE = '''  - "{name}" (type: {type})'''


class CodeSolver(LLMTask):
    """
    Generate a Python code to solve user query.
    """

    def __init__(
        self,
        file_metadatas: List[FileMetadata],
        query: str,
        tags: List = [],
        metadata: Any = None
    ):
        self._tags = tags
        self.file_metadatas = file_metadatas
        self.query = query
        self.metadata = metadata

    @property
    def tags(self):
        return [TasksTag.code_solver] + self._tags

    def preprocess(self):
        return {
            "file_infos": generate_file_info(self.file_metadatas),
            "query": self.query
        }

    def prompt(self):
        return {"system": SYS_PROMPT, "user": USER_PROMPT}

    def postprocess(self, result: str) -> ModelResponse:
        tags = self.tags
        code = re.findall(REGEX_PATTERN, result, re.DOTALL)[-1]
        return ModelResponse(
            text=code,
            tags=tags,
            metadata=self.metadata
        )


def fetch_fields(fields: List[CSVFileMetadata]):
    return '\n'.join([
        CSV_FIELD_INFO_TEMPLATE.format(
            name=field.name,
            type=field.field_type,
            unique_count=field.unique_count,
            uniques=" ,uniques=[{unique}]".format(
                unique=", ".join(field.uniques)
            ) if len(field.uniques) > 0 else '',
        ) for field in fields
    ])


def fetch_json_keys(keys: List[JSONKey]):
    return '\n'.join([
        JSON_KEY_INFO_TEMPLATE.format(
            name=item.key,
            type=item.type
        ) for item in keys
    ])


def fetch_file_info(idx, file_metadata: FileMetadata) -> str:
    if file_metadata.file_format == FileDataFormat.CSV:
        return CSV_INFO_TEMPLATE.format(
            idx=idx,
            file_id=file_metadata.id,
            file_name=file_metadata.file_name,
            fields=fetch_fields(file_metadata.fields)
        )
    elif file_metadata.file_format == FileDataFormat.LOG:
        return CSV_INFO_TEMPLATE.format(
            idx=idx,
            file_id=file_metadata.id,
            file_name=file_metadata.file_name,
            fields=fetch_fields(file_metadata.fields)
        )
    elif file_metadata.file_format == FileDataFormat.JSON:
        return JSON_INFO_TEMPLATE.format(
            idx=idx,
            file_id=file_metadata.id,
            file_name=file_metadata.file_name,
            keys=fetch_json_keys(file_metadata.json_keys)
        )
    return ""


def generate_file_info(file_metadatas: List[FileMetadata]) -> str:
    return "\n".join([
        fetch_file_info(idx + 1, metadata) for idx, metadata in enumerate(file_metadatas)
    ])
