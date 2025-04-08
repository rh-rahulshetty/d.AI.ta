import re
from typing import Any, List

from src.modals.llm_data import (
    TasksTag,
    ModelResponse
)
from src.modals.file_types.base import FileMetadata

from src.llm.tasks.code_solver import generate_file_info, REGEX_PATTERN
from src.llm.llm_task import LLMTask
from src.utils.logger import get_module_logger


logger = get_module_logger(__name__)


SYS_PROMPT = """Given below is a definition of dataframes and a Python Code, your task is to create a refine the method 'solver' based on user's query.

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

USER_PROMPT = """### Code
```python
{code}
```

### Query
{query}
"""


class CodeRefinement(LLMTask):
    """
    Refine the given code based on user feedback.
    """

    def __init__(
        self,
        code: str,
        file_metadatas: List[FileMetadata],
        query: str,
        tags: List = [],
        metadata: Any = None
    ):
        self._tags = tags
        self.code = code
        self.file_metadatas = file_metadatas
        self.query = query
        self.metadata = metadata

    @property
    def tags(self):
        return [TasksTag.code_refinement] + self._tags

    def preprocess(self):
        return {
            "code": self.code,
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
