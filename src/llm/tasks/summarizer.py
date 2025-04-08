from typing import Any, List

from src.modals.llm_data import (
    TasksTag,
    ModelResponse
)

from src.llm.llm_task import LLMTask
from src.utils.logger import get_module_logger


logger = get_module_logger(__name__)

SYS_PROMPT = """Given a data for some type that is converted into string format. Your job is to generate a summary paragraph for the data.

Make sure to follow the below rules while generating the code:
- Summary shouldn't be more than 3 lines.
- Do NOT under any circumstances generate a code.
- Do NOT return anything else, directly start with the summary.
- Do NOT make assumption about the data. Otherwise assume its just raw value.
- Don't randomly refer to values in percentage. Just paraphrase it using raw value. E.g: If data is "0.99" and query is "What's the max cpu usage?" then you need to need something like "The max cpu usage observed is 0.99".

"""

USER_PROMPT = """### Query
{user_query}

### Data
{data}
"""


class Summarizer(LLMTask):
    """
    Summarizes the given data.
    """

    def __init__(
        self,
        data: str,
        user_query: str,
        tags: List = [],
        metadata: Any = None
    ):
        self._tags = tags
        self.data = data
        self.user_query = user_query
        self.metadata = metadata

    @property
    def tags(self):
        return [TasksTag.data_for_task] + self._tags

    def preprocess(self):
        return {
            "data": self.data,
            "user_query": self.user_query
        }

    def prompt(self):
        return {"system": SYS_PROMPT, "user": USER_PROMPT}

    def postprocess(self, result: str) -> ModelResponse:
        return ModelResponse(
            text=result,
            tags=self.tags,
            metadata=self.metadata
        )
