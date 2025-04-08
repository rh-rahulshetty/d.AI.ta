from typing import Any, List

from src.modals.llm_data import (
    TasksTag,
    ModelBooleanResponse
)
from src.modals.file_types.base import FileMetadata, FileDataFormat

from src.llm.llm_task import LLMTask
from src.utils.logger import get_module_logger


logger = get_module_logger(__name__)


SYS_PROMPT = """Given below is a definition of a file.
Validate whether the given file has any relevant context to the user query.
You must return "Yes" if the file contains relevant context, otherwise "No".
Do NOT under any circumstances try to generate a code. Keep your response short.

Filename: {file_name}
{additional_file_info}
"""

USER_PROMPT = """### Query
{query}
"""


def fetch_additional_file_info(file_metadata: FileMetadata) -> str:
    if file_metadata.file_format == FileDataFormat.CSV:
        return "CSV Fields: " + ", ".join([
            f"{field.name} ({field.field_type})" for field in file_metadata.fields
        ])
    elif file_metadata.file_format == FileDataFormat.LOG:
        return "CSV Fields: " + ", ".join([
            f"{field.name} ({field.field_type})" for field in file_metadata.fields
        ])
    elif file_metadata.file_format == FileDataFormat.JSON:
        return "JSON Keys: " + ", ".join([
            f"{json_key.key}" for json_key in file_metadata.json_keys
        ])

    return ""


class DataForTask(LLMTask):
    """
    Checks whether the given metadata file is relevant to the user query or not.
    """

    def __init__(
        self,
        file_metadata: FileMetadata,
        query: str,
        tags: List = [],
        metadata: Any = None
    ):
        self._tags = tags
        self.file_metadata = file_metadata
        self.query = query
        self.metadata = metadata

    @property
    def tags(self):
        return [TasksTag.data_for_task] + self._tags

    def preprocess(self):
        return {
            "file_name": self.file_metadata.file_name,
            "additional_file_info": fetch_additional_file_info(self.file_metadata),
            "query": self.query
        }

    def prompt(self):
        return {"system": SYS_PROMPT, "user": USER_PROMPT}

    def postprocess(self, result: str) -> ModelBooleanResponse:
        tags = self.tags
        check_result = result.lstrip().startswith("Yes")
        return ModelBooleanResponse(
            bool_value=check_result,
            text=result,
            tags=tags,
            metadata=self.metadata
        )
