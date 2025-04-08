from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel


SYS_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# LLM Tasks
class TasksTag(str, Enum):
    data_for_task = 'data_for_task'
    code_solver = 'code_solver'
    code_refinement = 'code_refinement'
    log_field_extractor = 'log_field_extractor'
    summarizer = 'summarizer'


# LLM Response
class ModelResponse(BaseModel):
    tags: list = []
    text: str = ""
    metadata: Any = None


class ModelResponseList(ModelResponse):
    items: List[str | Dict | Any] = []


class ModelResponseDict(ModelResponse):
    data: Dict[str, Any] = {}


class ModelBooleanResponse(ModelResponse):
    bool_value: bool
