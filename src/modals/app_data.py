from typing import List, Any
from pydantic import BaseModel, ConfigDict


CODE_EXECUTION_RETRIES = 5


class AppResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_prompt: str
    generation_status: bool  # Where app result is success or not.
    message: str | None = None
    model_result: Any = None
    code: str = ""
    file_metadata_ids: List[str] = []
