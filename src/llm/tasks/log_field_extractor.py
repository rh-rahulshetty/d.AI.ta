import re
from typing import Any, List

from src.modals.llm_data import (
    TasksTag,
    ModelResponse
)

from src.llm.llm_task import LLMTask
from src.utils.logger import get_module_logger


logger = get_module_logger(__name__)


SYS_PROMPT = """
Given a log file with an unknown structure, analyze these sample log lines to infer the recurring components and generate a Python regex pattern.

Make sure to follow the below rules while generating the regex:
- Return only the regex pattern as a raw string with no extra explanation.
- Use named capture groups (with the syntax (?P<name>...)) to extract the key parts of each log entry.
- There should be no duplicate names in named capture groups.
- Order of the names in named capture group, must stricly follow the items in the log lines.
- The pattern should match a single log line completely from start to end.
- If there are brackets in the message, make sure to include them and escape.
- Escape double quotes as required.

### Example:

Sample Log lines:
```
Jun 14 15:16:01 authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4
```

Output:
r'^(?P<date>\w{{3}} \d{{2}} \d{{2}}:\d{{2}}:\d{{2}}) (?P<message>.*)'

"""

USER_PROMPT = """
Here are {k} log lines sampled randomly from {file_name}:

```
{log_lines}
```
"""

REGEX_PATTERN = r"```([\s\S]*)```"


def extract_field_result(result: str) -> str:
    result = result.strip()
    if result.startswith('```'):
        regex = re.findall(REGEX_PATTERN, result, re.DOTALL)[-1]
        if regex.startswith('r'):
            regex = regex[1:].strip()
        return regex.strip()
    elif result.startswith("r'") and result.endswith("'"):
        return result[2:-1]
    else:
        return result.strip()


class LogFieldExtractorTask(LLMTask):
    """
    Checks whether the given metadata file is relevant to the user query or not.
    """

    def __init__(
        self,
        file_name: str,
        log_lines: List[str],
        tags: List = [],
        metadata: Any = None
    ):
        self.file_name = file_name
        self.log_lines = log_lines
        self._tags = tags
        self.metadata = metadata

    @property
    def tags(self):
        return [TasksTag.log_field_extractor] + self._tags

    def preprocess(self):
        log_lines = "\n".join([
            line.strip() for line in self.log_lines
        ])
        return {
            "file_name": self.file_name,
            "k": len(self.log_lines),
            "log_lines": log_lines,
        }

    def prompt(self):
        return {"system": SYS_PROMPT, "user": USER_PROMPT}

    def postprocess(self, result: str) -> ModelResponse:
        tags = self.tags
        regex = extract_field_result(result)
        logger.info("Extracted regex: %s", regex)
        return ModelResponse(
            text=regex,
            tags=tags,
            metadata=self.metadata
        )
