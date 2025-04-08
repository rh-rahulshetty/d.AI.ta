import re
import os
import random
from typing import List
import pandas as pd
from src.data_utils.csv_field_processor import CSV_UNIQUE_COL_LIMIT
from src.modals.file_types.csv_data import CSVField
from src.modals.file_types.log_data import LogFileMetadata
from src.llm.llm_executor import LLMTaskExecutor
from src.llm.tasks.log_field_extractor import LogFieldExtractorTask
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)

TOP_K_LINES = 5  # No. of log lines to pass in the context of LogFieldExtractor


def create_log_metadata(_id: str, file_path: str) -> LogFileMetadata:
    '''Generate metadata for a Log file'''
    log_lines = open(file_path, 'r', encoding='utf-8').readlines()

    # Generate regex pattern to extract fields from log
    regex_pattern = extract_regex_pattern(file_path, log_lines)

    parsed_logs = []

    for i, line in enumerate(log_lines):
        try:
            match = re.match(regex_pattern, line)
            if match:
                parsed_logs.append(
                    match.groupdict()
                )
            else:
                raise Exception("Match not found!")
        except Exception as error:
            logger.error("Unable to parse log line id: %s (error: %s)", i, error)

    if len(parsed_logs) == 0:
        raise Exception("Unable to parse log file")

    df = pd.DataFrame(parsed_logs).infer_objects()

    logger.info("No. of log fields identified: %d", df.shape[1])

    fields = []

    for col in df.columns.tolist():
        col_series = df[col]
        uniq_list = [str(x) for x in col_series.unique().tolist()]
        fields.append(
            CSVField(
                name=str(col),
                field_type=str(col_series.dtype),
                null_count=col_series.isna().sum(),
                unique_count=len(uniq_list),
                uniques=uniq_list if len(uniq_list) <= CSV_UNIQUE_COL_LIMIT else []
            )
        )

    # Store Parsed Log into CSV
    df_path = file_path.replace(".log", "") + ".csv"
    df.to_csv(df_path, index=False)

    logger.debug("Creating Log metadata for file: '%s'", file_path)

    return LogFileMetadata(
        id=_id,
        file_path=file_path,
        row_count=len(df),
        col_count=len(fields),
        fields=fields,
        log_line_count=len(log_lines),
        csv_file_path=df_path
    )


def extract_regex_pattern(file_path: str, log_lines: List[str]) -> str:
    '''Pass logs to LLM to extract log fileds'''
    llm_executor = LLMTaskExecutor()

    regex_pattern = llm_executor.execute(LogFieldExtractorTask(
        file_name=os.path.basename(file_path),
        log_lines=random.choices(log_lines, k=TOP_K_LINES)
    )).text

    return regex_pattern
