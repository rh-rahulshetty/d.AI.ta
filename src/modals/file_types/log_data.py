from src.modals.file_types.base import FileDataFormat
from src.modals.file_types.csv_data import CSVFileMetadata


class LogFileMetadata(CSVFileMetadata):
    log_line_count: int
    file_format: FileDataFormat = FileDataFormat.LOG
    csv_file_path: str
