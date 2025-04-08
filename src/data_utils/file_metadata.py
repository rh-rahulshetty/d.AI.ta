import json
import zipfile
import tempfile
from typing import List
from zipfile import ZipFile

from src.data_utils.csv_field_processor import create_csv_metadata
from src.data_utils.log_file_processor import create_log_metadata
from src.data_utils.json_file_processor import create_json_metadata

from src.errors import UnknownFileTypeError, UnknownObjectTypeError

from src.modals.file_types.base import FileDataFormat, FileMetadata
from src.modals.file_types.csv_data import CSVFileMetadata
from src.modals.file_types.log_data import LogFileMetadata
from src.modals.file_types.json_data import JSONFileMetadata


from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)


def create_metadata_from_file(
    _id: str, file_path: str
) -> FileMetadata | List[FileMetadata]:
    """Generate File metadata"""
    logger.info("Generating file metadata: '%s'", file_path)
    file_format = file_path.split('.')[-1].lower()

    if file_format == FileDataFormat.CSV:
        return create_csv_metadata(_id, file_path)
    elif file_format == FileDataFormat.LOG:
        return create_log_metadata(_id, file_path)
    elif file_format == FileDataFormat.ZIP:
        return create_zip_metadata(_id, file_path)
    elif file_format == FileDataFormat.JSON:
        return create_json_metadata(_id, file_path)

    logger.error("Unknown file format '%s' for '%s'", file_format, file_format)
    raise UnknownFileTypeError(f"Unknown file format '{file_format}' found.")


def convert_metadata_to_json(file_metadata: FileMetadata) -> dict:
    '''Convert file metadata into json for indexing'''
    metadata_json = file_metadata.model_dump(exclude={'doc_string'})

    if file_metadata.file_format in [FileDataFormat.CSV, FileDataFormat.LOG]:
        # TODO: Chromo doesn't support list types for indexing
        # https://github.com/chroma-core/chroma/issues/3415
        metadata_json['fields'] = json.dumps(metadata_json['fields'])

    elif file_metadata.file_format == FileDataFormat.JSON:
        metadata_json['json_keys'] = json.dumps(metadata_json['json_keys'])

    return metadata_json


def get_metadata_from_json(obj: dict):
    '''Converts document from vectordb back into model'''
    try:
        file_format = obj["file_format"]
        if file_format == FileDataFormat.CSV:
            # Convert back fields into JSON dict
            obj["fields"] = json.loads(obj["fields"])
            return CSVFileMetadata(**obj)
        elif file_format == FileDataFormat.LOG:
            obj["fields"] = json.loads(obj["fields"])
            return LogFileMetadata(**obj)
        elif file_format == FileDataFormat.JSON:
            obj["json_keys"] = json.loads(obj["json_keys"])
            return JSONFileMetadata(**obj)
        raise UnknownFileTypeError()
    except Exception as error:
        logger.error("Unable to parse metadata object: %s", error)
        raise UnknownObjectTypeError() from error


def create_zip_metadata(_id: str, file_path: str) -> List[FileMetadata]:
    """Looks into ZIP file and extracts valid metadata files"""
    temp_dir = tempfile.mkdtemp(prefix="daita")
    # First extract the file into temp directory then move it into vectordb later in the callee
    file_metadatas = []
    if zipfile.is_zipfile(file_path):
        with ZipFile(file_path, "r") as zip_ref:
            for file in zip_ref.namelist():
                new_file_id = f"{_id}-{len(file_metadatas)}"
                target_path = zip_ref.extract(file, path=temp_dir)
                try:
                    if not target_path.endswith(".zip"):
                        file_metadatas.append(
                            create_metadata_from_file(new_file_id, target_path)
                        )
                except UnknownFileTypeError:
                    pass
    return file_metadatas
