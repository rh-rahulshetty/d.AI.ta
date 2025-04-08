import os
import json
import tempfile
from shutil import rmtree
from typing import List

import chromadb

from src.data_utils.file_metadata import (
    create_metadata_from_file,
    get_metadata_from_json,
    convert_metadata_to_json,
)
from src.utils.logger import get_module_logger
from src.modals.file_types.base import FileMetadata

VECTORDB_COLLECTION_NAME = "data-collection"
TEMP_DIR_PREFIX = "daita"
TOP_N_RESULTS = 5

logger = get_module_logger(__name__)


class VectorDBSession:
    """
    Used to manage data files uploaded as part of VectorDB Session.
    """

    def __init__(self):
        # DONT REMOVE!
        # https://github.com/langchain-ai/langchain/issues/26884#issuecomment-2376276522
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        self.temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)

        self.chroma_client = chromadb.EphemeralClient()  # In-memory storage

        # Generate unique collection per user session
        random_id = os.path.basename(self.temp_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=(VECTORDB_COLLECTION_NAME + f"-{random_id}").rstrip('_-')
        )
        logger.info("Initialized Logger")

    def __del__(self):
        rmtree(self.temp_dir)

    def __len__(self):
        return self.collection.count()

    def add_file(self, file_name: str, file_bytes: bytes):
        """
        The method persists the file into a temp directory location, while
        creating the metadata in VectorDB.
        """
        # Save file into temp location
        file_path = os.path.join(self.temp_dir, file_name)
        file_id = f"file-{self.collection.count()}"

        with open(file_path, "wb") as file_writer:
            file_writer.write(file_bytes)

        # Save metadata
        file_metadata_s = create_metadata_from_file(_id=file_id, file_path=file_path)
        if isinstance(file_metadata_s, list):
            for file_metadata in file_metadata_s:
                self.add_file_metadata_to_db(file_metadata)
        else:
            self.add_file_metadata_to_db(file_metadata_s)

    def add_file_metadata_to_db(self, file_metadata: FileMetadata):
        """Adds file_metadata info into VectorDB"""
        metadata_json = convert_metadata_to_json(file_metadata)

        logger.debug("%s", json.dumps(metadata_json, indent=2))

        self.collection.add(
            ids=file_metadata.id,
            documents=file_metadata.doc_string,
            metadatas=metadata_json,
        )
        logger.info("Created new %s document", file_metadata.file_format)

    def query(self, query_text, top_n=TOP_N_RESULTS) -> List[FileMetadata]:
        """Query for similar data items in vector db"""
        results = self.collection.query(
            query_texts=query_text,  # Chroma will embed this for you
            n_results=top_n,  # how many results to return
        )
        # Convert ChromaDB Query Result into FileMetadata
        file_metadata_lists: List[FileMetadata] = []

        # 0'th index because only 1 text query requested at a time
        total_results = len(results["documents"][0])

        for i in range(total_results):
            try:
                metadata = results["metadatas"][0][i]
                file_metadata = get_metadata_from_json(metadata)
                file_metadata_lists.append(file_metadata)
            except Exception as error:
                logger.error("Something happened while parsing metadata: %s", error)

        return file_metadata_lists

    def fetch_metadata_by_ids(self, ids: List[str]) -> List[FileMetadata]:
        """Return file metadata by ids"""
        results = self.collection.get(ids=ids)
        file_metadata_lists: List[FileMetadata] = []

        # Directly returns a list containing document items unlike in query().
        total_results = len(results["documents"])

        for i in range(total_results):
            try:
                metadata = results["metadatas"][i]
                file_metadata = get_metadata_from_json(metadata)
                file_metadata_lists.append(file_metadata)
            except Exception as error:
                logger.error("Something happened while parsing metadata: %s", error)
        return file_metadata_lists
