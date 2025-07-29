import abc
import logging
import magic
from pathlib import Path
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)

class KnowledgeBaseConnector(abc.ABC):
    @abc.abstractmethod
    def list_files_recursive(self) -> List[Dict[str, str]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_file_content(self, file_id: str) -> Optional[bytes]:
        raise NotImplementedError

class MockConnector(KnowledgeBaseConnector):
    def __init__(self) -> None:
        self.base_path = Path("./mock_disk")
        self.base_path.mkdir(exist_ok=True)
        logger.info(f"MockConnector initialized with base path: {self.base_path.resolve()}")

    def list_files_recursive(self) -> List[Dict[str, str]]:
        logger.info(f"Scanning for files in {self.base_path.resolve()}")
        files_metadata: List[Dict[str, str]] = []
        for item in self.base_path.rglob('*'):
            if item.is_file():
                try:
                    mime_type = magic.from_file(str(item), mime=True)
                    file_meta = {
                        "id": str(item.relative_to(self.base_path)),
                        "name": item.name,
                        "path": str(item.resolve()),
                        "mime_type": mime_type,
                    }
                    files_metadata.append(file_meta)
                except Exception as e:
                    logger.error(f"Could not process file {item}: {e}")
        return files_metadata

    def get_file_content(self, file_id: str) -> Optional[bytes]:
        logger.info(f"Requesting content for file_id: {file_id}")
        
        try:
            file_path = (self.base_path / file_id).resolve()
            
            if not file_path.is_relative_to(self.base_path.resolve()):
                logger.error(f"Path traversal attempt blocked for file_id: {file_id}")
                raise ValueError("Access to the requested file path is not allowed.")

            if file_path.is_file():
                logger.info(f"Successfully retrieved content for file: {file_path}")
                return file_path.read_bytes()
            else:
                logger.warning(f"File not found at path: {file_path}")
                return None
        except Exception as e:
            logger.error(f"An error occurred while getting content for file_id {file_id}: {e}")
            return None
