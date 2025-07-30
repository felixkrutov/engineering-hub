import logging
import io
from typing import List, Dict, Optional

import yadisk
from yadisk.exceptions import UnauthorizedError, NotFoundError

from .connector import KnowledgeBaseConnector

logger = logging.getLogger(__name__)


class YandexDiskConnector(KnowledgeBaseConnector):
    def __init__(self, token: str):
        if not token:
            raise ValueError("Yandex.Disk API token is required.")
        self.token = token
        self.client: yadisk.YaDisk = yadisk.YaDisk(token=self.token)
        try:
            logger.info("Verifying Yandex.Disk API token...")
            self.client.get_disk_info()
            logger.info("YandexDiskConnector initialized and token verified successfully.")
        except UnauthorizedError:
            logger.critical("Yandex.Disk API token is invalid or has expired.")
            raise ValueError("Invalid Yandex.Disk API token.")

    def _scan_path_recursive(self, path: str) -> List[Dict[str, str]]:
        files_metadata: List[Dict[str, str]] = []
        try:
            logger.info(f"Scanning Yandex.Disk path: {path}")
            resource = self.client.get_meta(path, limit=1000)
            items = resource._embedded.items
            for item in items:
                item_type = item.type
                item_path = item.path
                if not item_path:
                    continue
                if item_type == 'dir':
                    files_metadata.extend(self._scan_path_recursive(item_path))
                elif item_type == 'file':
                    file_meta = {
                        "id": item_path,
                        "name": item.name,
                        "path": item_path,
                        "mime_type": item.mime_type,
                    }
                    files_metadata.append(file_meta)
        except NotFoundError:
            logger.warning(f"Path not found on Yandex.Disk: {path}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while scanning Yandex.Disk path {path}: {e}", exc_info=True)
        return files_metadata

    def list_files_recursive(self, path: str) -> List[Dict[str, str]]:
        return self._scan_path_recursive(path)

    def get_file_content(self, file_id: str) -> Optional[bytes]:
        logger.info(f"Requesting content for file from Yandex.Disk: {file_id}")
        try:
            buffer = io.BytesIO()
            self.client.download(file_id, buffer)
            content = buffer.getvalue()
            logger.info(f"Successfully downloaded {len(content)} bytes for file: {file_id}")
            return content
        except NotFoundError:
            logger.warning(f"File not found on Yandex.Disk: {file_id}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading file {file_id} from Yandex.Disk: {e}", exc_info=True)
            return None
