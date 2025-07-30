import logging
import io
import yadisk
from typing import List, Dict, Optional

from .connector import KnowledgeBaseConnector

logger = logging.getLogger(__name__)

class YandexDiskConnector(KnowledgeBaseConnector):
    def __init__(self, token: str):
        if not token:
            raise ValueError("Yandex.Disk API token is required.")
        self.token = token
        self.client = yadisk.YaDisk(token=self.token)
        try:
            self.client.get_disk_info()
            logger.info("YandexDiskConnector initialized and token is valid.")
        except yadisk.exceptions.AuthError:
            logger.critical("Invalid Yandex.Disk API token.")
            raise ValueError("Invalid Yandex.Disk API token.")

    def list_files_recursive(self, path: str) -> List[Dict[str, str]]:
        files_list = []
        try:
            items_iterator = self.client.get_files(path=path, limit=200)
            for item in items_iterator:
                if item.type == 'file':
                    files_list.append({
                        "id": item.path,
                        "name": item.name,
                        "path": item.path,
                        "mime_type": item.get('mime_type', 'application/octet-stream')
                    })
        except yadisk.exceptions.PathNotFoundError:
            logger.warning(f"Path not found on Yandex.Disk: {path}")
        except Exception as e:
            logger.error(f"Error listing files on Yandex.Disk path {path}: {e}", exc_info=True)
        return files_list

    def get_file_content(self, file_id: str) -> Optional[bytes]:
        try:
            buffer = io.BytesIO()
            self.client.download(file_id, buffer)
            buffer.seek(0)
            return buffer.getvalue()
        except yadisk.exceptions.PathNotFoundError:
            logger.warning(f"File not found on Yandex.Disk: {file_id}")
            return None
        except Exception as e:
            logger.error(f"Error downloading file {file_id} from Yandex.Disk: {e}", exc_info=True)
            return None
