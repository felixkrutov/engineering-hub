import logging
from typing import List, Dict, Optional
from .connector import KnowledgeBaseConnector

logger = logging.getLogger(__name__)

class KnowledgeBaseIndexer:
    def __init__(self, connector: KnowledgeBaseConnector) -> None:
        self.connector = connector
        self.index: List[Dict[str, str]] = []

    def build_index(self) -> None:
        logger.info("Starting knowledge base index build.")
        self.index = self.connector.list_files_recursive('/')
        logger.info(f"Knowledge base index build complete. Indexed {len(self.index)} files.")

    def search(self, query: str) -> List[Dict[str, str]]:
        logger.info(f"Performing search with query: '{query}'")
        if not query:
            return []
        
        lower_query = query.lower()
        results = [
            file_meta for file_meta in self.index 
            if lower_query in file_meta.get('name', '').lower()
        ]
        
        logger.info(f"Search found {len(results)} results.")
        return results

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, str]]:
        for file_meta in self.index:
            if file_meta.get('id') == file_id:
                return file_meta
        return None
