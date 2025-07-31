import logging
from typing import List, Dict, Optional

import faiss
import numpy as np
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .connector import KnowledgeBaseConnector
from .parser import parse_document

logger = logging.getLogger(__name__)

class KnowledgeBaseIndexer:
    def __init__(self, connector: KnowledgeBaseConnector) -> None:
        self.connector = connector
        self.files: Dict[str, Dict] = {}  # Stores file metadata, keyed by file_id
        self.index: Optional[faiss.Index] = None  # The FAISS index
        self.chunks: List[Dict] = []  # Stores chunk text and metadata
        self.embedding_model = 'models/embedding-001'
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def build_index(self) -> None:
        logger.info("Starting production knowledge base index build.")
        all_files = self.connector.list_files_recursive('/')
        
        # Reset state
        self.files = {file['id']: file for file in all_files}
        self.chunks = []
        self.index = None

        for file_id, file_meta in self.files.items():
            try:
                content = self.connector.get_file_content(file_id)
                if not content:
                    continue
                
                parsed_text = parse_document(file_meta['name'], content, file_meta.get('mime_type'))
                if not parsed_text or not parsed_text.strip():
                    continue

                split_chunks = self.text_splitter.split_text(parsed_text)
                for chunk_text in split_chunks:
                    self.chunks.append({
                        'text': chunk_text, 
                        'file_id': file_id, 
                        'file_name': file_meta['name']
                    })
            except Exception as e:
                logger.error(f"Failed to process file {file_meta.get('name', file_id)}: {e}")

        if not self.chunks:
            logger.warning("No chunks were created from the documents. Index is empty.")
            return

        logger.info(f"Generated {len(self.chunks)} chunks. Now creating embeddings...")
        chunk_texts = [c['text'] for c in self.chunks]
        result = genai.embed_content(
            model=self.embedding_model, 
            content=chunk_texts, 
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = np.array(result['embedding'])
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index built successfully with {len(self.chunks)} chunks from {len(self.files)} files.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        logger.info(f"Performing semantic search with query: '{query}'")
        if not self.index or not query:
            return []

        query_embedding_result = genai.embed_content(
            model=self.embedding_model, 
            content=query, 
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)
        
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        logger.info(f"Search found {len(results)} relevant chunks.")
        return results

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, str]]:
        # This is now a fast O(1) lookup instead of a slow O(n) loop
        return self.files.get(file_id)
