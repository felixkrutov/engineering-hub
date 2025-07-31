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
        self.index: Optional[faiss.Index] = None  # The FAISS index for all chunks
        self.embeddings: Optional[np.ndarray] = None # Raw embeddings for all chunks
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
        self.embeddings = None

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
        # Store raw embeddings
        self.embeddings = np.array(result['embedding']).astype('float32')
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built successfully with {len(self.chunks)} chunks from {len(self.files)} files.")

    def search(self, query: str, top_k: int = 5, file_id: Optional[str] = None) -> List[Dict]:
        logger.info(f"Performing search with query: '{query}'" + (f" within file_id: {file_id}" if file_id else ""))
        if (not self.index and not file_id) or not query or not self.chunks:
            return []

        query_embedding_result = genai.embed_content(
            model=self.embedding_model, 
            content=query, 
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')
        
        if file_id:
            # --- Filtered search within a specific file ---
            if self.embeddings is None:
                logger.error("Embeddings not available for filtered search.")
                return []
            
            # 1. Get chunks and their original indices for the target file
            target_chunks = []
            original_indices = []
            for i, chunk in enumerate(self.chunks):
                if chunk['file_id'] == file_id:
                    target_chunks.append(chunk)
                    original_indices.append(i)

            if not target_chunks:
                logger.warning(f"No chunks found for file_id: {file_id}")
                return []

            # 2. Create a temporary index with just the embeddings for these chunks
            filtered_embeddings = self.embeddings[original_indices]
            dimension = filtered_embeddings.shape[1]
            temp_index = faiss.IndexFlatL2(dimension)
            temp_index.add(filtered_embeddings)
            
            # 3. Search the temporary index
            distances, temp_indices = temp_index.search(query_embedding, min(top_k, len(target_chunks)))
            
            # 4. Map the results from the temporary index back to the original chunk objects
            results = [target_chunks[i] for i in temp_indices[0] if i < len(target_chunks)]

        else:
            # --- Global search across all files ---
            if self.index is None: return []
            distances, indices = self.index.search(query_embedding, top_k)
            results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

        logger.info(f"Search found {len(results)} relevant chunks.")
        return results

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, str]]:
        return self.files.get(file_id)

    def get_all_files(self) -> List[Dict]:
        """Returns a list of all file metadata known to the indexer."""
        return list(self.files.values()) if self.files else []
