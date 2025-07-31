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
        # INSTRUCTION 1: Add a new property to __init__
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
        
        # INSTRUCTION 2: Store embeddings in build_index
        embeddings_array = np.array(result['embedding']).astype('float32')
        self.embeddings = embeddings_array
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built successfully with {len(self.chunks)} chunks from {len(self.files)} files.")

    # INSTRUCTION 3: Completely rewrite the search method
    def search(self, query: str, top_k: int = 5, file_id: Optional[str] = None) -> List[Dict]:
        logger.info(f"Performing semantic search for '{query}'" + (f" within file {file_id}" if file_id else ""))
        if self.index is None or self.embeddings is None or not query:
            return []

        # Generate embedding for the query
        query_embedding_result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array([query_embedding_result['embedding']]).astype('float32')

        if file_id:
            # --- FOCUSED SEARCH LOGIC ---
            # 1. Find indices of chunks belonging to the target file
            target_chunk_indices = [i for i, chunk in enumerate(self.chunks) if chunk.get('file_id') == file_id]
            
            if not target_chunk_indices:
                logger.warning(f"No chunks found for file_id: {file_id}")
                return []

            # 2. Extract the embeddings for these specific chunks
            target_embeddings = self.embeddings[target_chunk_indices]

            # 3. Build a temporary, in-memory FAISS index for just these chunks
            dimension = target_embeddings.shape[1]
            temp_index = faiss.IndexFlatL2(dimension)
            temp_index.add(target_embeddings)

            # 4. Search in the temporary index
            distances, local_indices = temp_index.search(query_embedding, k=min(top_k, len(target_chunk_indices)))

            # 5. Map local indices back to the original chunk indices
            original_indices = [target_chunk_indices[i] for i in local_indices[0]]
            
            results = [self.chunks[i] for i in original_indices]

        else:
            # --- GLOBAL SEARCH LOGIC ---
            distances, original_indices = self.index.search(query_embedding, k=top_k)
            results = [self.chunks[i] for i in original_indices[0] if i < len(self.chunks)]

        logger.info(f"Search found {len(results)} relevant chunks.")
        return results

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, str]]:
        return self.files.get(file_id)

    def get_all_files(self) -> List[Dict]:
        """Returns a list of all file metadata known to the indexer."""
        return list(self.files.values()) if self.files else []
