# Same contents as document_engine.py, just renamed file 

from pathlib import Path
import hashlib
import pdfplumber
import docx
from typing import Optional, Dict, List, Tuple
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import pickle
import json
import faiss
from datetime import datetime
from config import *

class LocalEngine:
    def __init__(self):
        self.document_metadata = self.load_or_create_metadata()
        self.document_titles, self.search_index = self.initialize_search_index()
        
    def initialize_search_index(self) -> Tuple[List[str], faiss.Index]:
        """Initialize the search index and return document titles and index."""
        model = self.get_embedding_model()
        titles, embeddings = self.load_documents(model)
        
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        if len(embeddings) > 0:
            try:
                index.add(embeddings)
            except AssertionError:
                # Handle dimension mismatch by reprocessing documents
                self.document_metadata = {}
                self.save_metadata()
                
                # Clear existing processed files
                for dir_path in [MARKDOWN_DIR, EMBEDDINGS_DIR]:
                    for file in dir_path.glob("*"):
                        file.unlink()
                
                titles, embeddings = self.load_documents(model)
                index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                if len(embeddings) > 0:
                    index.add(embeddings)
        
        index_path = INDEX_DIR / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        
        return titles, index

    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search documents using the query string."""
        # Ensure k is within reasonable bounds
        k = max(1, min(k, 20))  # Limit k between 1 and 20
        
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.search_index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.document_titles):
                result = self._format_search_result(idx, distances[0][i])
                if result:
                    results.append(result)
        
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def load_documents(self, model=None) -> tuple[List[str], np.ndarray]:
        """Load and process all documents, returning titles and embeddings."""
        document_titles = []
        all_embeddings = []
        
        # Process each document in the docs directory
        for file_path in DOCS_DIR.iterdir():
            if file_path.is_file():
                if self.process_document(file_path, model):
                    # Load processed chunks
                    doc_data = self.document_metadata[str(file_path)]
                    
                    # Skip if embedding model has changed
                    if doc_data.get("embedding_model") != EMBEDDING_MODEL.value:
                        if self.process_document(file_path, model):  # Reprocess with new model
                            doc_data = self.document_metadata[str(file_path)]
                        else:
                            continue
                    
                    for chunk in doc_data["chunks"]:
                        # Load embedding
                        embedding_path = BASE_DIR / chunk["embedding_file"]
                        with open(embedding_path, 'rb') as f:
                            embedding = pickle.load(f)
                        
                        # Load markdown for title
                        markdown_path = BASE_DIR / chunk["markdown_file"]
                        with open(markdown_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                        
                        # Add to collections
                        document_titles.append(f"{file_path.name} (Part {chunk['chunk_index'] + 1})")
                        all_embeddings.append(embedding[0])
        
        return document_titles, np.array(all_embeddings)

    def process_document(self, file_path: Path, model=None) -> bool:
        """Process a single document: convert to markdown, generate embeddings, and update metadata."""
        try:
            file_hash = self.get_file_hash(file_path)
            
            # Check if file has been processed and hasn't changed
            if str(file_path) in self.document_metadata:
                if self.document_metadata[str(file_path)]["hash"] == file_hash:
                    return True
            
            # Extract and convert text
            text = self.extract_text(file_path)
            if not text:
                return False
            
            if USE_LLM_MARKDOWN and openai.api_key:
                markdown_text = self.convert_to_markdown_with_llm(text, file_path.name)
                if not markdown_text:
                    return False
            else:
                # Simple markdown conversion without LLM
                markdown_text = self.convert_to_simple_markdown(text, file_path.name)
            
            # Split into chunks and process each chunk
            chunks = self.chunk_markdown(markdown_text)
            chunk_data = []
            
            # Generate embeddings for all chunks at once
            embeddings = self.generate_batch_embeddings(chunks, model)
            if embeddings is None:
                return False
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Save markdown chunk
                markdown_file = self.save_markdown(chunk, file_path, i)
                
                # Save embedding
                embedding_file = self.save_embeddings(np.array([embedding]), markdown_file)
                
                chunk_data.append({
                    "chunk_index": i,
                    "markdown_file": str(markdown_file.relative_to(BASE_DIR)),
                    "embedding_file": str(embedding_file.relative_to(BASE_DIR)),
                    "embedding_model": EMBEDDING_MODEL.value
                })
            
            # Update metadata
            self.document_metadata[str(file_path)] = {
                "hash": file_hash,
                "last_processed": datetime.now().isoformat(),
                "chunks": chunk_data,
                "embedding_model": EMBEDDING_MODEL.value
            }
            
            self.save_metadata()
            return True
        
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return False

    # Document Processing Methods
    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Extract text from various file types."""
        text = ""
        try:
            if file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file_path.suffix.lower() == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    text_parts = [page.extract_text(x_tolerance=3, y_tolerance=3) or "" 
                                for page in pdf.pages]
                    text = "\n\n".join(text_parts)
                    text = text.replace('\x00', '')
                    text = ' '.join(text.split())
            elif file_path.suffix.lower() == ".docx":
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        
        return text.strip()

    def convert_to_markdown_with_llm(self, text: str, file_name: str) -> Optional[str]:
        """Convert document text to markdown format using GPT."""
        try:
            # Limit input text to approximately 4000 tokens
            max_chars = 16000  # Approximate 4000 tokens
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            system_prompt = """
            Convert the following document into well-structured markdown format. Follow these rules:
            1. Create a clear hierarchy with headers (# for main title, ## for sections, ### for subsections)
            2. Preserve important formatting (lists, tables, code blocks if present)
            3. Add section breaks where appropriate
            4. Ensure paragraphs are well-separated
            5. Include a brief summary at the top
            6. Add metadata section at the start with filename and type
            """
            
            user_prompt = f"Please convert this document content to markdown. Filename: {file_name}\n\nContent:\n{text}"
            
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error converting to markdown: {str(e)}")
            return None

    def convert_to_simple_markdown(self, text: str, file_name: str) -> str:
        """Convert document text to simple markdown format without using LLM."""
        lines = text.split('\n')
        markdown_lines = [
            f"# {file_name}",
            "",
            "## Content",
            "",
        ]
        
        # Add content with basic formatting
        current_paragraph = []
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            elif current_paragraph:
                markdown_lines.append(' '.join(current_paragraph))
                markdown_lines.append('')
                current_paragraph = []
        
        if current_paragraph:
            markdown_lines.append(' '.join(current_paragraph))
        
        return '\n'.join(markdown_lines)

    def chunk_markdown(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split markdown text into chunks while preserving markdown structure."""
        # Ensure chunk size is within OpenAI's token limits (roughly 4 chars per token)
        max_tokens = 2000  # Safe limit for ada-002
        chunk_size = min(chunk_size, max_tokens * 4)
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            
            # If a single line is too long, split it
            if line_length > chunk_size:
                words = line.split()
                current_line = []
                current_line_length = 0
                
                for word in words:
                    if current_line_length + len(word) + 1 > chunk_size:
                        chunks.append(' '.join(current_line))
                        current_line = [word]
                        current_line_length = len(word)
                    else:
                        current_line.append(word)
                        current_line_length += len(word) + 1
                
                if current_line:
                    line = ' '.join(current_line)
                    line_length = len(line)
            
            if current_length + line_length > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                if overlap > 0:
                    # Limit overlap size
                    overlap = min(overlap, chunk_size // 4)
                    overlap_size = 0
                    overlap_lines = []
                    for prev_line in reversed(current_chunk):
                        if overlap_size + len(prev_line) > overlap:
                            break
                        overlap_lines.insert(0, prev_line)
                        overlap_size += len(prev_line) + 1
                    
                    current_chunk = overlap_lines
                    current_length = overlap_size
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(line)
            current_length += line_length + 1
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Add safety check for chunk sizes
        chunks = [chunk[:max_tokens * 4] for chunk in chunks]
        
        return chunks

    # Embedding Methods
    @staticmethod
    @lru_cache(maxsize=1)
    def get_embedding_model():
        """Get the configured embedding model."""
        if EMBEDDING_MODEL == EmbeddingModel.SENTENCE_TRANSFORMER:
            return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        return None

    def generate_embedding(self, text: str, model=None) -> Optional[np.ndarray]:
        """Generate embeddings using the configured model."""
        try:
            if EMBEDDING_MODEL == EmbeddingModel.OPENAI:
                response = openai.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            else:
                if model is None:
                    model = self.get_embedding_model()
                return model.encode([text], convert_to_numpy=True)[0]
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None

    def generate_batch_embeddings(self, texts: List[str], model=None) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            if EMBEDDING_MODEL == EmbeddingModel.OPENAI:
                # Process smaller batches with token limit check
                max_batch_size = 10  # Smaller batch size for safety
                all_embeddings = []
                
                for i in range(0, len(texts), max_batch_size):
                    batch = texts[i:i + max_batch_size]
                    # Truncate each text to stay within token limits
                    batch = [text[:8000] for text in batch]  # ~2000 tokens per text
                    response = openai.embeddings.create(
                        model=OPENAI_EMBEDDING_MODEL,
                        input=batch
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                
                return np.array(all_embeddings, dtype=np.float32)
            else:
                if model is None:
                    model = self.get_embedding_model()
                return model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            return None

    def save_markdown(self, content: str, original_file: Path, chunk_index: int = None) -> Path:
        """Save markdown content to file."""
        file_stem = original_file.stem
        if chunk_index is not None:
            markdown_file = MARKDOWN_DIR / f"{file_stem}_chunk_{chunk_index}.md"
        else:
            markdown_file = MARKDOWN_DIR / f"{file_stem}.md"
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return markdown_file

    def save_embeddings(self, embeddings: np.ndarray, file_path: Path) -> Path:
        """Save embeddings to file."""
        embedding_file = EMBEDDINGS_DIR / f"{file_path.stem}.pkl"
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
        return embedding_file

    # Metadata Management Methods
    def load_or_create_metadata(self) -> dict:
        """Load existing metadata or create new metadata file."""
        metadata_file = BASE_DIR / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_metadata(self):
        """Save metadata to file."""
        metadata_file = BASE_DIR / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_metadata, f, indent=2)

    # Helper Methods
    def _format_search_result(self, idx: int, distance: float) -> Optional[dict]:
        """Format search result for API response."""
        doc_title = self.document_titles[idx]
        file_name = doc_title.split(" (Part")[0]
        chunk_index = int(doc_title.split("Part ")[-1].rstrip(")")) - 1
        
        for doc_path, doc_data in self.document_metadata.items():
            if Path(doc_path).name == file_name:
                markdown_path = BASE_DIR / doc_data["chunks"][chunk_index]["markdown_file"]
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return {
                    "title": doc_title,
                    "url": f"local://{doc_path}",
                    "snippet": content[:300] + "...",
                    "content": content,
                    "source": "Local Documents",
                    "score": float(1 / (1 + distance if distance != 0 else 1e-6))
                }
        return None 