from flask import Flask, request, jsonify
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import docx
import openai
from typing import Optional, Dict, List
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from functools import lru_cache, wraps
from pyngrok import ngrok
import secrets
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

app = Flask(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory structure
BASE_DIR = Path("data")
DOCS_DIR = BASE_DIR / "docs"
MARKDOWN_DIR = BASE_DIR / "processed" / "markdown"
EMBEDDINGS_DIR = BASE_DIR / "processed" / "embeddings"
INDEX_DIR = BASE_DIR / "processed" / "index"

# Create directories if they don't exist
for dir_path in [DOCS_DIR, MARKDOWN_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize lists to store document data
document_metadata: Dict[str, dict] = {}

class EmbeddingModel(Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI = "openai"

# Configuration
EMBEDDING_MODEL = EmbeddingModel(os.getenv("EMBEDDING_MODEL", "sentence-transformer"))
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = 1536 if EMBEDDING_MODEL == EmbeddingModel.OPENAI else 384  # ada-002 = 1536, MiniLM = 384

# Generate a secure API key
API_KEY = os.getenv("LOCAL_API_KEY", secrets.token_urlsafe(32))
print(f"Your API Key: {API_KEY}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporaryAPIKey:
    def __init__(self, expiry_minutes=30):
        self.key = secrets.token_urlsafe(32)
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(minutes=expiry_minutes)
    
    def is_valid(self):
        return datetime.now() < self.expires_at

# Store temporary API keys
temp_api_keys = []

def cleanup_expired_keys():
    """Remove expired API keys"""
    global temp_api_keys
    temp_api_keys = [key for key in temp_api_keys if key.is_valid()]

def get_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_markdown(content: str, original_file: Path, chunk_index: int = None) -> Path:
    """Save markdown content to file."""
    file_stem = original_file.stem
    if chunk_index is not None:
        markdown_file = MARKDOWN_DIR / f"{file_stem}_chunk_{chunk_index}.md"
    else:
        markdown_file = MARKDOWN_DIR / f"{file_stem}.md"
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(content)
    return markdown_file

def save_embeddings(embeddings: np.ndarray, file_path: Path):
    """Save embeddings to file."""
    embedding_file = EMBEDDINGS_DIR / f"{file_path.stem}.pkl"
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings, f)
    return embedding_file

def load_or_create_metadata() -> Dict[str, dict]:
    """Load existing metadata or create new metadata file."""
    metadata_file = BASE_DIR / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata():
    """Save metadata to file."""
    metadata_file = BASE_DIR / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(document_metadata, f, indent=2)

def convert_to_markdown_with_llm(text: str, file_name: str) -> Optional[str]:
    """Convert document text to markdown format using GPT."""
    try:
        system_prompt = """
        Convert the following document into well-structured markdown format. Follow these rules:
        1. Create a clear hierarchy with headers (# for main title, ## for sections, ### for subsections)
        2. Preserve important formatting (lists, tables, code blocks if present)
        3. Add section breaks where appropriate
        4. Ensure paragraphs are well-separated
        5. Include a brief summary at the top
        6. Add metadata section at the start with filename and type
        """
        
        user_prompt = f"Please convert this document content to markdown. Filename: {file_name}\n\nContent:\n{text[:4000]}"
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
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

def extract_text(file_path: Path) -> str:
    """Extract text from various file types."""
    text = ""
    try:
        if file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_path.suffix.lower() == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        text_parts.append(page_text)
                text = "\n\n".join(text_parts)
                text = text.replace('\x00', '')
                text = ' '.join(text.split())
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    
    return text.strip()

@lru_cache(maxsize=1)
def get_embedding_model():
    """Get the configured embedding model."""
    if EMBEDDING_MODEL == EmbeddingModel.SENTENCE_TRANSFORMER:
        return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return None  # OpenAI doesn't need a local model

def generate_embedding(text: str, model=None) -> np.ndarray:
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
                model = get_embedding_model()
            return model.encode([text], convert_to_numpy=True)[0]
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def generate_batch_embeddings(texts: List[str], model=None) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
    try:
        if EMBEDDING_MODEL == EmbeddingModel.OPENAI:
            response = openai.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            return np.array([data.embedding for data in response.data], dtype=np.float32)
        else:
            if model is None:
                model = get_embedding_model()
            return model.encode(texts, convert_to_numpy=True)
    except Exception as e:
        print(f"Error generating batch embeddings: {str(e)}")
        return None

def chunk_markdown(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split markdown text into chunks while preserving markdown structure.
    
    Args:
        text: The markdown text to split
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of markdown text chunks
    """
    # Split text into lines to preserve markdown structure
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line)
        
        # Start a new chunk if adding this line would exceed chunk_size
        if current_length + line_length > chunk_size and current_chunk:
            # Join the current chunk and add to chunks
            chunks.append('\n'.join(current_chunk))
            
            # Keep last few lines for overlap
            if overlap > 0:
                # Find lines to keep for overlap
                overlap_size = 0
                overlap_lines = []
                for prev_line in reversed(current_chunk):
                    if overlap_size + len(prev_line) > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_size += len(prev_line) + 1  # +1 for newline
                
                # Start new chunk with overlap
                current_chunk = overlap_lines
                current_length = overlap_size
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(line)
        current_length += line_length + 1  # +1 for newline
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def process_document(file_path: Path, model=None) -> bool:
    """Process a single document: convert to markdown, generate embeddings, and update metadata."""
    try:
        file_hash = get_file_hash(file_path)
        
        # Check if file has been processed and hasn't changed
        if str(file_path) in document_metadata:
            if document_metadata[str(file_path)]["hash"] == file_hash:
                return True
        
        # Extract and convert text
        text = extract_text(file_path)
        if not text:
            return False
        
        markdown_text = convert_to_markdown_with_llm(text, file_path.name)
        if not markdown_text:
            return False
        
        # Split into chunks and process each chunk
        chunks = chunk_markdown(markdown_text)
        chunk_data = []
        
        # Generate embeddings for all chunks at once
        embeddings = generate_batch_embeddings(chunks, model)
        if embeddings is None:
            return False
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Save markdown chunk
            markdown_file = save_markdown(chunk, file_path, i)
            
            # Save embedding
            embedding_file = save_embeddings(np.array([embedding]), markdown_file)
            
            chunk_data.append({
                "chunk_index": i,
                "markdown_file": str(markdown_file.relative_to(BASE_DIR)),
                "embedding_file": str(embedding_file.relative_to(BASE_DIR)),
                "embedding_model": EMBEDDING_MODEL.value
            })
        
        # Update metadata
        document_metadata[str(file_path)] = {
            "hash": file_hash,
            "last_processed": datetime.now().isoformat(),
            "chunks": chunk_data,
            "embedding_model": EMBEDDING_MODEL.value
        }
        
        save_metadata()
        return True
    
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
        return False

def load_documents(model=None) -> tuple[List[str], np.ndarray]:
    """Load and process all documents, returning titles and embeddings."""
    document_titles = []
    all_embeddings = []
    
    # Load existing metadata
    global document_metadata
    document_metadata = load_or_create_metadata()
    
    # Process each document in the docs directory
    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file():
            if process_document(file_path, model):
                # Load processed chunks
                doc_data = document_metadata[str(file_path)]
                
                # Skip if embedding model has changed
                if doc_data.get("embedding_model") != EMBEDDING_MODEL.value:
                    if process_document(file_path, model):  # Reprocess with new model
                        doc_data = document_metadata[str(file_path)]
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

def initialize_search_index() -> tuple[List[str], faiss.Index]:
    """Initialize the search index and return titles and index."""
    model = get_embedding_model()
    
    # Load or process documents
    titles, embeddings = load_documents(model)
    
    # Create and save FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    if len(embeddings) > 0:
        index.add(embeddings)
    
    # Save index
    index_path = INDEX_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    
    return titles, index

# Initialize search components
document_titles, search_index = initialize_search_index()
print(f"Indexed {search_index.ntotal} document chunks.")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-Api-Key') or request.args.get('api_key')
        if not api_key:
            return jsonify({"error": "Missing API key"}), 401
            
        # Check master API key
        if api_key == API_KEY:
            return f(*args, **kwargs)
            
        # Check temporary API keys
        cleanup_expired_keys()
        if any(temp_key.key == api_key and temp_key.is_valid() for temp_key in temp_api_keys):
            return f(*args, **kwargs)
            
        return jsonify({"error": "Invalid API key"}), 401
    return decorated_function

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

@app.route("/")
def index():
    return jsonify({
        "status": "running",
        "endpoints": {
            "search": {
                "url": "/api/search",
                "method": "GET",
                "params": {
                    "q": "search query (required)",
                    "api_key": "your API key (required)"
                }
            },
            "source_info": {
                "url": "/api/source-info",
                "method": "GET"
            },
            "preview": {
                "url": "/api/preview/<file_path>",
                "method": "GET"
            }
        },
        "indexed_documents": len(document_metadata),
        "total_chunks": search_index.ntotal
    })

@app.route("/test")
def test_page():
    return """
    <html>
        <head><title>Local Search Test</title></head>
        <body>
            <h1>Local Document Search</h1>
            <form id="searchForm">
                <input type="text" id="query" placeholder="Enter search query">
                <input type="text" id="apiKey" placeholder="Enter API key">
                <button type="submit">Search</button>
            </form>
            <div id="results"></div>
            
            <script>
                document.getElementById('searchForm').onsubmit = async (e) => {
                    e.preventDefault();
                    const query = document.getElementById('query').value;
                    const apiKey = document.getElementById('apiKey').value;
                    
                    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&api_key=${apiKey}`);
                    const data = await response.json();
                    
                    const results = document.getElementById('results');
                    results.innerHTML = '<h2>Results:</h2>' + 
                        data.results.map(r => `
                            <div style="margin-bottom: 20px;">
                                <h3>${r.title}</h3>
                                <p>${r.snippet}</p>
                                <small>Score: ${r.score}</small>
                            </div>
                        `).join('');
                };
            </script>
        </body>
    </html>
    """

@app.route("/api/search", methods=["GET"])
@require_api_key
@limiter.limit("10 per minute")
def search():
    query = request.args.get("q", "")
    logger.info(f"Search request received - Query: {query}, IP: {request.remote_addr}")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'."}), 400

    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return jsonify({"error": "Failed to generate query embedding"}), 500
    
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search for the top 5 most similar documents
    k = 5
    distances, indices = search_index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(document_titles):
            doc_title = document_titles[idx]
            file_name = doc_title.split(" (Part")[0]
            chunk_index = int(doc_title.split("Part ")[-1].rstrip(")")) - 1
            
            for doc_path, doc_data in document_metadata.items():
                if Path(doc_path).name == file_name:
                    markdown_path = BASE_DIR / doc_data["chunks"][chunk_index]["markdown_file"]
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Format result to match web search results
                    result = {
                        "title": doc_title,
                        "url": f"local://{doc_path}",  # Use custom protocol to identify local files
                        "snippet": content[:300] + "...",  # Preview snippet
                        "content": content,
                        "source": "Local Documents",
                        "score": float(1 / (1 + distances[0][i] if distances[0][i] != 0 else 1e-6))
                    }
                    results.append(result)
                    break
    
    return jsonify({
        "query": query,
        "results": sorted(results, key=lambda x: x["score"], reverse=True)
    })

# Add metadata endpoint for DeepResearch to understand the source
@app.route("/api/source-info", methods=["GET"])
def source_info():
    return jsonify({
        "name": "Local Documents",
        "description": "Private document collection stored locally",
        "document_count": len(document_metadata),
        "file_types": list(set(Path(p).suffix for p in document_metadata.keys())),
        "last_updated": max(doc["last_processed"] for doc in document_metadata.values())
    })

# Add document preview endpoint
@app.route("/api/preview/<path:file_path>")
def get_document_preview(file_path):
    try:
        full_path = Path(file_path)
        if str(full_path) not in document_metadata:
            return jsonify({"error": "Document not found"}), 404
            
        doc_data = document_metadata[str(full_path)]
        preview_data = {
            "title": full_path.name,
            "chunks": [],
            "metadata": {
                "file_type": full_path.suffix.lstrip('.'),
                "last_processed": doc_data["last_processed"],
                "chunk_count": len(doc_data["chunks"])
            }
        }
        
        # Get preview of each chunk
        for chunk in doc_data["chunks"]:
            markdown_path = BASE_DIR / chunk["markdown_file"]
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
                preview_data["chunks"].append({
                    "index": chunk["chunk_index"],
                    "preview": content[:200] + "..."
                })
        
        return jsonify(preview_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/create-temp-key", methods=["POST"])
@require_api_key  # Still require master API key to create temporary keys
def create_temporary_key():
    cleanup_expired_keys()
    temp_key = TemporaryAPIKey()
    temp_api_keys.append(temp_key)
    
    return jsonify({
        "key": temp_key.key,
        "expires_at": temp_key.expires_at.isoformat(),
        "valid_for_minutes": 30
    })

if __name__ == "__main__":
    # Use a different port (e.g., 5001)
    PORT = 5001
    
    try:
        # Start ngrok tunnel with the new port
        public_url = ngrok.connect(PORT).public_url
        
        # Create initial temporary key
        temp_key = TemporaryAPIKey()
        temp_api_keys.append(temp_key)
        
        print(f"\nLocal documents are accessible at: {public_url}/api/search")
        print(f"Temporary API key (valid for 30 minutes): {temp_key.key}")
        print("\nUse this in your DeepResearch prompt:")
        print(f"Search my local documents at {public_url}/api/search?api_key={temp_key.key}")
        
        # Run the Flask app with the new port, debug mode disabled
        app.run(host="0.0.0.0", port=PORT, debug=False)
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        # Kill any existing ngrok processes
        ngrok.kill()
