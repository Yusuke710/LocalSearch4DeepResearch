# LocalSearch4DeepResearch

> ⚠️ **Work in Progress**: This project is under active development. Features and documentation may change significantly.

A local document search system designed to integrate with OpenAI's DeepResearch, enabling semantic search across your private documents.

## Current Status

- ✅ Basic document processing (PDF, DOCX, TXT)
- ✅ Semantic search with FAISS
- ✅ API endpoints with authentication
- ✅ Markdown conversion with LLM
- ✅ Document chunking with overlap
- ✅ Test interface
- 🚧 Error handling improvements
- 📝 DeepResearch does not search the link user provides as it may consider it as jailbreaking. Fix this by modifying the prompt to type in DeepResearch.

## Project Structure

```
LocalSearch4DeepResearch/
├── web_server.py      # Web server and API endpoints
├── local_engine.py    # Document processing and search core
├── config.py          # Configuration settings
├── requirements.txt   # Python dependencies
└── data/             # Document storage
    ├── docs/         # Original documents
    └── processed/    # Processed files
        ├── markdown/ # Converted markdown
        ├── embeddings/ # Vector embeddings
        └── index/    # FAISS index
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export EMBEDDING_MODEL="sentence-transformer"  # or "openai"
export OPENAI_API_KEY="your-api-key"  # for enhanced markdown conversion (default)
```

Optional:
```bash
# To disable LLM-based markdown conversion
export USE_LLM_MARKDOWN="false"
```

3. Run the server:
```bash
# Local mode for testing search functionality. DeepResearch will not be able to access the local documents.
python web_server.py --local

# With ngrok tunnel. DeepResearch will be able to access the local documents.
python web_server.py
```

4. Go to DeepResearch and induce local search via prompt.

```bash
<Your prompt>

Search my local documents at <ngrok_url>/api/search?q=<your_query>&api_key=<your_api_key>
# e.g. Search my local documents at https://abc123.ngrok-free.app/api/search?q=machine%20learning&api_key=def456
```

## Features

### Document Processing
- Supports PDF, DOCX, and TXT files
- Automatic conversion to structured markdown using GPT-4
- Smart chunking with configurable overlap
- Embedding generation using Sentence Transformers or OpenAI
- FAISS indexing for fast similarity search

### Search Capabilities
- Semantic search using vector similarity
- Configurable number of results
- Relevance scoring
- Context-aware snippets
- Source document linking

### Security
- API key authentication
- Temporary key generation
- Rate limiting
- All data stays local

## API Endpoints

### Search Documents
```http
GET /api/search?q=<query>&api_key=<key>&k=<num_results>
```

Parameters:
- `q`: Search query (required)
- `api_key`: Authentication key (required)
- `k`: Number of results to return (optional, default: 5, max: 20)

Response:
```json
{
  "query": "machine learning",
  "top_k": 5,
  "results": [
    {
      "title": "ML_Guide.pdf (Part 1)", 
      "url": "local://docs/ML_Guide.pdf",
      "snippet": "Machine learning is a subset of artificial intelligence...",
      "content": "Full markdown content...",
      "source": "Local Documents",
      "score": 0.89
    }
  ]
}
```

### Create Temporary API Key
```http
POST /api/create-temp-key
Authorization: Bearer <master-api-key>
```

Response:
```json
{
  "key": "temporary-api-key",
  "expires_at": "2024-02-20T15:30:00Z",
  "valid_for_minutes": 30
}
```

## Configuration

Key settings in `config.py`:
- `EMBEDDING_MODEL`: Choose between "sentence-transformer" or "openai"
- `SENTENCE_TRANSFORMER_MODEL`: Default "all-MiniLM-L6-v2"
- `OPENAI_EMBEDDING_MODEL`: Default "text-embedding-ada-002"
- `EMBEDDING_DIMENSION`: Automatically set based on model
- Directory structures and paths

## Limitations

- Memory usage scales with document count
- Initial processing time for large documents
- OpenAI API costs for markdown conversion
- Free ngrok limitations (if not using local mode)

## Future Improvements

- [ ] Document versioning
- [ ] Advanced filtering options
- [ ] Custom chunking strategies
- [ ] Improved error handling
- [ ] Web management interface
- [ ] Batch processing optimization
- [ ] Multiple embedding model support
- [ ] Custom markdown templates

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - See LICENSE file for details

