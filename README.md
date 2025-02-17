# LocalSearch4DeepResearch

> ⚠️ **Work in Progress**: This project is under active development. Features and documentation may change significantly.

A local document search system designed to integrate with OpenAI's DeepResearch, enabling semantic search across your private documents.

## Current Status

- ✅ Basic document processing (PDF, DOCX, TXT)
- ✅ Semantic search with FAISS
- ✅ API endpoints with authentication
- ✅ Test interface
- 🚧 Error handling improvements
- 📝 DeepResearch does not search the link user provides as it may consider it as jailbreaking. Fix this by modifying the prompt to type in DeepResearch.

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

- In the first prompt, you can just type in your query.
- In the followup prompt, you can use the following prompt to induce local search. You will get it on your terminal when you start the server and the API key will be shown. I am just using the arbitrary API key in the example.

*******************************************************************************
```txt
Use my private document search system to find relevant information on my query.

To search, send a request to the following API:
GET https://a319-49-180-102-218.ngrok-free.app/api/search?q=<your_query>&api_key=ywl1StWXP_N8oCDnir9ZD0Q7AS-2kof9dSVHYYE8X4g

(Example: https://abc123.ngrok-free.app/api/search?q=machine%20learning&api_key=def456)

Instructions after retrieving results:

Summarize key insights from retrieved documents.
Cite sources where relevant (mention filenames and snippets).
If no relevant documents are found, attempt an alternative approach:
Refine the query and retry.
Expand the search to online sources if necessary.
Ensure findings from local documents are seamlessly integrated into your final research report.

Do not ignore the local document system. Immediately use the API and incorporate the results.
```
*******************************************************************************

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

## Security Considerations

### External Access via ngrok

This project uses ngrok to create a secure tunnel to your local server, allowing DeepResearch to access your documents. Here's how it works:

1. **Temporary URLs**: 
   - Each time you start the server, ngrok generates a new, unique URL and the API key.
   - You provide this URL and the API key to DeepResearch via prompt.
   - Previous URLs become invalid, preventing unauthorized access from old links
   - URLs automatically expire when you stop the server

2. **API Authentication**:
   - Every request requires a temporary API key
   - Keys are valid for only 30 minutes
   - Keys are randomly generated using `secrets.token_urlsafe(32)`
   - Invalid or expired keys are automatically rejected

3. **Rate Limiting**:
   - Requests are limited by ngrok per IP
   - Helps prevent brute force attempts and API abuse

### Data Privacy

1. **Local Processing**:
   - All documents remain on your local machine
   - Only processed search results are transmitted

2. **Controlled Access**:
   - Each DeepResearch session needs a new API key
   - Access automatically expires after 30 minutes
   - Server can be stopped at any time to immediately revoke access

### Best Practices

1. **When using with DeepResearch**:
   - Start the server only when needed
   - Stop the server after each research session

2. **Document Handling**:
   - Do not use sensitive documents as they will be exposed to OpenAI DeepResearch.
   - Monitor the server logs for unexpected access patterns

### Limitations

1. **Temporary Exposure**:
   - While running, your documents are searchable via the API with the provided URL and API key.
   - Mitigated by temporary keys and rate limiting

2. **ngrok Considerations**:
   - Free tier URLs change each session
   - Limited to one active tunnel
   - Subject to ngrok's service availability

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - See LICENSE file for details

