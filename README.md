# LocalSearch4DeepResearch

> ⚠️ **Work in Progress**: This project is under active development. Features and documentation may change significantly.

A local document search system designed to integrate with OpenAI's DeepResearch, enabling semantic search across your private documents.

## Current Status

- ✅ Basic document processing (PDF, DOCX, TXT)
- ✅ Semantic search implementation
- ✅ API endpoints with authentication
- ✅ Test document generation
- 🚧 Documentation
- 🚧 Error handling
- 🚧 Web interface
- 📝 More features planned

## Overview

This project creates a local search server that:
1. Processes local documents (PDF, DOCX, TXT)
2. Generates semantic embeddings
3. Provides a search API accessible to DeepResearch
4. Maintains document privacy while enabling AI-powered search


## API Endpoints

- `GET /api/search`
  - Query local documents
  - Parameters:
    - `q`: Search query
    - `api_key`: Authentication key

- `GET /api/source-info`
  - Get information about indexed documents

- `GET /api/preview/<file_path>`
  - Get document preview

## Technical Details

### Document Processing
- Documents are converted to markdown using GPT-4
- Text is split into overlapping chunks
- Each chunk is embedded using sentence transformers or OpenAI
- Embeddings are indexed using FAISS for fast similarity search

### Search Implementation
- Uses vector similarity search
- Supports semantic understanding
- Returns relevant document chunks
- Includes relevance scores

### Security Considerations
- All data stays local
- API key required for access
- Rate limiting prevents abuse
- Temporary keys expire automatically

## Limitations

- Free ngrok tier limitations:
  - URL changes on restart
  - Limited connections per minute
  - Single tunnel session

- Local processing requirements:
  - Memory usage scales with document count
  - Initial processing time for large documents
  - OpenAI API costs for markdown conversion

## Future Improvements

- [ ] Document versioning
- [ ] Advanced filtering options
- [ ] Custom chunking strategies
- [ ] Better error handling
- [ ] Web interface for management
- [ ] Batch processing optimization

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - See LICENSE file for details

