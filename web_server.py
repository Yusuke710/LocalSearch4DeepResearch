from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pyngrok import ngrok
import argparse
import logging
from datetime import datetime, timedelta
import secrets
from functools import wraps
from local_engine import LocalEngine
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
local_engine = LocalEngine()

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
            }
        },
        "indexed_documents": len(local_engine.document_metadata),
        "total_chunks": local_engine.search_index.ntotal
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
    try:
        k = int(request.args.get("k", "5"))  # Default to 5 if not specified
    except ValueError:
        k = 5  # Use default if invalid value provided
    
    logger.info(f"Search request received - Query: {query}, IP: {request.remote_addr}")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'."}), 400

    results = local_engine.search(query, k=k)
    return jsonify({
        "query": query,
        "top_k": k,
        "results": results
    })

@app.route("/api/create-temp-key", methods=["POST"])
@require_api_key
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
    parser = argparse.ArgumentParser(description='Run the local search server')
    parser.add_argument('--local', action='store_true', help='Run in local mode without ngrok')
    args = parser.parse_args()
    
    try:
        # Create initial temporary key
        temp_key = TemporaryAPIKey()
        temp_api_keys.append(temp_key)

        if args.local:
            print(f"\nRunning in local mode")
            print(f"Web Interface: http://localhost:{PORT}/test")
            print(f"API Endpoint: http://localhost:{PORT}/api/search")
            print(f"Temporary API key (valid for 30 minutes): {temp_key.key}")
            app.run(host="0.0.0.0", port=PORT, debug=True)
        else:
            public_url = ngrok.connect(PORT).public_url
            print(f"\nLocal documents are accessible at: {public_url}/api/search")
            print(f"Temporary API key (valid for 30 minutes): {temp_key.key}")
            print("\nUse this in your DeepResearch prompt:")
            print(f"Search my local documents at {public_url}/api/search?q=<your_query>&api_key={temp_key.key}")
            app.run(host="0.0.0.0", port=PORT, debug=False)
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        if not args.local:
            ngrok.kill() 