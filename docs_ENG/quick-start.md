# Quick Start Guide 🚀

Get the ShopGuard backend service up and running in minutes.

## Prerequisites

Before starting, ensure you have:

- **Python 3.8+** (Python 3.9+ recommended)
- **vivo AI Platform Account** with valid APP_ID and APP_KEY
- **Git** for cloning the repository

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/shopguard-backend.git
cd shopguard-backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Environment Configuration

Create your environment configuration:

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` file with your credentials:

```properties
# vivo AI Platform Configuration
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here

# API Endpoints (use defaults)
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# Web Search (optional)
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## Step 3: Verify Knowledge Base

Check if the knowledge base files exist:

```bash
# Check for knowledge base files
ls -la knowledge_base_embeddings/

# You should see:
# all_knowledge_embeddings.json
```

If the files are missing, contact the project maintainer to obtain them.

## Step 4: Start the Service

### Development Mode

```bash
# Start with Python directly
python newserver.py
```

### Production Mode (Recommended)

```bash
# Start with uvicorn
uvicorn newserver:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 5: Verify Installation

### Health Check

```bash
curl http://localhost:8000/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": 1703025600,
  "rag_available": true,
  "active_sessions": 0,
  "system_info": {
    "rag_initialized": true,
    "knowledge_base_size": 10297
  }
}
```

### Check Available Models

```bash
curl http://localhost:8000/v1/models
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "vivo-BlueLM-TB-Pro",
      "object": "model",
      "created": 1703025600,
      "owned_by": "vivo"
    },
    {
      "id": "vivo-BlueLM-V-2.0",
      "object": "model", 
      "created": 1703025600,
      "owned_by": "vivo"
    }
  ]
}
```

## Step 6: Test Basic Functionality

### Text-Only Query

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [
      {
        "role": "user",
        "content": "有人要我转账1000元买iPhone，说是特价，这靠谱吗？"
      }
    ]
  }'
```

### Test with Python

Create `test_api.py`:

```python
import requests

def test_basic_query():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "vivo-BlueLM-TB-Pro",
        "messages": [
            {
                "role": "user",
                "content": "有人说可以帮我代购iPhone便宜50%，这是诈骗吗？"
            }
        ],
        "enable_rag": True,
        "user_type": "普通用户"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("✅ API Response Success!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"❌ API Test Failed: {e}")

if __name__ == "__main__":
    test_basic_query()
```

Run the test:

```bash
python test_api.py
```

## Configuration Options

### Essential Settings

| Variable | Description | Required |
|----------|-------------|----------|
| `VIVO_APP_ID` | Your vivo AI platform app ID | ✅ |
| `VIVO_APP_KEY` | Your vivo AI platform app key | ✅ |
| `WEB_SEARCH_API_KEY` | Web search service API key | ❌ |

### Optional Performance Settings

```properties
# Performance Configuration
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_SECONDS=30
RAG_CACHE_TTL_SECONDS=3600
CONVERSATION_HISTORY_LIMIT=100

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
```

## Docker Quick Start (Alternative)

If you prefer Docker:

```bash
# Build Docker image
docker build -t shopguard-backend .

# Run container
docker run -d \
  --name shopguard \
  -p 8000:8000 \
  -e VIVO_APP_ID=your_app_id \
  -e VIVO_APP_KEY=your_app_key \
  shopguard-backend
```

## Common Issues and Solutions

### Issue: "ImportError: No module named 'XXX'"

**Solution**: Install missing dependencies

```bash
pip install -r requirements.txt
```

### Issue: "RAG system not available"

**Solution**: Check knowledge base files

```bash
# Verify files exist
ls -la knowledge_base_embeddings/all_knowledge_embeddings.json

# If missing, contact maintainer for knowledge base files
```

### Issue: "vivo API authentication failed"

**Solution**: Verify your credentials

```bash
# Check your .env file
cat .env | grep VIVO_

# Ensure APP_ID and APP_KEY are correct
# Test with vivo AI platform directly
```

### Issue: "Port 8000 already in use"

**Solution**: Use different port

```bash
# Option 1: Use different port
uvicorn newserver:app --port 8001

# Option 2: Kill existing process
sudo lsof -ti:8000 | xargs kill -9
```

### Issue: "Connection timeout"

**Solution**: Check network and proxy settings

```bash
# Test connectivity
curl -I https://api-ai.vivo.com.cn

# If behind proxy, configure:
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

## Next Steps

After successful installation:

1. **Read the API Documentation**: Check `/docs/api-reference.md` for detailed API usage
2. **Explore Image Input**: See `/docs/how-to-image-input.md` for multimodal capabilities  
3. **Configure Web Search**: Review `/docs/how-to-web-search.md` for enhanced search features
4. **Production Deployment**: Follow `/docs/production-deployment.md` for production setup

## Getting Help

### Check Service Status

```bash
# Health check
curl http://localhost:8000/v1/health

# Server stats
curl http://localhost:8000/v1/stats

# Service info
curl http://localhost:8000/
```

### Enable Debug Mode

For troubleshooting, enable debug mode:

```properties
# In .env file
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

Restart the service to see detailed logs.

### Community Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check `/docs/` folder for detailed guides
- **API Reference**: Visit `http://localhost:8000/docs` for interactive API documentation

## Success Checklist

✅ Python 3.8+ installed  
✅ Repository cloned and dependencies installed  
✅ Environment variables configured  
✅ Knowledge base files present  
✅ Service starts without errors  
✅ Health check returns "healthy"  
✅ Models endpoint returns available models  
✅ Basic API test succeeds  
✅ RAG system shows as available  

If all items are checked, your ShopGuard backend is ready for use! 🎉
