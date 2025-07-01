# Configuration Guide ⚙️

Complete guide for configuring the ShopGuard backend service.

## Environment Variables

### Required Configuration

#### vivo AI Platform Credentials

```properties
# Required: vivo AI Platform authentication
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here
```

**How to obtain:**

1. Register at [vivo AI Platform](https://developers.vivo.com/product/ai)
2. Create a new application
3. Copy the APP_ID and APP_KEY from your dashboard

#### API Endpoints

```properties
# vivo AI API endpoints (use these defaults)
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn
```

### Optional Configuration

#### Web Search Service

```properties
# Optional: Enable web search functionality
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search
```

**How to obtain:**

1. Register at [智谱AI](https://open.bigmodel.cn/)
2. Create API key for web search service
3. Copy the API key to WEB_SEARCH_API_KEY

#### Server Configuration

```properties
# Server binding configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Application settings
DEBUG_MODE=false
LOG_LEVEL=INFO
```

#### Performance Settings

```properties
# Concurrent request limits
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_SECONDS=30

# Cache configuration
RAG_CACHE_TTL_SECONDS=3600
CONVERSATION_HISTORY_LIMIT=100

# Memory management
MAX_MEMORY_USAGE_MB=2048
GARBAGE_COLLECTION_THRESHOLD=1000
```

#### Security Settings

```properties
# CORS configuration
ALLOWED_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false

# API security
API_KEY_REQUIRED=false
DEFAULT_API_KEY=your_default_api_key

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10
```

## Configuration Files

### Environment Templates

#### Development (.env.development)

```properties
# Development Environment Configuration

# vivo AI Platform (Development)
VIVO_APP_ID=dev_app_id
VIVO_APP_KEY=dev_app_key

# API Endpoints
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# Web Search (Development)
WEB_SEARCH_API_KEY=dev_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# Performance (Development)
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT_SECONDS=60
RAG_CACHE_TTL_SECONDS=1800
CONVERSATION_HISTORY_LIMIT=200

# Security (Development)
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CORS_ALLOW_CREDENTIALS=true
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=false
```

#### Production (.env.production)

```properties
# Production Environment Configuration

# vivo AI Platform (Production)
VIVO_APP_ID=prod_app_id
VIVO_APP_KEY=prod_app_key

# API Endpoints
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# Web Search (Production)
WEB_SEARCH_API_KEY=prod_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO

# Performance (Production)
MAX_CONCURRENT_REQUESTS=200
REQUEST_TIMEOUT_SECONDS=30
RAG_CACHE_TTL_SECONDS=7200
CONVERSATION_HISTORY_LIMIT=50

# Security (Production)
ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
CORS_ALLOW_CREDENTIALS=false
API_KEY_REQUIRED=true
DEFAULT_API_KEY=your_secure_api_key
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# Database (Production)
DATABASE_URL=postgresql://user:password@localhost/shopguard
REDIS_URL=redis://localhost:6379/0
```

### Application Configuration

#### RAG System Configuration

```python
# rag_config.py
RAG_CONFIG = {
    # Model settings
    "embedding_model": "m3e-base",
    "vector_dimension": 768,
    "similarity_threshold": 0.0,
    
    # Retrieval settings
    "default_top_k": 2,
    "max_top_k": 10,
    "max_context_length": 2000,
    
    # Cache settings
    "enable_cache": True,
    "cache_ttl_seconds": 3600,
    "max_cache_size": 10000,
    
    # Knowledge base
    "knowledge_base_path": "knowledge_base_embeddings/all_knowledge_embeddings.json",
    "auto_reload": False,
    "reload_interval_seconds": 3600
}
```

#### Web Search Configuration

```python
# search_config.py
SEARCH_CONFIG = {
    # Default search engine
    "default_engine": "search_std",
    
    # Available engines
    "available_engines": [
        "search_std",
        "search_pro", 
        "search_pro_sogou",
        "search_pro_quark",
        "search_pro_jina",
        "search_pro_bing"
    ],
    
    # Search parameters
    "default_count": 4,
    "max_count": 10,
    "content_size": "medium",  # small, medium, large
    "timeout_seconds": 10,
    
    # Result processing
    "enable_summarization": True,
    "max_summary_length": 500,
    "remove_duplicates": True,
    
    # Rate limiting
    "requests_per_minute": 60,
    "burst_size": 10
}
```

#### Model Configuration

```python
# model_config.py
MODEL_CONFIG = {
    "vivo-BlueLM-TB-Pro": {
        "type": "text",
        "max_tokens": 4096,
        "default_temperature": 0.7,
        "supports_function_call": True,
        "supports_streaming": True
    },
    
    "vivo-BlueLM-V-2.0": {
        "type": "multimodal",
        "max_tokens": 4096,
        "default_temperature": 0.9,
        "supports_function_call": True,
        "supports_streaming": True,
        "supports_vision": True,
        "max_image_size_mb": 10,
        "supported_formats": ["jpeg", "png", "webp", "gif"]
    }
}
```

## Configuration Loading

### Environment-based Configuration

The application automatically loads configuration based on the environment:

```python
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self):
        self.load_environment()
        self.validate_config()
    
    def load_environment(self):
        # Determine environment
        env = os.getenv('ENVIRONMENT', 'development')
        
        # Load base configuration
        base_env = Path('.env')
        if base_env.exists():
            load_dotenv(base_env)
        
        # Load environment-specific configuration
        env_file = Path(f'.env.{env}')
        if env_file.exists():
            load_dotenv(env_file, override=True)
        
        # Load local overrides
        local_env = Path('.env.local')
        if local_env.exists():
            load_dotenv(local_env, override=True)
    
    def validate_config(self):
        required_vars = [
            'VIVO_APP_ID',
            'VIVO_APP_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

# Usage
config = Config()
```

### Configuration Validation

```python
# config_validator.py
import os
import json
from pathlib import Path

def validate_vivo_credentials():
    """Validate vivo AI platform credentials"""
    app_id = os.getenv('VIVO_APP_ID')
    app_key = os.getenv('VIVO_APP_KEY')
    
    if not app_id or not app_key:
        return False, "Missing vivo AI credentials"
    
    if len(app_id) < 10 or len(app_key) < 20:
        return False, "Invalid credential format"
    
    return True, "Credentials valid"

def validate_knowledge_base():
    """Validate knowledge base files"""
    kb_path = Path("knowledge_base_embeddings/all_knowledge_embeddings.json")
    
    if not kb_path.exists():
        return False, f"Knowledge base file not found: {kb_path}"
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return False, "Knowledge base is empty or invalid format"
        
        return True, f"Knowledge base loaded: {len(data)} entries"
    
    except Exception as e:
        return False, f"Failed to load knowledge base: {e}"

def validate_web_search():
    """Validate web search configuration"""
    api_key = os.getenv('WEB_SEARCH_API_KEY')
    api_url = os.getenv('WEB_SEARCH_URL')
    
    if not api_key:
        return False, "Web search API key not configured"
    
    if not api_url:
        return False, "Web search URL not configured"
    
    return True, "Web search configuration valid"

def run_all_validations():
    """Run all configuration validations"""
    validations = [
        ("vivo Credentials", validate_vivo_credentials),
        ("Knowledge Base", validate_knowledge_base),
        ("Web Search", validate_web_search)
    ]
    
    results = {}
    for name, validator in validations:
        try:
            valid, message = validator()
            results[name] = {"valid": valid, "message": message}
        except Exception as e:
            results[name] = {"valid": False, "message": str(e)}
    
    return results

if __name__ == "__main__":
    results = run_all_validations()
    for name, result in results.items():
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {name}: {result['message']}")
```

## Docker Configuration

### Docker Environment Variables

```yaml
# docker-compose.yml
version: '3.8'

services:
  shopguard-backend:
    build: .
    environment:
      # vivo AI Platform
      - VIVO_APP_ID=${VIVO_APP_ID}
      - VIVO_APP_KEY=${VIVO_APP_KEY}
      
      # API Configuration
      - VIVOGPT_API_URI=/vivogpt/completions
      - VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
      - MULTIMODAL_URI=/vivogpt/completions
      - MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
      - RAG_API_URI=/embedding-model-api/predict/batch
      - RAG_API_DOMAIN=api-ai.vivo.com.cn
      
      # Web Search
      - WEB_SEARCH_API_KEY=${WEB_SEARCH_API_KEY}
      - WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search
      
      # Server Configuration
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8000
      - DEBUG_MODE=false
      - LOG_LEVEL=INFO
      
      # Performance
      - MAX_CONCURRENT_REQUESTS=100
      - REQUEST_TIMEOUT_SECONDS=30
      - RAG_CACHE_TTL_SECONDS=3600
      
    ports:
      - "8000:8000"
    volumes:
      - ./knowledge_base_embeddings:/app/knowledge_base_embeddings:ro
      - ./logs:/app/logs
    restart: unless-stopped
```

### Docker Secrets

For production Docker deployments, use Docker secrets:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  shopguard-backend:
    image: shopguard-backend:latest
    secrets:
      - vivo_app_id
      - vivo_app_key
      - web_search_api_key
    environment:
      - VIVO_APP_ID_FILE=/run/secrets/vivo_app_id
      - VIVO_APP_KEY_FILE=/run/secrets/vivo_app_key
      - WEB_SEARCH_API_KEY_FILE=/run/secrets/web_search_api_key

secrets:
  vivo_app_id:
    external: true
  vivo_app_key:
    external: true
  web_search_api_key:
    external: true
```

## Configuration Management

### Configuration Hot Reload

Enable configuration hot reload for development:

```python
# hot_reload.py
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.env'):
            current_time = time.time()
            if (event.src_path not in self.last_modified or 
                current_time - self.last_modified[event.src_path] > 1):
                self.last_modified[event.src_path] = current_time
                print(f"Configuration file changed: {event.src_path}")
                self.callback()

def setup_config_watcher(reload_callback):
    """Setup configuration file watcher"""
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        event_handler = ConfigReloadHandler(reload_callback)
        observer = Observer()
        observer.schedule(event_handler, '.', recursive=False)
        observer.start()
        return observer
    return None
```

### Configuration CLI

Create a configuration management CLI:

```python
# config_cli.py
import click
import json
import os
from pathlib import Path

@click.group()
def cli():
    """Configuration management CLI"""
    pass

@cli.command()
def validate():
    """Validate current configuration"""
    from config_validator import run_all_validations
    
    results = run_all_validations()
    for name, result in results.items():
        status = "✅" if result["valid"] else "❌"
        click.echo(f"{status} {name}: {result['message']}")

@cli.command()
@click.option('--env', default='development', help='Environment name')
def create_env(env):
    """Create environment configuration template"""
    env_file = Path(f'.env.{env}')
    
    if env_file.exists():
        click.echo(f"Environment file already exists: {env_file}")
        return
    
    template = """# {env} Environment Configuration

# vivo AI Platform
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here

# API Endpoints
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# Web Search
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO
""".format(env=env.upper())
    
    env_file.write_text(template)
    click.echo(f"Created environment file: {env_file}")

@cli.command()
def show_config():
    """Show current configuration"""
    config_vars = [
        'VIVO_APP_ID',
        'VIVO_APP_KEY',
        'WEB_SEARCH_API_KEY',
        'SERVER_HOST',
        'SERVER_PORT',
        'DEBUG_MODE',
        'LOG_LEVEL'
    ]
    
    click.echo("Current Configuration:")
    click.echo("=" * 50)
    
    for var in config_vars:
        value = os.getenv(var, 'Not set')
        if 'KEY' in var and value != 'Not set':
            value = value[:8] + '...' if len(value) > 8 else value
        click.echo(f"{var}: {value}")

if __name__ == '__main__':
    cli()
```

Usage:

```bash
# Validate configuration
python config_cli.py validate

# Create new environment
python config_cli.py create-env --env staging

# Show current configuration
python config_cli.py show-config
```

## Best Practices

### Security Best Practices

1. **Never commit secrets to version control**

   ```bash
   # Add to .gitignore
   .env*
   !.env.example
   secrets/
   ```

2. **Use environment-specific configurations**

   ```bash
   # Different files for different environments
   .env.development
   .env.staging
   .env.production
   ```

3. **Rotate credentials regularly**

   ```bash
   # Update credentials monthly
   VIVO_APP_KEY=new_key_here
   WEB_SEARCH_API_KEY=new_search_key
   ```

### Performance Best Practices

1. **Tune for your workload**

   ```properties
   # High-traffic sites
   MAX_CONCURRENT_REQUESTS=500
   RAG_CACHE_TTL_SECONDS=14400
   
   # Low-traffic sites
   MAX_CONCURRENT_REQUESTS=50
   RAG_CACHE_TTL_SECONDS=1800
   ```

2. **Monitor resource usage**

   ```properties
   # Enable monitoring
   ENABLE_METRICS=true
   METRICS_PORT=9090
   ```

### Maintenance Best Practices

1. **Regular configuration audits**

   ```bash
   # Monthly configuration review
   python config_cli.py validate
   ```

2. **Backup configurations**

   ```bash
   # Backup before changes
   cp .env.production .env.production.backup.$(date +%Y%m%d)
   ```

3. **Document configuration changes**

   ```bash
   # Keep a changelog
   echo "$(date): Updated RAG cache TTL to 7200s" >> config_changes.log
   ```

This configuration guide provides comprehensive coverage of all configuration options and best practices for the ShopGuard backend service.
