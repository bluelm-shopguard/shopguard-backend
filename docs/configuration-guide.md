# 配置指南 ⚙️

ShopGuard 后端服务的完整配置指南。

## 环境变量

### 必需配置

#### vivo AI 平台凭证

```properties
# 必需：vivo AI 平台认证
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here
```

**如何获取：**

1. 在 [vivo AI 平台](https://developers.vivo.com/product/ai) 注册
2. 创建新应用程序
3. 从您的仪表板复制 APP_ID 和 APP_KEY

#### API 端点

```properties
# vivo AI API 端点（使用这些默认值）
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn
```

### 可选配置

#### 网络搜索服务

```properties
# 可选：启用网络搜索功能
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search
```

**如何获取：**

1. 在 [智谱AI](https://open.bigmodel.cn/) 注册
2. 为网络搜索服务创建 API 密钥
3. 将 API 密钥复制到 WEB_SEARCH_API_KEY

#### 服务器配置

```properties
# 服务器绑定配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# 应用程序设置
DEBUG_MODE=false
LOG_LEVEL=INFO
```

#### 性能设置

```properties
# 并发请求限制
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_SECONDS=30

# 缓存配置
RAG_CACHE_TTL_SECONDS=3600
CONVERSATION_HISTORY_LIMIT=100

# 内存管理
MAX_MEMORY_USAGE_MB=2048
GARBAGE_COLLECTION_THRESHOLD=1000
```

#### 安全设置

```properties
# CORS 配置
ALLOWED_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false

# API 安全
API_KEY_REQUIRED=false
DEFAULT_API_KEY=your_default_api_key

# 速率限制
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10
```

## 配置文件

### 环境模板

#### 开发环境 (.env.development)

```properties
# 开发环境配置

# vivo AI 平台（开发）
VIVO_APP_ID=dev_app_id
VIVO_APP_KEY=dev_app_key

# API 端点
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# 网络搜索（开发）
WEB_SEARCH_API_KEY=dev_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 服务器配置
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# 性能（开发）
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT_SECONDS=60
RAG_CACHE_TTL_SECONDS=1800
CONVERSATION_HISTORY_LIMIT=200

# 安全（开发）
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CORS_ALLOW_CREDENTIALS=true
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=false
```

#### 生产环境 (.env.production)

```properties
# 生产环境配置

# vivo AI 平台（生产）
VIVO_APP_ID=prod_app_id
VIVO_APP_KEY=prod_app_key

# API 端点
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# 网络搜索（生产）
WEB_SEARCH_API_KEY=prod_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 服务器配置
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO

# 性能（生产）
MAX_CONCURRENT_REQUESTS=200
REQUEST_TIMEOUT_SECONDS=30
RAG_CACHE_TTL_SECONDS=7200
CONVERSATION_HISTORY_LIMIT=50

# 安全（生产）
ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com
CORS_ALLOW_CREDENTIALS=false
API_KEY_REQUIRED=true
DEFAULT_API_KEY=your_secure_api_key
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# 数据库（生产）
DATABASE_URL=postgresql://user:password@localhost/shopguard
REDIS_URL=redis://localhost:6379/0
```

### 应用程序配置

#### RAG 系统配置

```python
# rag_config.py
RAG_CONFIG = {
    # 模型设置
    "embedding_model": "m3e-base",
    "vector_dimension": 768,
    "similarity_threshold": 0.0,
    
    # 检索设置
    "default_top_k": 2,
    "max_top_k": 10,
    "max_context_length": 2000,
    
    # 缓存设置
    "enable_cache": True,
    "cache_ttl_seconds": 3600,
    "max_cache_size": 10000,
    
    # 知识库
    "knowledge_base_path": "knowledge_base_embeddings/all_knowledge_embeddings.json",
    "auto_reload": False,
    "reload_interval_seconds": 3600
}
```

#### 网络搜索配置

```python
# search_config.py
SEARCH_CONFIG = {
    # 默认搜索引擎
    "default_engine": "search_std",
    
    # 可用引擎
    "available_engines": [
        "search_std",
        "search_pro", 
        "search_pro_sogou",
        "search_pro_quark",
        "search_pro_jina",
        "search_pro_bing"
    ],
    
    # 搜索参数
    "default_count": 4,
    "max_count": 10,
    "content_size": "medium",  # small, medium, large
    "timeout_seconds": 10,
    
    # 结果处理
    "enable_summarization": True,
    "max_summary_length": 500,
    "remove_duplicates": True,
    
    # 速率限制
    "requests_per_minute": 60,
    "burst_size": 10
}
```

#### 模型配置

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

## 配置加载

### 基于环境的配置

应用程序根据环境自动加载配置：

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
        # 确定环境
        env = os.getenv('ENVIRONMENT', 'development')
        
        # 加载基础配置
        base_env = Path('.env')
        if base_env.exists():
            load_dotenv(base_env)
        
        # 加载特定环境的配置
        env_file = Path(f'.env.{env}')
        if env_file.exists():
            load_dotenv(env_file, override=True)
        
        # 加载本地覆盖
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
            raise ValueError(f"缺少必需的环境变量: {missing_vars}")

# 使用方法
config = Config()
```

### 配置验证

```python
# config_validator.py
import os
import json
from pathlib import Path

def validate_vivo_credentials():
    """验证 vivo AI 平台凭证"""
    app_id = os.getenv('VIVO_APP_ID')
    app_key = os.getenv('VIVO_APP_KEY')
    
    if not app_id or not app_key:
        return False, "缺少 vivo AI 凭证"
    
    if len(app_id) < 10 or len(app_key) < 20:
        return False, "凭证格式无效"
    
    return True, "凭证有效"

def validate_knowledge_base():
    """验证知识库文件"""
    kb_path = Path("knowledge_base_embeddings/all_knowledge_embeddings.json")
    
    if not kb_path.exists():
        return False, f"知识库文件未找到: {kb_path}"
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return False, "知识库为空或格式无效"
        
        return True, f"知识库已加载: {len(data)} 条记录"
    
    except Exception as e:
        return False, f"加载知识库失败: {e}"

def validate_web_search():
    """验证网络搜索配置"""
    api_key = os.getenv('WEB_SEARCH_API_KEY')
    api_url = os.getenv('WEB_SEARCH_URL')
    
    if not api_key:
        return False, "网络搜索 API 密钥未配置"
    
    if not api_url:
        return False, "网络搜索 URL 未配置"
    
    return True, "网络搜索配置有效"

def run_all_validations():
    """运行所有配置验证"""
    validations = [
        ("vivo 凭证", validate_vivo_credentials),
        ("知识库", validate_knowledge_base),
        ("网络搜索", validate_web_search)
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

## Docker 配置

### Docker 环境变量

```yaml
# docker-compose.yml
version: '3.8'

services:
  shopguard-backend:
    build: .
    environment:
      # vivo AI 平台
      - VIVO_APP_ID=${VIVO_APP_ID}
      - VIVO_APP_KEY=${VIVO_APP_KEY}
      
      # API 配置
      - VIVOGPT_API_URI=/vivogpt/completions
      - VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
      - MULTIMODAL_URI=/vivogpt/completions
      - MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
      - RAG_API_URI=/embedding-model-api/predict/batch
      - RAG_API_DOMAIN=api-ai.vivo.com.cn
      
      # 网络搜索
      - WEB_SEARCH_API_KEY=${WEB_SEARCH_API_KEY}
      - WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search
      
      # 服务器配置
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8000
      - DEBUG_MODE=false
      - LOG_LEVEL=INFO
      
      # 性能
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

### Docker 密钥

对于生产环境的 Docker 部署，使用 Docker 密钥：

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

## 配置管理

### 配置热重载

启用开发环境的配置热重载：

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
                print(f"配置文件已更改: {event.src_path}")
                self.callback()

def setup_config_watcher(reload_callback):
    """设置配置文件监听器"""
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        event_handler = ConfigReloadHandler(reload_callback)
        observer = Observer()
        observer.schedule(event_handler, '.', recursive=False)
        observer.start()
        return observer
    return None
```

### 配置 CLI

创建配置管理命令行工具：

```python
# config_cli.py
import click
import json
import os
from pathlib import Path

@click.group()
def cli():
    """配置管理 CLI"""
    pass

@cli.command()
def validate():
    """验证当前配置"""
    from config_validator import run_all_validations
    
    results = run_all_validations()
    for name, result in results.items():
        status = "✅" if result["valid"] else "❌"
        click.echo(f"{status} {name}: {result['message']}")

@cli.command()
@click.option('--env', default='development', help='环境名称')
def create_env(env):
    """创建环境配置模板"""
    env_file = Path(f'.env.{env}')
    
    if env_file.exists():
        click.echo(f"环境文件已存在: {env_file}")
        return
    
    template = """# {env} 环境配置

# vivo AI 平台
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here

# API 端点
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# 网络搜索
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 服务器配置
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO
""".format(env=env.upper())
    
    env_file.write_text(template)
    click.echo(f"已创建环境文件: {env_file}")

@cli.command()
def show_config():
    """显示当前配置"""
    config_vars = [
        'VIVO_APP_ID',
        'VIVO_APP_KEY',
        'WEB_SEARCH_API_KEY',
        'SERVER_HOST',
        'SERVER_PORT',
        'DEBUG_MODE',
        'LOG_LEVEL'
    ]
    
    click.echo("当前配置:")
    click.echo("=" * 50)
    
    for var in config_vars:
        value = os.getenv(var, '未设置')
        if 'KEY' in var and value != '未设置':
            value = value[:8] + '...' if len(value) > 8 else value
        click.echo(f"{var}: {value}")

if __name__ == '__main__':
    cli()
```

使用方法：

```bash
# 验证配置
python config_cli.py validate

# 创建新环境
python config_cli.py create-env --env staging

# 显示当前配置
python config_cli.py show-config
```

## 最佳实践

### 安全最佳实践

1. **永远不要将机密信息提交到版本控制**

   ```bash
   # 添加到 .gitignore
   .env*
   !.env.example
   secrets/
   ```

2. **使用特定环境的配置**

   ```bash
   # 不同环境使用不同文件
   .env.development
   .env.staging
   .env.production
   ```

3. **定期轮换凭证**

   ```bash
   # 每月更新凭证
   VIVO_APP_KEY=new_key_here
   WEB_SEARCH_API_KEY=new_search_key
   ```

### 性能最佳实践

1. **针对您的工作负载进行调优**

   ```properties
   # 高流量网站
   MAX_CONCURRENT_REQUESTS=500
   RAG_CACHE_TTL_SECONDS=14400
   
   # 低流量网站
   MAX_CONCURRENT_REQUESTS=50
   RAG_CACHE_TTL_SECONDS=1800
   ```

2. **监控资源使用情况**

   ```properties
   # 启用监控
   ENABLE_METRICS=true
   METRICS_PORT=9090
   ```

### 维护最佳实践

1. **定期配置审核**

   ```bash
   # 每月配置审查
   python config_cli.py validate
   ```

2. **备份配置**

   ```bash
   # 更改前备份
   cp .env.production .env.production.backup.$(date +%Y%m%d)
   ```

3. **记录配置更改**

   ```bash
   # 保持更改日志
   echo "$(date): 更新 RAG 缓存 TTL 为 7200 秒" >> config_changes.log
   ```

此配置指南为 ShopGuard 后端服务提供了所有配置选项和最佳实践的全面覆盖。
