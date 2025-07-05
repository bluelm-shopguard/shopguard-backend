# 快速开始指南 🚀

让 ShopGuard 后端服务在几分钟内启动运行。

## 前提条件

开始之前，请确保您已安装：

- **Python 3.8+** (推荐 Python 3.9+)
- **vivo AI 平台账户**，拥有有效的 APP_ID 和 APP_KEY
- **Git** 用于克隆仓库

## 步骤 1：克隆和设置

```bash
# 克隆仓库
git clone https://github.com/your-org/shopguard-backend.git
cd shopguard-backend

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 步骤 2：环境配置

创建您的环境配置：

```bash
# 复制环境模板
cp .env.example .env
```

编辑 `.env` 文件添加您的凭证：

```properties
# vivo AI 平台配置
VIVO_APP_ID=your_app_id_here
VIVO_APP_KEY=your_app_key_here

# API 端点（使用默认值）
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# 网络搜索（可选）
WEB_SEARCH_API_KEY=your_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## 步骤 3：验证知识库

检查知识库文件是否存在：

```bash
# 检查知识库文件
ls -la knowledge_base_embeddings/

# 您应该看到：
# all_knowledge_embeddings.json
```

如果文件缺失，请联系项目维护者获取这些文件。

## 步骤 4：启动服务

### 开发模式

```bash
# 直接使用 Python 启动
python newserver.py
```

### 生产模式（推荐）

```bash
# 使用 uvicorn 启动
uvicorn newserver:app --host 0.0.0.0 --port 8000 --reload
```

您应该看到类似这样的输出：

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 步骤 5：验证安装

### 健康检查

```bash
curl http://localhost:8000/v1/health
```

期望的响应：

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

### 检查可用模型

```bash
curl http://localhost:8000/v1/models
```

期望的响应：

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

## 步骤 6：测试基本功能

### 纯文本查询

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

### 使用 Python 测试

创建 `test_api.py`：

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
        print("✅ API 响应成功!")
        print(f"响应: {result['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"❌ API 测试失败: {e}")

if __name__ == "__main__":
    test_basic_query()
```

运行测试：

```bash
python test_api.py
```

## 配置选项

### 基本设置

| 变量 | 说明 | 必需 |
|------|------|------|
| `VIVO_APP_ID` | 您的 vivo AI 平台应用 ID | ✅ |
| `VIVO_APP_KEY` | 您的 vivo AI 平台应用密钥 | ✅ |
| `WEB_SEARCH_API_KEY` | 网络搜索服务 API 密钥 | ❌ |

### 可选性能设置

```properties
# 性能配置
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_SECONDS=30
RAG_CACHE_TTL_SECONDS=3600
CONVERSATION_HISTORY_LIMIT=100

# 日志
LOG_LEVEL=INFO
DEBUG_MODE=false
```

## Docker 快速启动（备选）

如果您偏好使用 Docker：

```bash
# 构建 Docker 镜像
docker build -t shopguard-backend .

# 运行容器
docker run -d \
  --name shopguard \
  -p 8000:8000 \
  -e VIVO_APP_ID=your_app_id \
  -e VIVO_APP_KEY=your_app_key \
  shopguard-backend
```

## 常见问题和解决方案

### 问题："ImportError: No module named 'XXX'"

**解决方案**：安装缺失的依赖

```bash
pip install -r requirements.txt
```

### 问题："RAG system not available"

**解决方案**：检查知识库文件

```bash
# 验证文件存在
ls -la knowledge_base_embeddings/all_knowledge_embeddings.json

# 如果缺失，请联系维护者获取知识库文件
```

### 问题："vivo API authentication failed"

**解决方案**：验证您的凭证

```bash
# 检查您的 .env 文件
cat .env | grep VIVO_

# 确保 APP_ID 和 APP_KEY 正确
# 直接在 vivo AI 平台测试
```

### 问题："Port 8000 already in use"

**解决方案**：使用不同端口

```bash
# 选项 1：使用不同端口
uvicorn newserver:app --port 8001

# 选项 2：杀死现有进程
sudo lsof -ti:8000 | xargs kill -9
```

### 问题："Connection timeout"

**解决方案**：检查网络和代理设置

```bash
# 测试连接
curl -I https://api-ai.vivo.com.cn

# 如果在代理后面，配置：
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

## 下一步

安装成功后：

1. **阅读 API 文档**：查看 `/docs/api-reference.md` 了解详细 API 使用方法
2. **探索图片输入**：查看 `/docs/how-to-image-input.md` 了解多模态功能  
3. **配置网络搜索**：查看 `/docs/how-to-web-search.md` 了解增强搜索功能
4. **生产环境部署**：按照 `/docs/production-deployment.md` 进行生产设置

## 获取帮助

### 检查服务状态

```bash
# 健康检查
curl http://localhost:8000/v1/health

# 服务统计
curl http://localhost:8000/v1/stats

# 服务信息
curl http://localhost:8000/
```

### 启用调试模式

用于故障排除，启用调试模式：

```properties
# 在 .env 文件中
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

重启服务查看详细日志。

### 社区支持

- **GitHub Issues**：报告错误和功能请求
- **文档**：查看 `/docs/` 文件夹中的详细指南
- **API 参考**：访问 `http://localhost:8000/docs` 获取交互式 API 文档

## 成功检查清单

✅ 已安装 Python 3.8+  
✅ 已克隆仓库并安装依赖  
✅ 已配置环境变量  
✅ 知识库文件存在  
✅ 服务启动无错误  
✅ 健康检查返回"healthy"  
✅ 模型端点返回可用模型  
✅ 基本 API 测试成功  
✅ RAG 系统显示为可用  

如果所有项目都已检查，您的 ShopGuard 后端已准备就绪！ 🎉
