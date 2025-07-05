# API 参考 📚

ShopGuard 后端服务的完整 API 文档。

## 基本信息

- **基础 URL**: `http://localhost:8000`
- **API 版本**: v1
- **内容类型**: `application/json`
- **认证**: Bearer Token（可选）

## 核心端点

### 1. 健康检查

检查服务状态和系统健康情况。

```http
GET /v1/health
```

**响应：**

```json
{
  "status": "healthy",
  "timestamp": 1703025600,
  "rag_available": true,
  "active_sessions": 12,
  "system_info": {
    "rag_initialized": true,
    "knowledge_base_size": 10297
  }
}
```

**状态码：**

- `200` - 服务健康
- `503` - 服务不健康

### 2. 服务统计

获取当前服务统计信息和指标。

```http
GET /v1/stats
```

**响应：**

```json
{
  "active_sessions": 12,
  "total_messages": 1847,
  "rag_status": "available",
  "knowledge_base_entries": 10297
}
```

### 3. 可用模型

列出所有可用的 AI 模型。

```http
GET /v1/models
```

**响应：**

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

### 4. 聊天补全（主要端点）

创建支持文本、图像、RAG 和网络搜索的聊天补全。

```http
POST /v1/chat/completions
```

## 聊天补全 API

### 请求参数

#### 核心参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | ✅ | - | 用于补全的模型 |
| `messages` | array | ✅ | - | 对话消息列表 |
| `temperature` | float | ❌ | 0.7 | 随机性控制 (0.0-2.0) |
| `max_tokens` | integer | ❌ | 1024 | 最大生成令牌数 |
| `top_p` | float | ❌ | 1.0 | 核采样参数 |
| `stream` | boolean | ❌ | false | 启用流式响应 |
| `user` | string | ❌ | - | 用户标识符 |

#### ShopGuard 特有参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `enable_rag` | boolean | ❌ | true | 启用 RAG 知识检索 |
| `rag_top_k` | integer | ❌ | 2 | 检索的知识项数量 |
| `user_type` | string | ❌ | "普通用户" | 用户类型，用于个性化响应 |
| `user_id` | string | ❌ | - | 用于会话管理的用户 ID |
| `extra` | object | ❌ | {} | 额外的模型参数 |

#### 高级参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `presence_penalty` | float | ❌ | 0.0 | 存在惩罚 (-2.0 到 2.0) |
| `frequency_penalty` | float | ❌ | 0.0 | 频率惩罚 (-2.0 到 2.0) |
| `stop` | array | ❌ | null | 停止序列 |
| `n` | integer | ❌ | 1 | 要生成的补全数量 |

### 可用模型

#### vivo-BlueLM-TB-Pro

- **类型**: 纯文本模型
- **功能**: 文本生成、函数调用、RAG 增强
- **最适合**: 纯文本对话、购物欺诈分析
- **最大令牌数**: 4096

#### vivo-BlueLM-V-2.0  

- **类型**: 多模态模型（文本 + 视觉）
- **功能**: 图像理解、OCR、多模态分析
- **最适合**: 图像分析、屏幕截图检查、视觉欺诈检测
- **最大令牌数**: 4096

### 消息格式

#### 1. 简单文本消息

```json
{
  "role": "user",
  "content": "这个商品价格合理吗？"
}
```

#### 2. OpenAI 视觉格式

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "分析这个商品页面是否有诈骗风险"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQ..."
      }
    }
  ]
}
```

#### 3. Base64 图像格式

```json
{
  "role": "user",
  "content": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

#### 4. 混合内容格式

```json
{
  "role": "user",
  "contentType": "image",
  "content": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

### 用户类型

可用于个性化响应的用户类型：

- `学生` - 学生（教育重点）
- `老师` - 教师（教育权威）  
- `开发者` - 开发者（技术重点）
- `普通用户` - 普通用户（一般消费者）

### 请求示例

#### 基本文本查询

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [
      {
        "role": "user",
        "content": "有人说iPhone只要1000元，这是诈骗吗？"
      }
    ],
    "enable_rag": true,
    "user_type": "普通用户"
  }'
```

#### 图像分析查询

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-V-2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这个购物页面安全吗？"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE"
            }
          }
        ]
      }
    ],
    "enable_rag": true,
    "rag_top_k": 3,
    "user_type": "普通用户"
  }'
```

#### 流式请求

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [
      {
        "role": "user",
        "content": "网购时如何避免诈骗？"
      }
    ],
    "stream": true,
    "enable_rag": true
  }'
```

### 响应格式

#### 标准响应

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703025600,
  "model": "vivo-BlueLM-TB-Pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant", 
        "content": "根据我的分析，这个价格确实存在诈骗风险...",
        "function_call": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}
```

#### 流式响应

服务器发送事件格式：

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"content":"根据"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"content":"我的"},"finish_reason":null}]}

data: [DONE]
```

## 错误处理

### 错误响应格式

```json
{
  "error": {
    "message": "错误描述",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### 常见错误码

| HTTP 状态 | 错误类型 | 说明 |
|----------|----------|------|
| 400 | `invalid_request_error` | 无效的请求参数 |
| 401 | `authentication_error` | 无效的 API 密钥 |
| 403 | `permission_error` | 权限不足 |
| 404 | `not_found_error` | 资源未找到 |
| 429 | `rate_limit_error` | 超过速率限制 |
| 500 | `internal_server_error` | 服务器错误 |
| 502 | `service_unavailable` | 上游服务不可用 |

### 错误响应示例

#### 无效模型错误

```json
{
  "error": {
    "message": "模型 'invalid-model' 不存在",
    "type": "invalid_request_error", 
    "param": "model",
    "code": "model_not_found"
  }
}
```

#### 速率限制错误

```json
{
  "error": {
    "message": "超过速率限制。请稍后再试。",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

## Python SDK 示例

### 基本用法

```python
import requests
import json

class ShopGuardAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def chat_completion(self, messages, model="vivo-BlueLM-TB-Pro", **kwargs):
        """创建聊天补全"""
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers
        )
        
        response.raise_for_status()
        return response.json()
    
    def analyze_text(self, text, user_type="普通用户"):
        """分析文本以检测欺诈"""
        messages = [{"role": "user", "content": text}]
        
        return self.chat_completion(
            messages=messages,
            enable_rag=True,
            user_type=user_type
        )
    
    def analyze_image(self, image_base64, question, user_type="普通用户"):
        """分析图像和问题"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }]
        
        return self.chat_completion(
            messages=messages,
            model="vivo-BlueLM-V-2.0",
            enable_rag=True,
            user_type=user_type
        )
    
    def health_check(self):
        """检查服务健康"""
        response = requests.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return response.json()

# 使用示例
api = ShopGuardAPI()

# 文本分析
result = api.analyze_text("有人让我转账买iPhone，靠谱吗？")
print(result['choices'][0]['message']['content'])

# 健康检查
health = api.health_check()
print(f"服务状态: {health['status']}")
```

### 流式示例

```python
import requests
import json

def stream_chat(messages, model="vivo-BlueLM-TB-Pro"):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "enable_rag": True
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # 移除 'data: ' 前缀
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    continue

# 使用方法
messages = [{"role": "user", "content": "如何识别网购诈骗？"}]
stream_chat(messages)
```

## 速率限制

### 默认限制

- **每分钟请求数**: 60
- **并发请求数**: 10
- **每分钟令牌数**: 100,000

### 响应头

响应头中包含速率限制信息：

```text
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1703025660
```

## WebSocket 支持（未来功能）

WebSocket 实时通信支持计划在未来版本中推出：

```javascript
// 未来的 WebSocket API（即将推出）
const ws = new WebSocket('ws://localhost:8000/v1/ws');
ws.send(JSON.stringify({
    type: 'chat',
    model: 'vivo-BlueLM-TB-Pro',
    message: 'Hello'
}));
```

## 最佳实践

### 请求优化

1. **批量处理相关查询**在对话上下文中
2. **使用适当的模型**（文本 vs 多模态）
3. **启用 RAG**用于欺诈检测查询
4. **设置合理的超时**用于长请求
5. **优雅处理错误**带重试逻辑

### 性能提示

1. **重用会话**用于相关对话
2. **缓存常见查询**
3. **使用流式**用于长响应
4. **压缩图像**在发送前
5. **监控使用情况**保持在限制内

### 安全考虑

1. **验证输入**在发送到 API 前
2. **清理响应**在显示给用户前
3. **在生产环境使用 HTTPS**
4. **定期轮换 API 密钥**
5. **记录安全事件**用于监控

此 API 参考为与 ShopGuard 后端服务集成提供了完整的文档。有关其他示例和高级用法，请参阅 `/docs` 文件夹中的其他文档文件。
