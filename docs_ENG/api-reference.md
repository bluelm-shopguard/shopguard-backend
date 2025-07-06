# API Reference 📚

Complete API documentation for the ShopGuard backend service.

## Base Information

- **Base URL**: `http://localhost:8000`
- **API Version**: v1
- **Content Type**: `application/json`
- **Authentication**: Bearer Token (optional)

## Core Endpoints

### 1. Health Check

Check service status and system health.

```http
GET /v1/health
```

**Response:**

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

**Status Codes:**

- `200` - Service healthy
- `503` - Service unhealthy

### 2. Service Statistics

Get current service statistics and metrics.

```http
GET /v1/stats
```

**Response:**

```json
{
  "active_sessions": 12,
  "total_messages": 1847,
  "rag_status": "available",
  "knowledge_base_entries": 10297
}
```

### 3. Available Models

List all available AI models.

```http
GET /v1/models
```

**Response:**

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

### 4. Chat Completions (Main Endpoint)

Create chat completions with support for text, images, RAG, and web search.

```http
POST /v1/chat/completions
```

## Chat Completions API

### Request Parameters

#### Core Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | ✅ | - | Model to use for completion |
| `messages` | array | ✅ | - | List of conversation messages |
| `temperature` | float | ❌ | 0.7 | Randomness control (0.0-2.0) |
| `max_tokens` | integer | ❌ | 1024 | Maximum tokens to generate |
| `top_p` | float | ❌ | 1.0 | Nucleus sampling parameter |
| `stream` | boolean | ❌ | false | Enable streaming response |
| `user` | string | ❌ | - | User identifier for tracking |

#### ShopGuard-Specific Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `enable_rag` | boolean | ❌ | true | Enable RAG knowledge retrieval |
| `rag_top_k` | integer | ❌ | 2 | Number of knowledge items to retrieve |
| `user_type` | string | ❌ | "普通用户" | User type for personalized responses |
| `user_id` | string | ❌ | - | User ID for session management |
| `extra` | object | ❌ | {} | Additional model parameters |

#### Advanced Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `presence_penalty` | float | ❌ | 0.0 | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | ❌ | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `stop` | array | ❌ | null | Stop sequences |
| `n` | integer | ❌ | 1 | Number of completions to generate |

### Available Models

#### vivo-BlueLM-TB-Pro

- **Type**: Text-only model
- **Capabilities**: Text generation, function calling, RAG enhancement
- **Best for**: Pure text conversations, shopping fraud analysis
- **Max tokens**: 4096

#### vivo-BlueLM-V-2.0  

- **Type**: Multimodal model (text + vision)
- **Capabilities**: Image understanding, OCR, multimodal analysis
- **Best for**: Image analysis, screenshot examination, visual fraud detection
- **Max tokens**: 4096

### Message Formats

#### 1. Simple Text Message

```json
{
  "role": "user",
  "content": "这个商品价格合理吗？"
}
```

#### 2. OpenAI Vision Format

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

#### 3. Base64 Image Format

```json
{
  "role": "user",
  "content": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

#### 4. Mixed Content Format

```json
{
  "role": "user",
  "contentType": "image",
  "content": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

### User Types

Available user types for personalized responses:

- `学生` - Student (educational focus)
- `老师` - Teacher (educational authority)  
- `开发者` - Developer (technical focus)
- `普通用户` - Regular user (general consumer)

### Request Examples

#### Basic Text Query

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

#### Image Analysis Query

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

#### Streaming Request

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

### Response Formats

#### Standard Response

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

#### Streaming Response

Server-Sent Events format:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"content":"根据"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703025600,"model":"vivo-BlueLM-TB-Pro","choices":[{"index":0,"delta":{"content":"我的"},"finish_reason":null}]}

data: [DONE]
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Common Error Codes

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| 400 | `invalid_request_error` | Invalid request parameters |
| 401 | `authentication_error` | Invalid API key |
| 403 | `permission_error` | Insufficient permissions |
| 404 | `not_found_error` | Resource not found |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `internal_server_error` | Server error |
| 502 | `service_unavailable` | Upstream service unavailable |

### Example Error Responses

#### Invalid Model Error

```json
{
  "error": {
    "message": "The model 'invalid-model' does not exist",
    "type": "invalid_request_error", 
    "param": "model",
    "code": "model_not_found"
  }
}
```

#### Rate Limit Error

```json
{
  "error": {
    "message": "Rate limit exceeded. Please try again later.",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

## Python SDK Example

### Basic Usage

```python
import requests
import json

class ShopGuardAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def chat_completion(self, messages, model="vivo-BlueLM-TB-Pro", **kwargs):
        """Create a chat completion"""
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
        """Analyze text for fraud detection"""
        messages = [{"role": "user", "content": text}]
        
        return self.chat_completion(
            messages=messages,
            enable_rag=True,
            user_type=user_type
        )
    
    def analyze_image(self, image_base64, question, user_type="普通用户"):
        """Analyze image with question"""
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
        """Check service health"""
        response = requests.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return response.json()

# Usage Example
api = ShopGuardAPI()

# Text analysis
result = api.analyze_text("有人让我转账买iPhone，靠谱吗？")
print(result['choices'][0]['message']['content'])

# Health check
health = api.health_check()
print(f"Service status: {health['status']}")
```

### Streaming Example

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
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    continue

# Usage
messages = [{"role": "user", "content": "如何识别网购诈骗？"}]
stream_chat(messages)
```

## Rate Limits

### Default Limits

- **Requests per minute**: 60
- **Concurrent requests**: 10
- **Tokens per minute**: 100,000

### Headers

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1703025660
```

## WebSocket Support (Future)

WebSocket support for real-time communication is planned for future releases:

```javascript
// Future WebSocket API (coming soon)
const ws = new WebSocket('ws://localhost:8000/v1/ws');
ws.send(JSON.stringify({
    type: 'chat',
    model: 'vivo-BlueLM-TB-Pro',
    message: 'Hello'
}));
```

## Best Practices

### Request Optimization

1. **Batch related queries** in conversation context
2. **Use appropriate models** (text vs multimodal)
3. **Enable RAG** for fraud detection queries
4. **Set reasonable timeouts** for long requests
5. **Handle errors gracefully** with retry logic

### Performance Tips

1. **Reuse sessions** for related conversations
2. **Cache frequent queries** when appropriate
3. **Use streaming** for long responses
4. **Compress images** before sending
5. **Monitor usage** to stay within limits

### Security Considerations

1. **Validate input** before sending to API
2. **Sanitize responses** before displaying to users
3. **Use HTTPS** in production
4. **Rotate API keys** regularly
5. **Log security events** for monitoring

This API reference provides complete documentation for integrating with the ShopGuard backend service. For additional examples and advanced usage, see the other documentation files in the `/docs` folder.
