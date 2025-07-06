# 如何使用网络搜索 🔍

本指南说明如何在 ShopGuard 后端服务中启用和使用网络搜索功能。

## 网络搜索概述

网络搜索功能会自动从互联网检索实时信息，以增强欺诈检测并提供最新的市场数据。

## 自动网络搜索触发

网络搜索基于查询的内容和上下文**自动触发**。系统会智能地决定何时搜索额外信息。

### 购物上下文检测

服务会自动检测购物相关查询，并为以下情况启用网络搜索：

- 价格比较和市场验证
- 平台信誉检查  
- 产品可用性确认
- 当前市场趋势和评论

### 搜索激活场景

网络搜索通常在以下情况被触发：

1. **价格相关查询**: "这个价格合理吗？", "市场价格是多少？"
2. **平台验证**: "这个网站可靠吗？", "这个平台安全吗？"
3. **产品真实性**: "这个商品是正品吗？", "官方售价多少？"
4. **实时信息**: "最新的诈骗手段", "当前市场行情"

## 配置选项

### 用户类型影响

设置 `user_type` 参数以影响搜索行为：

```json
{
  "model": "vivo-BlueLM-TB-Pro",
  "messages": [...],
  "user_type": "学生"
}
```

**可用用户类型：**

- `学生` - 学生（教育重点）
- `老师` - 教师（教育权威）
- `开发者` - 开发者（技术重点）
- `普通用户` - 普通用户（一般消费者）

### RAG 集成

启用 RAG（检索增强生成）以将网络搜索与知识库相结合：

```json
{
  "enable_rag": true,
  "rag_top_k": 3
}
```

## API 请求示例

### 基本购物查询（网络搜索）

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [
      {
        "role": "user",
        "content": "这个商品价格合理吗？帮我查一下网上的价格对比"
      }
    ],
    "user_type": "普通用户",
    "enable_rag": true
  }'
```

### 平台验证请求

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro", 
    "messages": [
      {
        "role": "user",
        "content": "淘宝上这个店铺可靠吗？请帮我查一下商家信息"
      }
    ],
    "user_type": "普通用户",
    "enable_rag": true,
    "rag_top_k": 5
  }'
```

### 图片 + 网络搜索组合

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "分析这个购物页面，并查询市场价格对比"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE_HERE"
            }
          }
        ]
      }
    ],
    "user_type": "普通用户",
    "enable_rag": true
  }'
```

## 可用搜索引擎

该服务支持多个搜索引擎以获得全面的结果：

### 主要搜索引擎

1. **search_std** - 标准搜索（默认）
2. **search_pro** - 专业搜索，增强结果
3. **search_pro_sogou** - 搜狗搜索引擎
4. **search_pro_quark** - 夸克搜索引擎  
5. **search_pro_jina** - Jina 搜索引擎
6. **search_pro_bing** - 必应搜索引擎

### 搜索引擎选择

系统会根据以下因素自动选择最合适的搜索引擎：

- 查询类型和复杂度
- 区域偏好
- 搜索结果质量
- 响应时间要求

## Python 客户端示例

```python
import requests

def search_enabled_query(question, user_type="普通用户"):
    payload = {
        "model": "vivo-BlueLM-TB-Pro",
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "user_type": user_type,
        "enable_rag": True,
        "rag_top_k": 3,
        "stream": False
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# 使用示例
result = search_enabled_query(
    "这个iPhone价格1000元是真的吗？请帮我查一下市场价格",
    "普通用户"
)

print(result['choices'][0]['message']['content'])
```

## 搜索结果处理

### 自动摘要

该服务会自动：

- **压缩**长搜索结果为关键信息
- **提取**关键事实和价格
- **摘要**多个来源成连贯的响应
- **过滤**不相关或重复的信息

### 内容大小选项

搜索结果以不同的详细程度进行处理：

- **小**: 仅简要摘要
- **中**: 平衡的详细信息（默认）  
- **大**: 全面的信息

## 环境配置

确保设置这些环境变量：

```properties
# 网络搜索配置
WEB_SEARCH_API_KEY=your_web_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 搜索引擎设置
SEARCH_DEFAULT_ENGINE=search_std
SEARCH_TIMEOUT_SECONDS=10
SEARCH_DEFAULT_COUNT=4
SEARCH_CONTENT_SIZE=medium
```

## 高级配置

### 搜索参数

您可以通过额外参数影响搜索行为：

```json
{
  "model": "vivo-BlueLM-TB-Pro",
  "messages": [...],
  "extra": {
    "search_timeout": 15,
    "search_count": 6,
    "search_engine": "search_pro"
  }
}
```

### 自定义搜索查询

对于特定的搜索需求，您可以暗示搜索词：

```json
{
  "role": "user", 
  "content": "请搜索'iPhone 15 官方价格'并分析价格合理性"
}
```

## 常见使用场景

### 1. 价格验证

查询产品市场价格以检测不现实的定价：

```python
query = "这个MacBook Pro只要5000元，是真的吗？"
```

### 2. 平台信誉

验证购物平台的合法性：

```python
query = "拼多多这个平台可靠吗？"
```

### 3. 诈骗检测

搜索已知的诈骗模式和报告：

```python
query = "有人让我扫码领取iPhone，这是诈骗吗？"
```

### 4. 市场趋势分析

获取当前市场信息和趋势：

```python
query = "最近有什么新的网购诈骗手段？"
```

## 最佳实践

### 查询优化

- **具体化**: 包含产品名称、价格或平台名称
- **要求比较**: 请求市场价格比较
- **提及验证**: 询问信誉或合法性检查
- **包含上下文**: 提供背景信息

### 错误处理

```python
def handle_search_query(query):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'error' in result:
            return f"搜索失败: {result['error']['message']}"
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        return "搜索超时，请稍后再试"
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"
```

## 故障排除

### 常见问题

1. **无搜索结果**
   - 检查网络搜索 API 密钥是否已配置
   - 验证互联网连接
   - 尝试不同的查询措辞

2. **搜索超时**
   - 增加超时设置
   - 使用更具体的查询
   - 检查 API 服务状态

3. **结果不相关**
   - 使查询更具体
   - 包含上下文和细节
   - 尝试不同的用户类型

### 监控搜索使用

通过健康端点检查搜索功能：

```bash
curl http://localhost:8000/v1/health
```

在响应中查找 `web_search_available`。

## 速率限制和配额

请注意：

- **API 配额**: 网络搜索 API 有使用限制
- **速率限制**: 自动节流防止过度使用  
- **成本优化**: 仅在必要时触发搜索

## 获取帮助

如果网络搜索不工作：

1. 验证 API 密钥是否正确配置
2. 检查服务日志中的搜索相关错误
3. 先使用简单、清晰的查询进行测试
4. 检查您的 API 配额使用情况

网络搜索功能通过提供实时市场情报和验证能力来增强欺诈检测。
