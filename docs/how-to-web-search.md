# How to Use Web Search 🔍

This guide explains how to enable and use the web search functionality in the ShopGuard backend service.

## Web Search Overview

The web search feature automatically retrieves real-time information from the internet to enhance fraud detection and provide up-to-date market data.

## Automatic Web Search Triggering

Web search is **automatically triggered** based on the content and context of your queries. The system intelligently determines when to search for additional information.

### Shopping Context Detection

The service automatically detects shopping-related queries and enables web search for:

- Price comparisons and market verification
- Platform credibility checking  
- Product availability confirmation
- Current market trends and reviews

### Search Activation Scenarios

Web search is typically triggered for:

1. **Price-related queries**: "这个价格合理吗？", "市场价格是多少？"
2. **Platform verification**: "这个网站可靠吗？", "这个平台安全吗？"
3. **Product authenticity**: "这个商品是正品吗？", "官方售价多少？"
4. **Real-time information**: "最新的诈骗手段", "当前市场行情"

## Configuration Options

### User Type Influence

Set the `user_type` parameter to influence search behavior:

```json
{
  "model": "vivo-BlueLM-TB-Pro",
  "messages": [...],
  "user_type": "学生"
}
```

**Available User Types:**

- `学生` - Student (educational focus)
- `老师` - Teacher (educational authority)
- `开发者` - Developer (technical focus)
- `普通用户` - Regular user (general consumer)

### RAG Integration

Enable RAG (Retrieval Augmented Generation) to combine web search with knowledge base:

```json
{
  "enable_rag": true,
  "rag_top_k": 3
}
```

## API Request Examples

### Basic Shopping Query with Web Search

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

### Platform Verification Request

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

### Image + Web Search Combination

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

## Available Search Engines

The service supports multiple search engines for comprehensive results:

### Primary Search Engines

1. **search_std** - Standard search (default)
2. **search_pro** - Professional search with enhanced results
3. **search_pro_sogou** - Sogou search engine
4. **search_pro_quark** - Quark search engine  
5. **search_pro_jina** - Jina search engine
6. **search_pro_bing** - Bing search engine

### Search Engine Selection

The system automatically selects the most appropriate search engine based on:

- Query type and complexity
- Regional preferences
- Search result quality
- Response time requirements

## Python Client Example

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

# Example usage
result = search_enabled_query(
    "这个iPhone价格1000元是真的吗？请帮我查一下市场价格",
    "普通用户"
)

print(result['choices'][0]['message']['content'])
```

## Search Result Processing

### Automatic Summarization

The service automatically:

- **Compresses** long search results to essential information
- **Extracts** key facts and prices
- **Summarizes** multiple sources into coherent responses
- **Filters** irrelevant or duplicate information

### Content Size Options

Search results are processed with different detail levels:

- **Small**: Brief summaries only
- **Medium**: Balanced detail (default)  
- **Large**: Comprehensive information

## Environment Configuration

Ensure these environment variables are set:

```properties
# Web Search Configuration
WEB_SEARCH_API_KEY=your_web_search_api_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Search Engine Settings
SEARCH_DEFAULT_ENGINE=search_std
SEARCH_TIMEOUT_SECONDS=10
SEARCH_DEFAULT_COUNT=4
SEARCH_CONTENT_SIZE=medium
```

## Advanced Configuration

### Search Parameters

You can influence search behavior through additional parameters:

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

### Custom Search Queries

For specific search needs, you can hint at search terms:

```json
{
  "role": "user", 
  "content": "请搜索'iPhone 15 官方价格'并分析价格合理性"
}
```

## Common Use Cases

### 1. Price Verification

Query market prices for products to detect unrealistic pricing:

```python
query = "这个MacBook Pro只要5000元，是真的吗？"
```

### 2. Platform Credibility

Verify the legitimacy of shopping platforms:

```python
query = "超级购物网这个平台可靠吗？"
```

### 3. Scam Detection

Search for known scam patterns and reports:

```python
query = "有人让我扫码领取iPhone，这是诈骗吗？"
```

### 4. Market Trend Analysis

Get current market information and trends:

```python
query = "最近有什么新的网购诈骗手段？"
```

## Best Practices

### Query Optimization

- **Be specific**: Include product names, prices, or platform names
- **Ask for comparisons**: Request market price comparisons
- **Mention verification**: Ask for credibility or legitimacy checks
- **Include context**: Provide background information

### Error Handling

```python
def handle_search_query(query):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'error' in result:
            return f"Search failed: {result['error']['message']}"
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        return "Search timed out, please try again"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
```

## Troubleshooting

### Common Issues

1. **No search results**
   - Check if web search API key is configured
   - Verify internet connectivity
   - Try different query phrasing

2. **Search timeout**
   - Increase timeout settings
   - Use more specific queries
   - Check API service status

3. **Irrelevant results**
   - Make queries more specific
   - Include context and details
   - Try different user types

### Monitoring Search Usage

Check search functionality through health endpoint:

```bash
curl http://localhost:8000/v1/health
```

Look for `web_search_available` in the response.

## Rate Limits and Quotas

Be aware of:

- **API quotas**: Web search API has usage limits
- **Rate limiting**: Automatic throttling prevents overuse  
- **Cost optimization**: Search is triggered only when necessary

## Getting Help

If web search isn't working:

1. Verify API keys are correctly configured
2. Check the service logs for search-related errors
3. Test with simple, clear queries first
4. Review your API quota usage

The web search feature enhances fraud detection by providing real-time market intelligence and verification capabilities.
