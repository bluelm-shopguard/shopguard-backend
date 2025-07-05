# 如何使用图片输入 🖼️

本指南说明如何向 ShopGuard 后端服务发送图片以进行分析。

## 支持的图片格式

该服务通过 `/v1/chat/completions` 端点支持多种图片输入格式：

### 1. OpenAI 视觉格式（推荐）

使用标准的 OpenAI 视觉 API 格式发送图片和文本：

```json
{
  "model": "vivo-BlueLM-V-2.0",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "分析这个商品页面"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        }
      ]
    }
  ]
}
```

### 2. Base64 图片格式

仅发送 base64 编码的图片：

```json
{
  "model": "vivo-BlueLM-V-2.0", 
  "messages": [
    {
      "role": "user",
      "content": "data:image/jpeg;base64,/9j/4AAQ..."
    }
  ]
}
```

### 3. 混合内容格式

使用自定义内容类型发送多个图片和文本：

```json
{
  "model": "vivo-BlueLM-V-2.0",
  "messages": [
    {
      "role": "user",
      "contentType": "image",
      "content": "data:image/jpeg;base64,/9j/4AAQ..."
    },
    {
      "role": "user", 
      "contentType": "text",
      "content": "请分析这张图片中的商品信息"
    }
  ]
}
```

## 图片处理能力

### OCR 文字提取

该服务可以使用 `extract_text` 功能从图片中提取文字：

- **支持格式**: JPEG, PNG, WebP, GIF
- **使用场景**: 屏幕截图、产品标签、聊天对话
- **准确性**: 高精度文字识别

### 图片理解  

该服务使用 `interpret_image` 功能提供详细的图片分析：

- **场景分析**: 购物页面、产品图片、对话
- **物体检测**: 产品、价格、促销横幅
- **上下文理解**: 购物欺诈检测场景

## cURL 示例

### 基本图片分析

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
            "text": "这张图片是否有诈骗风险？"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_IMAGE_HERE"
            }
          }
        ]
      }
    ]
  }'
```

### 购物分析结合网络搜索

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
            "text": "这个商品价格合理吗？帮我查一下网上的价格对比"
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
    "enable_rag": true,
    "user_type": "普通用户"
  }'
```

## Python 客户端示例

```python
import requests
import base64

def analyze_image(image_path, question):
    # 读取并编码图片
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 准备请求
    payload = {
        "model": "vivo-BlueLM-V-2.0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "enable_rag": True,
        "user_type": "普通用户"
    }
    
    # 发送请求
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# 使用方法
result = analyze_image(
    "product_screenshot.jpg", 
    "这个商品页面是否可靠？价格是否合理？"
)
print(result['choices'][0]['message']['content'])
```

## 最佳实践

### 图片质量

- **分辨率**: 更高分辨率的图片提供更好的 OCR 准确性
- **格式**: 推荐使用 JPEG 和 PNG 格式
- **大小**: 图片应在 10MB 以下以获得最佳处理效果

### 查询优化

- **具体问题**: 针对欺诈检测提出有针对性的问题
- **上下文**: 提供需要什么类型分析的上下文
- **多角度**: 对于复杂场景，发送多张图片

### 错误处理

```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    if 'error' in result:
        print(f"API 错误: {result['error']['message']}")
    else:
        print(result['choices'][0]['message']['content'])
        
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
```

## 常见使用场景

### 1. 购物截图分析

分析产品页面、购物应用程序或电商平台的欺诈指标。

### 2. 聊天对话分析  

检查对话截图以识别可疑的沟通模式。

### 3. 价格比较

将图片中的产品价格与当前市场价格进行比较（通过网络搜索）。

### 4. 平台验证

验证图片中显示的购物平台或应用程序是否合法。

## 故障排除

### 常见问题

1. **Base64 编码错误**
   - 确保正确的 base64 编码且无换行符
   - 检查图片文件格式兼容性

2. **图片过大**
   - 发送前压缩图片
   - 考虑调整大小以减少文件大小

3. **OCR 结果不佳**
   - 确保图片有良好的对比度
   - 检查文字是否清晰可见
   - 尝试不同的图片预处理

### 获取帮助

如果遇到图片处理问题：

1. 检查服务健康端点：`GET /v1/health`
2. 验证您的图片编码是否正确
3. 先使用简单的图片进行测试
4. 查看响应中的错误消息

## 模型选择

- **vivo-BlueLM-V-2.0**: 用于多模态（图像 + 文本）分析
- **vivo-BlueLM-TB-Pro**: 仅限文本模型，无法处理图片

在发送图片时，始终使用 `vivo-BlueLM-V-2.0` 以确保正确处理。
