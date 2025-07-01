# How to Use Image Input 🖼️

This guide explains how to send images to the ShopGuard backend service for analysis.

## Supported Image Formats

The service supports multiple image input formats through the `/v1/chat/completions` endpoint:

### 1. OpenAI Vision Format (Recommended)

Send images with text using the standard OpenAI Vision API format:

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

### 2. Base64 Image Format

Send only base64 encoded images:

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

### 3. Mixed Content Format

Send multiple images with text using custom content types:

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

## Image Processing Capabilities

### OCR Text Extraction

The service can extract text from images using the `extract_text` function:

- **Supported formats**: JPEG, PNG, WebP, GIF
- **Use cases**: Screenshots, product labels, chat conversations
- **Accuracy**: High precision text recognition

### Image Understanding  

The service provides detailed image analysis using the `interpret_image` function:

- **Scene analysis**: Shopping pages, product images, conversations
- **Object detection**: Products, prices, promotional banners
- **Context understanding**: Shopping fraud detection scenarios

## cURL Examples

### Basic Image Analysis

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

### Shopping Analysis with Web Search

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

## Python Client Example

```python
import requests
import base64

def analyze_image(image_path, question):
    # Read and encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare request
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
    
    # Send request
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# Usage
result = analyze_image(
    "product_screenshot.jpg", 
    "这个商品页面是否可靠？价格是否合理？"
)
print(result['choices'][0]['message']['content'])
```

## Best Practices

### Image Quality

- **Resolution**: Higher resolution images provide better OCR accuracy
- **Format**: JPEG and PNG are preferred formats
- **Size**: Images should be under 10MB for optimal processing

### Query Optimization

- **Specific questions**: Ask targeted questions about fraud detection
- **Context**: Provide context about what type of analysis you need
- **Multiple angles**: For complex scenarios, send multiple images

### Error Handling

```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    if 'error' in result:
        print(f"API Error: {result['error']['message']}")
    else:
        print(result['choices'][0]['message']['content'])
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## Common Use Cases

### 1. Shopping Screenshot Analysis

Analyze product pages, shopping apps, or e-commerce platforms for fraud indicators.

### 2. Chat Conversation Analysis  

Examine conversation screenshots to identify suspicious communication patterns.

### 3. Price Comparison

Compare product prices in images with current market rates through web search.

### 4. Platform Verification

Verify if shopping platforms or apps shown in images are legitimate.

## Troubleshooting

### Common Issues

1. **Base64 encoding errors**
   - Ensure proper base64 encoding without newlines
   - Check image file format compatibility

2. **Image too large**
   - Compress images before sending
   - Consider resizing to reduce file size

3. **Poor OCR results**
   - Ensure image has good contrast
   - Check if text is clearly visible
   - Try different image preprocessing

### Getting Help

If you encounter issues with image processing:

1. Check the service health endpoint: `GET /v1/health`
2. Verify your image encoding is correct
3. Test with simpler images first
4. Review the error messages in the response

## Model Selection

- **vivo-BlueLM-V-2.0**: Use for multimodal (image + text) analysis
- **vivo-BlueLM-TB-Pro**: Text-only model, cannot process images

Always use `vivo-BlueLM-V-2.0` when sending images to ensure proper processing.
