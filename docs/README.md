# Documentation Index 📖

Welcome to the ShopGuard Backend documentation. This directory contains focused guides for different aspects of the service.

## Quick Navigation

### Getting Started

- **[Quick Start Guide](quick-start.md)** - Get up and running in 5 minutes
- **[Configuration Guide](configuration-guide.md)** - Complete configuration reference

### API Documentation

- **[API Reference](api-reference.md)** - Complete API documentation
- **[How to Use Image Input](how-to-image-input.md)** - Multimodal image processing guide
- **[How to Use Web Search](how-to-web-search.md)** - Web search functionality guide

### Deployment

- **[Production Deployment](production-deployment.md)** - Production deployment guide

### Existing Guides

- **[Host Docker Image in China](host%20docker%20image%20and%20server%20in%20China.md)** - Deployment in China
- **[SSH to Aliyun ECS](how%20to%20ssh%20aliyun%20ECS.md)** - Aliyun server management

## Documentation Structure

```
docs/
├── README.md                           # This index file
├── quick-start.md                      # 🚀 5-minute setup guide
├── api-reference.md                    # 📚 Complete API docs
├── how-to-image-input.md              # 🖼️ Image processing guide
├── how-to-web-search.md               # 🔍 Web search guide
├── configuration-guide.md             # ⚙️ Configuration reference
├── production-deployment.md           # 🏭 Production deployment
├── host docker image and server in China.md
└── how to ssh aliyun ECS.md
```

## Quick Reference

### Core Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **Shopping Fraud Detection** - Specialized for e-commerce security
- **Multimodal Processing** - Text + image analysis
- **RAG Knowledge Base** - 10,000+ fraud prevention examples
- **Web Search Integration** - Real-time information retrieval
- **Streaming Support** - Real-time response streaming

### Supported Models

- `vivo-BlueLM-TB-Pro` - Text-only model for conversations
- `vivo-BlueLM-V-2.0` - Multimodal model for image + text

### Key Endpoints

- `POST /v1/chat/completions` - Main chat API
- `GET /v1/models` - List available models
- `GET /v1/health` - Service health check

## Quick Examples

### Text Query

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [{"role": "user", "content": "这个价格1000元的iPhone是真的吗？"}]
  }'
```

### Image Analysis

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-V-2.0",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "分析这个购物页面"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

## Getting Help

### For Developers

1. Start with [Quick Start Guide](quick-start.md)
2. Check [API Reference](api-reference.md) for detailed usage
3. Explore feature-specific guides for advanced usage

### For DevOps

1. Follow [Production Deployment](production-deployment.md)
2. Reference [Configuration Guide](configuration-guide.md) for tuning
3. Check existing China deployment guides for regional setup

### For Integration

1. Review [API Reference](api-reference.md) for request/response formats
2. See [Image Input Guide](how-to-image-input.md) for multimodal features
3. Check [Web Search Guide](how-to-web-search.md) for enhanced capabilities

## Documentation Status

| Guide | Status | Last Updated |
|-------|--------|--------------|
| Quick Start | ✅ Complete | 2025-01-01 |
| API Reference | ✅ Complete | 2025-01-01 |
| Image Input | ✅ Complete | 2025-01-01 |
| Web Search | ✅ Complete | 2025-01-01 |
| Configuration | ✅ Complete | 2025-01-01 |
| Production Deployment | ✅ Complete | 2025-01-01 |
| China Deployment | ✅ Existing | Earlier |
| SSH Guide | ✅ Existing | Earlier |

## Contributing to Documentation

To improve the documentation:

1. **For new features**: Create focused how-to guides
2. **For clarifications**: Update existing guides
3. **For examples**: Add to relevant sections
4. **For corrections**: Submit pull requests

### Documentation Standards

- Use clear, actionable titles
- Include working code examples
- Provide troubleshooting sections
- Follow markdown best practices
- Test all commands and examples

## Feedback

Found an issue or have a suggestion? Please:

- Open a GitHub issue
- Submit a pull request
- Contact the maintainers

---

**Last Updated**: January 1, 2025  
**Documentation Version**: 1.0.0
