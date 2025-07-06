# 文档目录 📖

欢迎使用 ShopGuard 后端服务文档。此目录包含服务各个方面的详细指南。

## 快速导航

### 入门指南

- **[快速开始指南](quick-start.md)** - 5分钟快速上手
- **[配置指南](configuration-guide.md)** - 完整配置参考

### API 文档

- **[API 参考](api-reference.md)** - 完整的 API 文档
- **[如何使用图片输入](how-to-image-input.md)** - 多模态图片处理指南
- **[如何使用网络搜索](how-to-web-search.md)** - 网络搜索功能指南

### 部署

- **[生产环境部署](production-deployment.md)** - 生产环境部署指南

### 现有指南

- **[在中国部署 Docker 镜像](host%20docker%20image%20and%20server%20in%20China.md)** - 在中国的部署
- **[SSH 连接阿里云 ECS](how%20to%20ssh%20aliyun%20ECS.md)** - 阿里云服务器管理

## 文档结构

```
docs_CN/
├── README.md                           # 此目录文件
├── quick-start.md                      # 🚀 5分钟设置指南
├── api-reference.md                    # 📚 完整 API 文档
├── how-to-image-input.md              # 🖼️ 图片处理指南
├── how-to-web-search.md               # 🔍 网络搜索指南
├── configuration-guide.md             # ⚙️ 配置参考
├── production-deployment.md           # 🏭 生产环境部署
├── host docker image and server in China.md
└── how to ssh aliyun ECS.md
```

## 快速参考

### 核心功能

- **OpenAI 兼容 API** - OpenAI API 的直接替代品
- **购物欺诈检测** - 专门针对电商安全
- **多模态处理** - 文本 + 图像分析
- **检索增强生成** - 更精确的回复
- **RAG 知识库** - 10,000+ 欺诈防护案例
- **网络搜索集成** - 实时信息检索
- **流式支持** - 实时响应流

### 支持的模型

- `vivo-BlueLM-TB-Pro` - 纯文本对话模型
- `vivo-BlueLM-V-2.0` - 图像 + 文本多模态模型

### 关键端点

- `POST /v1/chat/completions` - 主要聊天 API
- `GET /v1/models` - 列出可用模型
- `GET /v1/health` - 服务健康检查

## 快速示例

### 文本查询

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vivo-BlueLM-TB-Pro",
    "messages": [{"role": "user", "content": "这个价格1000元的iPhone是真的吗？"}]
  }'
```

### 图像分析

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

## 获取帮助

### 面向开发者

1. 从[快速开始指南](quick-start.md)开始
2. 查看[API 参考](api-reference.md)了解详细使用方法
3. 探索功能专用指南以了解高级用法
