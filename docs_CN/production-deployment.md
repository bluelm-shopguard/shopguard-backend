# 生产环境部署指南 🚀

在生产环境中部署 ShopGuard 后端服务的指南。

## 前提条件

部署到生产环境前：

- **Linux 服务器** (推荐 Ubuntu 20.04+)
- **Python 3.8+** 已安装
- **Nginx** 用于反向代理
- **systemd** 用于进程管理
- **SSL 证书** 用于 HTTPS
- **域名** 已配置

## 服务器设置

### 1. 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必需的包
sudo apt install -y python3 python3-pip python3-venv nginx supervisor

# 创建应用程序用户
sudo useradd -m -s /bin/bash shopguard
sudo usermod -aG sudo shopguard
```

### 2. 应用程序安装

```bash
# 切换到应用程序用户
sudo su - shopguard

# 克隆仓库
git clone https://github.com/your-org/shopguard-backend.git
cd shopguard-backend

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装额外的生产包
pip install gunicorn supervisor
```

### 3. 环境配置

```bash
# 创建生产环境文件
cp .env.example .env.production

# 编辑生产配置
nano .env.production
```

生产环境 `.env` 配置：

```properties
# 生产环境配置

# vivo AI 平台
VIVO_APP_ID=your_production_app_id
VIVO_APP_KEY=your_production_app_key

# API 端点
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# 网络搜索（生产）
WEB_SEARCH_API_KEY=your_production_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# 服务器配置
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO

# 性能设置
MAX_CONCURRENT_REQUESTS=200
REQUEST_TIMEOUT_SECONDS=60
RAG_CACHE_TTL_SECONDS=7200
CONVERSATION_HISTORY_LIMIT=50

# 安全
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ORIGINS=https://your-frontend.com

# 数据库（如果适用）
DATABASE_URL=postgresql://user:pass@localhost/shopguard
REDIS_URL=redis://localhost:6379/0
```

## 部署选项

### 选项 1: Gunicorn + Nginx（推荐）

#### Gunicorn 配置

创建 `gunicorn.conf.py`：

```python
# Gunicorn 配置文件
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 120
keepalive = 5

# 日志
accesslog = "/var/log/shopguard/access.log"
errorlog = "/var/log/shopguard/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 进程命名
proc_name = "shopguard-backend"

# 服务器机制
daemon = False
pidfile = "/var/run/shopguard/shopguard.pid"
tmp_upload_dir = None

# SSL（如果在 Gunicorn 处终止 SSL）
# keyfile = "/path/to/private.key"
# certfile = "/path/to/certificate.crt"
```

#### Systemd 服务配置

创建 `/etc/systemd/system/shopguard.service`：

```ini
[Unit]
Description=ShopGuard Backend API Server
After=network.target

[Service]
Type=notify
User=shopguard
Group=shopguard
RuntimeDirectory=shopguard
WorkingDirectory=/home/shopguard/shopguard-backend
Environment=PATH=/home/shopguard/shopguard-backend/venv/bin
EnvironmentFile=/home/shopguard/shopguard-backend/.env.production
ExecStart=/home/shopguard/shopguard-backend/venv/bin/gunicorn newserver:app -c gunicorn.conf.py
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 启动服务

```bash
# 创建日志目录
sudo mkdir -p /var/log/shopguard /var/run/shopguard
sudo chown shopguard:shopguard /var/log/shopguard /var/run/shopguard

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable shopguard
sudo systemctl start shopguard

# 检查状态
sudo systemctl status shopguard
```

### 选项 2: Docker 部署

#### 生产 Dockerfile

创建 `Dockerfile.prod`：

```dockerfile
FROM python:3.9-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建应用程序用户
RUN useradd --create-home --shell /bin/bash app

# 设置工作目录
WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# 复制应用程序代码
COPY . .

# 更改所有权给应用程序用户
RUN chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# 启动命令
CMD ["gunicorn", "newserver:app", "-c", "gunicorn.conf.py"]
```

#### 生产环境 Docker Compose

创建 `docker-compose.prod.yml`：

```yaml
version: '3.8'

services:
  shopguard-backend:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: shopguard-backend
    restart: unless-stopped
    env_file:
      - .env.production
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./knowledge_base_embeddings:/app/knowledge_base_embeddings:ro
    depends_on:
      - redis
    networks:
      - shopguard-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  redis:
    image: redis:7-alpine
    container_name: shopguard-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - shopguard-network

  nginx:
    image: nginx:alpine
    container_name: shopguard-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - shopguard-backend
    networks:
      - shopguard-network

volumes:
  redis_data:

networks:
  shopguard-network:
    driver: bridge
```

## Nginx 配置

### 基本 Nginx 配置

创建 `/etc/nginx/sites-available/shopguard`：

```nginx
upstream shopguard_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# 速率限制
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=health:10m rate=1r/s;

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # 重定向 HTTP 到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL 配置
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # 现代 SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # Gzip 压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json application/javascript text/css text/plain;

    # 主要 API 端点
    location /v1/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://shopguard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 流支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        
        # WebSocket 支持（未来）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 健康检查端点
    location /v1/health {
        limit_req zone=health burst=5 nodelay;
        
        proxy_pass http://shopguard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 健康检查快速超时
        proxy_read_timeout 10s;
        proxy_connect_timeout 5s;
    }

    # 阻止访问敏感文件
    location ~ /\. {
        deny all;
    }
    
    location ~ \.(env|ini|conf)$ {
        deny all;
    }

    # Robots.txt
    location = /robots.txt {
        add_header Content-Type text/plain;
        return 200 "User-agent: *\nDisallow: /\n";
    }
}
```

启用网站：

```bash
# 启用网站
sudo ln -s /etc/nginx/sites-available/shopguard /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx
```

## SSL 证书设置

### 选项 1: Let's Encrypt（免费）

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# 测试自动续期
sudo certbot renew --dry-run
```

### 选项 2: 自定义证书

```bash
# 创建 SSL 目录
sudo mkdir -p /etc/ssl/private /etc/ssl/certs

# 复制您的证书
sudo cp your-domain.crt /etc/ssl/certs/
sudo cp your-domain.key /etc/ssl/private/

# 设置权限
sudo chmod 600 /etc/ssl/private/your-domain.key
sudo chmod 644 /etc/ssl/certs/your-domain.crt
```

## 监控和日志

### 日志配置

```bash
# 创建日志目录
sudo mkdir -p /var/log/shopguard
sudo chown shopguard:shopguard /var/log/shopguard

# 配置日志轮转
sudo nano /etc/logrotate.d/shopguard
```

日志轮转配置：

```text
/var/log/shopguard/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 shopguard shopguard
    postrotate
        systemctl reload shopguard
    endscript
}
```

### 健康监控

创建监控脚本 `/home/shopguard/monitor.sh`：

```bash
#!/bin/bash

HEALTH_URL="http://localhost:8000/v1/health"
LOG_FILE="/var/log/shopguard/monitor.log"

check_health() {
    local response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$response" = "200" ]; then
        echo "[$timestamp] 服务健康" >> $LOG_FILE
        return 0
    else
        echo "[$timestamp] 服务不健康 (HTTP $response)" >> $LOG_FILE
        # 发送警报（根据需要定制）
        # curl -X POST "https://your-alert-webhook.com" -d "服务不健康"
        return 1
    fi
}

check_health
```

添加到 crontab：

```bash
# 每5分钟检查健康状态
*/5 * * * * /home/shopguard/monitor.sh
```

## 性能优化

### Gunicorn Workers

计算最佳工作进程数：

```bash
# 公式: (2 x CPU 核心数) + 1
# 对于 4 个 CPU 核心: (2 x 4) + 1 = 9 个工作进程

# 检查 CPU 核心数
nproc

# 更新 gunicorn.conf.py
workers = 9  # 根据您的 CPU 调整
```

### 系统调优

添加到 `/etc/sysctl.conf`：

```properties
# 网络性能
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.core.netdev_max_backlog = 5000

# 文件描述符限制
fs.file-max = 65536

# 内存设置
vm.swappiness = 10
```

应用设置：

```bash
sudo sysctl -p
```

### 资源限制

添加到 `/etc/security/limits.conf`：

```text
shopguard soft nofile 65536
shopguard hard nofile 65536
shopguard soft nproc 32768
shopguard hard nproc 32768
```

## 安全加固

### 防火墙配置

```bash
# 安装 UFW
sudo apt install ufw

# 默认策略
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许 SSH（如果需要调整端口）
sudo ufw allow 22/tcp

# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 启用防火墙
sudo ufw enable
```

### Fail2Ban 设置

```bash
# 安装 Fail2Ban
sudo apt install fail2ban

# 创建 Nginx 监狱
sudo nano /etc/fail2ban/jail.local
```

Fail2Ban 配置：

```ini
[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
findtime = 600
bantime = 3600
```

## 备份策略

### 自动备份脚本

创建 `/home/shopguard/backup.sh`：

```bash
#!/bin/bash

BACKUP_DIR="/home/shopguard/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/shopguard/shopguard-backend"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份应用程序和配置
tar -czf "$BACKUP_DIR/shopguard_$DATE.tar.gz" \
    -C /home/shopguard \
    --exclude='shopguard-backend/venv' \
    --exclude='shopguard-backend/__pycache__' \
    --exclude='shopguard-backend/*.log' \
    shopguard-backend

# 备份环境文件
cp $APP_DIR/.env.production "$BACKUP_DIR/env_$DATE.backup"

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "env_*.backup" -mtime +30 -delete

echo "备份完成: shopguard_$DATE.tar.gz"
```

计划备份：

```bash
# 添加到 crontab（每天凌晨2点）
0 2 * * * /home/shopguard/backup.sh
```

## 部署检查清单

### 部署前

- [ ] 服务器已配置和设置
- [ ] 域名 DNS 已配置
- [ ] SSL 证书已获取
- [ ] 环境变量已配置
- [ ] 知识库文件已上传
- [ ] 依赖项已安装

### 部署

- [ ] 应用程序已部署
- [ ] 服务已配置并启动
- [ ] Nginx 已配置并重载
- [ ] SSL 证书已安装
- [ ] 防火墙规则已应用

### 部署后

- [ ] 健康检查通过
- [ ] API 端点响应正常
- [ ] SSL 证书有效
- [ ] 监控已配置
- [ ] 备份策略已实施
- [ ] 文档已更新

## 故障排除

### 常见问题

1. **服务无法启动**

   ```bash
   # 检查日志
   sudo journalctl -u shopguard -f
   
   # 检查配置
   sudo systemctl status shopguard
   ```

2. **内存使用过高**

   ```bash
   # 监控内存
   htop
   
   # 如果需要可以减少 Gunicorn workers
   # 编辑 gunicorn.conf.py
   ```

3. **SSL 证书问题**

   ```bash
   # 测试 SSL
   sudo nginx -t
   
   # 检查证书
   openssl x509 -in /etc/ssl/certs/your-domain.crt -text -noout
   ```

### 紧急恢复

```bash
# 停止所有服务
sudo systemctl stop shopguard nginx

# 从备份恢复
cd /home/shopguard
tar -xzf backups/shopguard_YYYYMMDD_HHMMSS.tar.gz

# 重启服务
sudo systemctl start shopguard nginx
```

## 维护

### 定期任务

1. **每周**: 检查日志和系统资源
2. **每月**: 更新依赖项和安全补丁
3. **每季度**: 审查和更新 SSL 证书
4. **每半年**: 性能测试和优化

### 更新程序

```bash
# 切换到应用程序用户
sudo su - shopguard

# 备份当前版本
./backup.sh

# 拉取最新更改
cd shopguard-backend
git pull origin main

# 更新依赖项
source venv/bin/activate
pip install -r requirements.txt

# 重启服务
sudo systemctl restart shopguard

# 验证部署
curl http://localhost:8000/v1/health
```

此生产环境部署指南确保了 ShopGuard 后端服务的安全、可扩展和可维护的部署。
