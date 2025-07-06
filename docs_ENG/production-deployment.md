# Production Deployment Guide 🚀

Guide for deploying ShopGuard backend service in production environments.

## Prerequisites

Before deploying to production:

- **Linux server** (Ubuntu 20.04+ recommended)
- **Python 3.8+** installed
- **Nginx** for reverse proxy
- **systemd** for process management
- **SSL certificate** for HTTPS
- **Domain name** configured

## Server Setup

### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nginx supervisor

# Create application user
sudo useradd -m -s /bin/bash shopguard
sudo usermod -aG sudo shopguard
```

### 2. Application Installation

```bash
# Switch to application user
sudo su - shopguard

# Clone repository
git clone https://github.com/your-org/shopguard-backend.git
cd shopguard-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install additional production packages
pip install gunicorn supervisor
```

### 3. Environment Configuration

```bash
# Create production environment file
cp .env.example .env.production

# Edit production configuration
nano .env.production
```

Production `.env` configuration:

```properties
# Production Environment Configuration

# vivo AI Platform
VIVO_APP_ID=your_production_app_id
VIVO_APP_KEY=your_production_app_key

# API Endpoints
VIVOGPT_API_URI=/vivogpt/completions
VIVOGPT_API_DOMAIN=api-ai.vivo.com.cn
MULTIMODAL_URI=/vivogpt/completions
MULTIMODAL_DOMAIN=api-ai.vivo.com.cn
RAG_API_URI=/embedding-model-api/predict/batch
RAG_API_DOMAIN=api-ai.vivo.com.cn

# Web Search (Production)
WEB_SEARCH_API_KEY=your_production_search_key
WEB_SEARCH_URL=https://open.bigmodel.cn/api/paas/v4/web_search

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
DEBUG_MODE=false
LOG_LEVEL=INFO

# Performance Settings
MAX_CONCURRENT_REQUESTS=200
REQUEST_TIMEOUT_SECONDS=60
RAG_CACHE_TTL_SECONDS=7200
CONVERSATION_HISTORY_LIMIT=50

# Security
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ORIGINS=https://your-frontend.com

# Database (if applicable)
DATABASE_URL=postgresql://user:pass@localhost/shopguard
REDIS_URL=redis://localhost:6379/0
```

## Deployment Options

### Option 1: Gunicorn + Nginx (Recommended)

#### Gunicorn Configuration

Create `gunicorn.conf.py`:

```python
# Gunicorn configuration file
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/shopguard/access.log"
errorlog = "/var/log/shopguard/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "shopguard-backend"

# Server mechanics
daemon = False
pidfile = "/var/run/shopguard/shopguard.pid"
tmp_upload_dir = None

# SSL (if terminating SSL at Gunicorn)
# keyfile = "/path/to/private.key"
# certfile = "/path/to/certificate.crt"
```

#### Systemd Service Configuration

Create `/etc/systemd/system/shopguard.service`:

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

#### Start Services

```bash
# Create log directory
sudo mkdir -p /var/log/shopguard /var/run/shopguard
sudo chown shopguard:shopguard /var/log/shopguard /var/run/shopguard

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable shopguard
sudo systemctl start shopguard

# Check status
sudo systemctl status shopguard
```

### Option 2: Docker Deployment

#### Production Dockerfile

Create `Dockerfile.prod`:

```dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Start command
CMD ["gunicorn", "newserver:app", "-c", "gunicorn.conf.py"]
```

#### Docker Compose for Production

Create `docker-compose.prod.yml`:

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

## Nginx Configuration

### Basic Nginx Configuration

Create `/etc/nginx/sites-available/shopguard`:

```nginx
upstream shopguard_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=health:10m rate=1r/s;

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json application/javascript text/css text/plain;

    # Main API endpoints
    location /v1/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://shopguard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        
        # WebSocket support (future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check endpoint
    location /v1/health {
        limit_req zone=health burst=5 nodelay;
        
        proxy_pass http://shopguard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Quick timeout for health checks
        proxy_read_timeout 10s;
        proxy_connect_timeout 5s;
    }

    # Block access to sensitive files
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

Enable the site:

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/shopguard /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

## SSL Certificate Setup

### Option 1: Let's Encrypt (Free)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### Option 2: Custom Certificate

```bash
# Create SSL directory
sudo mkdir -p /etc/ssl/private /etc/ssl/certs

# Copy your certificates
sudo cp your-domain.crt /etc/ssl/certs/
sudo cp your-domain.key /etc/ssl/private/

# Set permissions
sudo chmod 600 /etc/ssl/private/your-domain.key
sudo chmod 644 /etc/ssl/certs/your-domain.crt
```

## Monitoring and Logging

### Log Configuration

```bash
# Create log directories
sudo mkdir -p /var/log/shopguard
sudo chown shopguard:shopguard /var/log/shopguard

# Configure log rotation
sudo nano /etc/logrotate.d/shopguard
```

Logrotate configuration:

```
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

### Health Monitoring

Create monitoring script `/home/shopguard/monitor.sh`:

```bash
#!/bin/bash

HEALTH_URL="http://localhost:8000/v1/health"
LOG_FILE="/var/log/shopguard/monitor.log"

check_health() {
    local response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$response" = "200" ]; then
        echo "[$timestamp] Service healthy" >> $LOG_FILE
        return 0
    else
        echo "[$timestamp] Service unhealthy (HTTP $response)" >> $LOG_FILE
        # Send alert (customize as needed)
        # curl -X POST "https://your-alert-webhook.com" -d "Service unhealthy"
        return 1
    fi
}

check_health
```

Add to crontab:

```bash
# Check health every 5 minutes
*/5 * * * * /home/shopguard/monitor.sh
```

## Performance Optimization

### Gunicorn Workers

Calculate optimal worker count:

```bash
# Formula: (2 x CPU cores) + 1
# For 4 CPU cores: (2 x 4) + 1 = 9 workers

# Check CPU cores
nproc

# Update gunicorn.conf.py
workers = 9  # Adjust based on your CPU
```

### System Tuning

Add to `/etc/sysctl.conf`:

```properties
# Network performance
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.core.netdev_max_backlog = 5000

# File descriptor limits
fs.file-max = 65536

# Memory settings
vm.swappiness = 10
```

Apply settings:

```bash
sudo sysctl -p
```

### Resource Limits

Add to `/etc/security/limits.conf`:

```
shopguard soft nofile 65536
shopguard hard nofile 65536
shopguard soft nproc 32768
shopguard hard nproc 32768
```

## Security Hardening

### Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port if needed)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

### Fail2Ban Setup

```bash
# Install Fail2Ban
sudo apt install fail2ban

# Create Nginx jail
sudo nano /etc/fail2ban/jail.local
```

Fail2Ban configuration:

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

## Backup Strategy

### Automated Backup Script

Create `/home/shopguard/backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/home/shopguard/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/shopguard/shopguard-backend"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application and configuration
tar -czf "$BACKUP_DIR/shopguard_$DATE.tar.gz" \
    -C /home/shopguard \
    --exclude='shopguard-backend/venv' \
    --exclude='shopguard-backend/__pycache__' \
    --exclude='shopguard-backend/*.log' \
    shopguard-backend

# Backup environment files
cp $APP_DIR/.env.production "$BACKUP_DIR/env_$DATE.backup"

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "env_*.backup" -mtime +30 -delete

echo "Backup completed: shopguard_$DATE.tar.gz"
```

Schedule backup:

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * /home/shopguard/backup.sh
```

## Deployment Checklist

### Pre-deployment

- [ ] Server provisioned and configured
- [ ] Domain DNS configured
- [ ] SSL certificate obtained
- [ ] Environment variables configured
- [ ] Knowledge base files uploaded
- [ ] Dependencies installed

### Deployment

- [ ] Application deployed
- [ ] Services configured and started
- [ ] Nginx configured and reloaded
- [ ] SSL certificate installed
- [ ] Firewall rules applied

### Post-deployment

- [ ] Health check passing
- [ ] API endpoints responding
- [ ] SSL certificate valid
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Documentation updated

## Troubleshooting

### Common Issues

1. **Service won't start**

   ```bash
   # Check logs
   sudo journalctl -u shopguard -f
   
   # Check configuration
   sudo systemctl status shopguard
   ```

2. **High memory usage**

   ```bash
   # Monitor memory
   htop
   
   # Reduce Gunicorn workers if needed
   # Edit gunicorn.conf.py
   ```

3. **SSL certificate issues**

   ```bash
   # Test SSL
   sudo nginx -t
   
   # Check certificate
   openssl x509 -in /etc/ssl/certs/your-domain.crt -text -noout
   ```

### Emergency Recovery

```bash
# Stop all services
sudo systemctl stop shopguard nginx

# Restore from backup
cd /home/shopguard
tar -xzf backups/shopguard_YYYYMMDD_HHMMSS.tar.gz

# Restart services
sudo systemctl start shopguard nginx
```

## Maintenance

### Regular Tasks

1. **Weekly**: Check logs and system resources
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Review and update SSL certificates
4. **Semi-annually**: Performance testing and optimization

### Update Procedure

```bash
# Switch to application user
sudo su - shopguard

# Backup current version
./backup.sh

# Pull latest changes
cd shopguard-backend
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl restart shopguard

# Verify deployment
curl http://localhost:8000/v1/health
```

This production deployment guide ensures a secure, scalable, and maintainable ShopGuard backend service deployment.
