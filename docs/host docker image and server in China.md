# 在中国托管 Docker 镜像和服务器

在中国大陆部署和托管服务时，由于网络环境和法规要求，需要采取一些特定的策略以确保服务的稳定性和合规性。本文档将指导您如何使用阿里云等中国云服务提供商来托管您的 Docker 镜像和应用服务器。

## 核心挑战

1. **网络延迟与访问性**：直接从 Docker Hub 等国外镜像仓库拉取镜像速度很慢，甚至可能失败。
2. **合规性要求**：在中国大陆提供服务需要遵守相关法律法规，例如 ICP 备案。
3. **基础设施选择**：需要选择在中国大陆有良好基础设施和支持的云服务提供商。

## 步骤一：使用阿里云容器镜像服务 (ACR)

为了解决 Docker 镜像拉取速度慢的问题，我们推荐使用阿里云的容器镜像服务（ACR）。ACR 提供了稳定、安全的国内镜像仓库。

### 1. 创建阿里云容器镜像仓库

- 登录[阿里云容器镜像服务控制台](https://cr.console.aliyun.com/)。
- 选择“实例列表”，然后创建一个新的“个人版”或“企业版”实例。个人版通常免费，适合入门。
- 在实例中，创建一个新的“命名空间”，例如 `shopguard`。
- 在该命名空间下，创建一个新的“镜像仓库”，例如 `backend-server`。

### 2. 登录到阿里云 ACR

在您的开发机器或 CI/CD 服务器上，使用 Docker CLI 登录到您的 ACR 实例。您需要在 ACR 控制台的“访问凭证”页面设置固定密码或获取临时密码。

```bash
# 将 <region> 和 <your-registry-domain> 替换为您的实际信息
docker login --username=<your-aliyun-username> registry.cn-<region>.aliyuncs.com
```

### 3. 标记并推送 Docker 镜像

构建完您的 Docker 镜像后，需要将其重新标记（tag）以匹配 ACR 的格式，然后推送到仓库。

```bash
# 1. 构建您的 Docker 镜像 (假设 Dockerfile 在当前目录)
docker build -t shopguard-backend:latest .

# 2. 标记镜像
# 格式: registry.cn-<region>.aliyuncs.com/<namespace>/<repository>:<tag>
docker tag shopguard-backend:latest registry.cn-hangzhou.aliyuncs.com/shopguard/backend-server:latest

# 3. 推送镜像到 ACR
docker push registry.cn-hangzhou.aliyuncs.com/shopguard/backend-server:latest
```

现在，您的镜像已经托管在阿里云上，可以从任何阿里云服务器上快速拉取。

## 步骤二：在阿里云 ECS 上部署服务

阿里云的弹性计算服务（ECS）是部署应用服务器的理想选择。

### 1. 创建和配置 ECS 实例

- 登录[阿里云 ECS 控制台](https://ecs.console.aliyun.com/)。
- 创建一个新的 ECS 实例。
  - **地域和可用区**：选择靠近您目标用户的地域，例如“华东1（杭州）”。
  - **操作系统**：推荐选择 Alinux、Ubuntu Server 或 CentOS。
  - **网络和安全组**：确保安全组开放了您应用所需的端口（例如 80, 443, 8000）以及 SSH 端口（22）。

### 2. SSH 连接到 ECS 实例

有关如何通过 SSH 连接到您的 ECS 实例的详细步骤，请参阅：
[如何通过 SSH 连接到阿里云 ECS](./how to ssh aliyun ECS.md)

### 3. 在 ECS 上安装 Docker

连接到 ECS 后，首先需要安装 Docker Engine。

```bash
# 更新包管理器
sudo yum update -y

# 安装 Docker
sudo yum install -y docker

# 启动并设置 Docker 开机自启
sudo systemctl start docker
sudo systemctl enable docker
```

### 4. 运行您的应用

现在，您可以从阿里云 ACR 拉取镜像并在 ECS 上运行您的服务了。

```bash
# 1. 登录到 ACR (与步骤一相同)
docker login --username=<your-aliyun-username> registry.cn-hangzhou.aliyuncs.com

# 2. 拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/shopguard/backend-server:latest

# 3. 运行容器
# -d: 后台运行
# -p: 端口映射 (将主机的 8000 端口映射到容器的 8000 端口)
# --name: 给容器命名
# --restart=always: 容器退出时总是自动重启
docker run -d -p 8000:8000 --name shopguard-app --restart=always registry.cn-hangzhou.aliyuncs.com/shopguard/backend-server:latest
```

## 步骤三：域名和 ICP 备案 (可选但推荐)

如果您的服务需要通过域名向公众提供，您需要完成以下步骤：

1. **购买域名**：通过阿里云（万网）或其他域名注册商购买一个域名。
2. **ICP 备案**：根据中国工信部的要求，所有在中国大陆境内服务器上托管的网站都必须进行 ICP 备案。您可以通过[阿里云备案系统](https://beian.aliyun.com/)提交备案申请。备案过程通常需要数天到数周。
3. **配置 DNS 解析**：备案成功后，将您的域名解析到 ECS 实例的公网 IP 地址。

完成以上步骤后，您的服务就可以在中国大陆稳定、合规地运行了。
