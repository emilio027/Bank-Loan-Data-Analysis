# Enterprise Credit Risk Intelligence Platform
## Deployment Guide

### Version 2.0.0 Enterprise
### Author: DevOps Documentation Team
### Date: August 2025

---

## Overview

This comprehensive deployment guide covers local development, Docker containerization, Kubernetes orchestration, and cloud deployment for the Enterprise Credit Risk Intelligence Platform. The platform supports multi-environment deployments with enterprise-grade security, scalability, and compliance requirements.

## Prerequisites

### System Requirements

**Minimum Requirements**:
- CPU: 8 cores (Intel Xeon or AMD EPYC recommended)
- RAM: 32GB (64GB recommended for production)
- Storage: 500GB SSD (1TB+ for production)
- Network: 1Gbps bandwidth
- OS: Ubuntu 20.04 LTS, RHEL 8+, or macOS 12+

**Production Requirements**:
- CPU: 16-32 cores per node
- RAM: 128GB per node
- Storage: 2TB+ NVMe SSD with RAID 10
- Network: 10Gbps with redundancy
- Load Balancer: HAProxy or AWS ALB

### Software Dependencies

- **Python**: 3.11+ with pip and virtual environments
- **Docker**: 24.0+ with Docker Compose 2.0+
- **Kubernetes**: 1.27+ (for container orchestration)
- **PostgreSQL**: 15+ (primary database)
- **Redis**: 7+ (caching and sessions)
- **Elasticsearch**: 8.8+ (search and analytics)
- **Git**: 2.40+ (version control)

## Local Development Setup

### 1. Environment Preparation

```bash
# Clone the repository
git clone https://github.com/enterprise/credit-risk-intelligence.git
cd credit-risk-intelligence

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
-- PostgreSQL commands
CREATE DATABASE credit_risk_intelligence;
CREATE USER creditrisk WITH PASSWORD 'secure_password_2025';
GRANT ALL PRIVILEGES ON DATABASE credit_risk_intelligence TO creditrisk;
\q
```

### 3. Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: maxmemory 4gb
# Set: maxmemory-policy allkeys-lru

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 4. Environment Configuration

Create `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://creditrisk:secure_password_2025@localhost:5432/credit_risk_intelligence
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_DB=1
REDIS_CACHE_DB=2

# Application Settings
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key

# API Configuration
API_VERSION=v2
API_BASE_URL=http://localhost:8000/api/v2
RATE_LIMIT_PER_MINUTE=1000

# Machine Learning Configuration
ML_MODEL_PATH=./models
ML_FEATURE_STORE_PATH=./data/features
ML_EXPERIMENT_TRACKING=true
MLFLOW_TRACKING_URI=http://localhost:5000

# Security Configuration
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret
ENCRYPTION_KEY=your-32-character-encryption-key
SSL_VERIFY=false

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true

# Third-party Integrations
POWER_BI_CLIENT_ID=your-powerbi-client-id
POWER_BI_CLIENT_SECRET=your-powerbi-secret
ELASTICSEARCH_URL=http://localhost:9200
```

### 5. Database Migration

```bash
# Initialize database schema
python -m flask db init
python -m flask db migrate -m "Initial migration"
python -m flask db upgrade

# Load sample data (optional for development)
python scripts/load_sample_data.py
```

### 6. Start Development Server

```bash
# Start the application
python -m flask run --host=0.0.0.0 --port=8000

# Or use Gunicorn for production-like testing
gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 120 app:app
```

### 7. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API test
curl -H "Content-Type: application/json" \
     -d '{"loan_amount": 100000, "credit_score": 720}' \
     http://localhost:8000/api/v2/credit-risk/score
```

## Docker Deployment

### 1. Docker Compose Setup

Review the `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  app:
    build: 
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/intelligence_platform
      - REDIS_URL=redis://redis:6379
      - FLASK_ENV=production
    depends_on:
      - db
      - redis
      - elasticsearch
    volumes:
      - ./Files/data:/app/data
      - ./Files/src:/app/src
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: intelligence_platform
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --locale=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 4gb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
  prometheus_data:
  grafana_data:
```

### 2. Dockerfile Configuration

Multi-stage Dockerfile for optimized builds:

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser Files/ ./Files/
COPY --chown=appuser:appuser *.py .
COPY --chown=appuser:appuser requirements.txt .

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "100", "app:app"]
```

### 3. Build and Deploy with Docker

```bash
# Build all services
docker-compose build

# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale application instances
docker-compose up -d --scale app=3

# Stop all services
docker-compose down

# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### 4. Docker Health Checks

```bash
# Check service health
docker-compose ps

# View service logs
docker-compose logs app
docker-compose logs db
docker-compose logs redis

# Execute commands in containers
docker-compose exec app python -c "from app import db; print(db.engine.table_names())"
docker-compose exec db psql -U postgres -d intelligence_platform -c "\dt"
```

## Kubernetes Deployment

### 1. Namespace Configuration

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: credit-risk-intelligence
  labels:
    name: credit-risk-intelligence
    environment: production
```

### 2. ConfigMap for Application Settings

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: credit-risk-config
  namespace: credit-risk-intelligence
data:
  FLASK_ENV: "production"
  API_VERSION: "v2"
  RATE_LIMIT_PER_MINUTE: "1000"
  ML_MODEL_PATH: "/app/models"
  LOG_LEVEL: "INFO"
  PROMETHEUS_ENABLED: "true"
```

### 3. Secrets Configuration

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: credit-risk-secrets
  namespace: credit-risk-intelligence
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAaG9zdDpwb3J0L2RiCg==  # base64 encoded
  JWT_SECRET_KEY: eW91ci1qd3Qtc2VjcmV0LWtleQo=  # base64 encoded
  ENCRYPTION_KEY: eW91ci0zMi1jaGFyYWN0ZXItZW5jcnlwdGlvbi1rZXkK  # base64 encoded
```

### 4. PostgreSQL Deployment

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: credit-risk-intelligence
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "intelligence_platform"
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: credit-risk-intelligence
spec:
  selector:
    app: postgres
  ports:
    - protocol: TCP
      port: 5432
      targetPort: 5432
```

### 5. Application Deployment

```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-app
  namespace: credit-risk-intelligence
spec:
  replicas: 3
  selector:
    matchLabels:
      app: credit-risk-app
  template:
    metadata:
      labels:
        app: credit-risk-app
    spec:
      containers:
      - name: credit-risk-app
        image: enterprise/credit-risk-intelligence:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: credit-risk-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        envFrom:
        - configMapRef:
            name: credit-risk-config
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: data-storage
          mountPath: /app/data
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: credit-risk-service
  namespace: credit-risk-intelligence
spec:
  selector:
    app: credit-risk-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### 6. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: credit-risk-ingress
  namespace: credit-risk-intelligence
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.creditrisk.enterprise.com
    secretName: credit-risk-tls
  rules:
  - host: api.creditrisk.enterprise.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: credit-risk-service
            port:
              number: 80
```

### 7. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: credit-risk-hpa
  namespace: credit-risk-intelligence
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: credit-risk-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 8. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f app-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Verify deployment
kubectl get pods -n credit-risk-intelligence
kubectl get services -n credit-risk-intelligence
kubectl get ingress -n credit-risk-intelligence

# Check application logs
kubectl logs -f deployment/credit-risk-app -n credit-risk-intelligence

# Scale deployment
kubectl scale deployment credit-risk-app --replicas=5 -n credit-risk-intelligence
```

## Cloud Deployment

### 1. AWS Deployment with EKS

#### EKS Cluster Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name credit-risk-cluster \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

#### RDS Database Setup

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier credit-risk-db \
  --db-instance-class db.r5.2xlarge \
  --engine postgres \
  --engine-version 15.3 \
  --master-username postgres \
  --master-user-password SecurePassword123 \
  --allocated-storage 500 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-12345678 \
  --db-subnet-group-name credit-risk-subnet-group \
  --backup-retention-period 30 \
  --multi-az \
  --storage-encrypted
```

#### ElastiCache Redis Setup

```bash
# Create ElastiCache Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id credit-risk-redis \
  --cache-node-type cache.r5.xlarge \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-87654321 \
  --subnet-group-name credit-risk-cache-subnet
```

### 2. Azure Deployment with AKS

#### AKS Cluster Setup

```bash
# Create resource group
az group create --name credit-risk-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group credit-risk-rg \
  --name credit-risk-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group credit-risk-rg --name credit-risk-cluster
```

#### Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres server create \
  --resource-group credit-risk-rg \
  --name credit-risk-db-server \
  --location eastus \
  --admin-user postgres \
  --admin-password SecurePassword123 \
  --sku-name GP_Gen5_4 \
  --storage-size 512000
```

### 3. Google Cloud Deployment with GKE

#### GKE Cluster Setup

```bash
# Create GKE cluster
gcloud container clusters create credit-risk-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

#### Cloud SQL PostgreSQL

```bash
# Create Cloud SQL instance
gcloud sql instances create credit-risk-db \
  --database-version POSTGRES_15 \
  --tier db-custom-4-16384 \
  --region us-central1 \
  --storage-size 500GB \
  --storage-type SSD
```

## Monitoring and Observability

### 1. Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'credit-risk-app'
    static_configs:
      - targets: ['credit-risk-service:80']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### 2. Grafana Dashboards

Key dashboards to import:
- Application Performance Monitoring (APM)
- Infrastructure Metrics
- Model Performance Metrics
- Business KPIs Dashboard

### 3. Alerting Rules

```yaml
# alerting-rules.yaml
groups:
- name: credit-risk-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected

  - alert: ModelPerformanceDegradation
    expr: model_accuracy < 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: Model performance below threshold
```

## Security Configuration

### 1. SSL/TLS Certificate Management

```bash
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create cluster issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: devops@enterprise.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 2. Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: credit-risk-network-policy
  namespace: credit-risk-intelligence
spec:
  podSelector:
    matchLabels:
      app: credit-risk-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Backup and Disaster Recovery

### 1. Database Backup Strategy

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="intelligence_platform"

# Create backup
pg_dump -h postgres-service -U postgres -d $DB_NAME > "$BACKUP_DIR/backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/backup_$DATE.sql"

# Upload to S3
aws s3 cp "$BACKUP_DIR/backup_$DATE.sql.gz" s3://credit-risk-backups/postgres/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### 2. Kubernetes Backup with Velero

```bash
# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.7.0 \
  --bucket credit-risk-velero-backups \
  --backup-location-config region=us-west-2 \
  --snapshot-location-config region=us-west-2

# Create backup schedule
velero create schedule credit-risk-daily \
  --schedule="0 2 * * *" \
  --include-namespaces credit-risk-intelligence
```

## Performance Tuning

### 1. Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();
```

### 2. Application Performance

```python
# Gunicorn configuration (gunicorn.conf.py)
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 2
preload_app = True
```

### 3. Redis Optimization

```bash
# Redis configuration optimizations
echo "maxmemory 4gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
echo "tcp-keepalive 60" >> /etc/redis/redis.conf
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n credit-risk-intelligence

# Adjust memory limits
kubectl patch deployment credit-risk-app -n credit-risk-intelligence -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "credit-risk-app",
            "resources": {
              "limits": {
                "memory": "12Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

#### 2. Database Connection Issues

```bash
# Check database connectivity
kubectl exec -it deployment/credit-risk-app -n credit-risk-intelligence -- \
  python -c "from app import db; print(db.engine.execute('SELECT 1').scalar())"

# Check connection pool
kubectl logs deployment/credit-risk-app -n credit-risk-intelligence | grep -i "connection"
```

#### 3. Model Loading Failures

```bash
# Check model files
kubectl exec -it deployment/credit-risk-app -n credit-risk-intelligence -- \
  ls -la /app/models/

# Verify model integrity
kubectl exec -it deployment/credit-risk-app -n credit-risk-intelligence -- \
  python -c "import pickle; pickle.load(open('/app/models/ensemble_model.pkl', 'rb'))"
```

## Maintenance and Updates

### 1. Rolling Updates

```bash
# Update application image
kubectl set image deployment/credit-risk-app \
  credit-risk-app=enterprise/credit-risk-intelligence:2.1.0 \
  -n credit-risk-intelligence

# Monitor rollout
kubectl rollout status deployment/credit-risk-app -n credit-risk-intelligence

# Rollback if needed
kubectl rollout undo deployment/credit-risk-app -n credit-risk-intelligence
```

### 2. Database Migrations

```bash
# Run migrations in production
kubectl exec -it deployment/credit-risk-app -n credit-risk-intelligence -- \
  python -m flask db upgrade
```

### 3. Certificate Renewal

```bash
# Force certificate renewal
kubectl delete secret credit-risk-tls -n credit-risk-intelligence
kubectl annotate ingress credit-risk-ingress -n credit-risk-intelligence \
  cert-manager.io/force-renew="$(date +%s)"
```

This comprehensive deployment guide provides all necessary information for deploying the Enterprise Credit Risk Intelligence Platform across different environments with enterprise-grade reliability, security, and performance.