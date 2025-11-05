# Cluster Setup - Darwin Core

**Version:** 2.0.0  
**Kubernetes:** 1.24+  
**Status:** Production-ready

---

## 🎯 Prerequisites

### Required

- ✅ Kubernetes cluster (v1.24+)
- ✅ kubectl configured and connected
- ✅ Namespace permissions
- ✅ Storage class available

### Optional

- Helm 3+ (for Helm charts)
- Docker registry access (for custom images)
- Cert-manager (for TLS)

---

## 🚀 Quick Setup (10 minutes)

```bash
# 1. Clone repository
git clone https://github.com/agourakis82/darwin-scaffold-studio.git
cd darwin-scaffold-studio

# 2. Load Darwin context
./.darwin/agents/darwin-omniscient-agent.sh

# 3. Configure cluster
cp .darwin/configs/.darwin-cluster.yaml.example .darwin/configs/.darwin-cluster.yaml
# Edit namespace, resources, etc

# 4. Deploy to K8s
kubectl apply -f .darwin/cluster/k8s/

# 5. Verify deployment
kubectl get pods -n darwin-scaffold-studio
kubectl logs -f deployment/darwin-scaffold-studio -n darwin-scaffold-studio

# 6. Test service
kubectl port-forward svc/darwin-scaffold-studio 8000:8000 -n darwin-scaffold-studio
curl http://localhost:8000/health
```

---

## 📋 Detailed Setup

### Step 1: Namespace Creation

```bash
# Create namespace
kubectl create namespace darwin-scaffold-studio

# Label namespace
kubectl label namespace darwin-scaffold-studio \
  darwin-ecosystem=true \
  environment=production

# Verify
kubectl get namespace darwin-scaffold-studio --show-labels
```

---

### Step 2: Secrets Configuration

**Create `.env` file:**
```bash
# API Keys (optional, for Multi-AI)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Database passwords
QDRANT_API_KEY=...
REDIS_PASSWORD=...

# Other secrets
JWT_SECRET=...
```

**Create K8s secret:**
```bash
kubectl create secret generic darwin-scaffold-studio-secrets \
  --from-env-file=.env \
  --namespace=darwin-scaffold-studio

# Verify
kubectl get secrets -n darwin-scaffold-studio
```

---

### Step 3: ConfigMap

```bash
# Apply configmap
kubectl apply -f .darwin/cluster/k8s/configmap.yaml

# Verify
kubectl get configmap darwin-scaffold-studio-config -n darwin-scaffold-studio -o yaml
```

---

### Step 4: Deploy Services

**Deploy in order:**

```bash
# 1. Namespace
kubectl apply -f .darwin/cluster/k8s/namespace.yaml

# 2. ConfigMap
kubectl apply -f .darwin/cluster/k8s/configmap.yaml

# 3. Deployment
kubectl apply -f .darwin/cluster/k8s/deployment.yaml

# 4. Service
kubectl apply -f .darwin/cluster/k8s/service.yaml

# 5. Ingress (optional, requires ingress controller)
kubectl apply -f .darwin/cluster/k8s/ingress.yaml

# 6. HPA (optional, requires metrics-server)
kubectl apply -f .darwin/cluster/k8s/hpa.yaml
```

**Or apply all at once:**
```bash
kubectl apply -f .darwin/cluster/k8s/
```

---

### Step 5: Verify Deployment

**Check pods:**
```bash
# Watch pods starting
kubectl get pods -n darwin-scaffold-studio -w

# Should see:
# NAME                           READY   STATUS    RESTARTS   AGE
# darwin-scaffold-studio-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
# darwin-scaffold-studio-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
```

**Check logs:**
```bash
# Follow logs
kubectl logs -f deployment/darwin-scaffold-studio -n darwin-scaffold-studio

# Should see:
# 🚀 DARWIN CORE 2.0 - Starting
# ✅ Pulsar connected
# ✅ gRPC server started
# ✅ DARWIN CORE 2.0 - Ready
```

**Check service:**
```bash
kubectl get svc -n darwin-scaffold-studio

# Should see:
# NAME          TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)
# darwin-scaffold-studio   ClusterIP   10.43.x.x      <none>        8000/TCP,50051/TCP
```

---

### Step 6: Health Check

**Port-forward:**
```bash
kubectl port-forward svc/darwin-scaffold-studio 8000:8000 -n darwin-scaffold-studio
```

**Test endpoints:**
```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "2.0.0"}

# Readiness check
curl http://localhost:8000/ready
# Expected: {"status": "ready", "services": {"pulsar": true, "qdrant": true}}

# API docs
open http://localhost:8000/docs
```

---

## 🔧 Darwin Integration

### Enable Darwin Core for Other Apps

**From other namespaces:**
```yaml
env:
- name: DARWIN_CORE_ENABLED
  value: "true"
- name: DARWIN_CORE_ENDPOINT
  value: "http://darwin-scaffold-studio.darwin-scaffold-studio.svc.cluster.local:8000"
```

**From apps (Python):**
```python
import os
from darwin_core.services.graph_rag import GraphRAG

# Check if running in cluster
if os.getenv('DARWIN_CLUSTER_ENABLED') == 'true':
    # Use cluster endpoint
    endpoint = os.getenv('DARWIN_CORE_ENDPOINT')
    graphrag = GraphRAG(endpoint=endpoint)
else:
    # Use local
    graphrag = GraphRAG()
```

---

## 🤖 Darwin Agents

### Omniscient Agent

**Purpose:** Load cross-repo context

**Usage:**
```bash
./.darwin/agents/darwin-omniscient-agent.sh
```

**Output:**
- Lists all Darwin repos
- Shows active agents
- Shows file locks
- Provides available commands

### Sync Check

**Purpose:** Verify synchronization

**Usage:**
```bash
./.darwin/agents/sync-check.sh
```

**Checks:**
- SYNC_STATE.json validity
- Active agents
- File locks
- Uncommitted changes
- Unpushed commits

### Auto-Deploy

**Purpose:** Automated deployment

**Usage:**
```bash
# Deploy to dev
./.darwin/agents/auto-deploy.sh dev

# Deploy to production
./.darwin/agents/auto-deploy.sh production v2.0.0
```

**Features:**
- Pre-deployment checks
- Automated testing
- Rolling deployment
- Health verification
- Automatic rollback on failure

---

## 📊 Monitoring

### Prometheus Metrics

**Endpoint:** `http://darwin-scaffold-studio:9090/metrics`

**Key metrics:**
- `darwin_requests_total`
- `darwin_rag_query_duration_seconds`
- `darwin_ai_routing_decisions_total`
- `darwin_cache_hit_ratio`

### Grafana Dashboards

**Access:**
```bash
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

**Dashboards:**
- Darwin Core Overview
- RAG++ Performance
- Multi-AI Routing
- Resource Utilization

### Logs (Loki)

**Query logs:**
```bash
# Using kubectl
kubectl logs -f deployment/darwin-scaffold-studio -n darwin-scaffold-studio

# Using Grafana Loki
# Open Grafana → Explore → Loki
# Query: {namespace="darwin-scaffold-studio", app="darwin-scaffold-studio"}
```

---

## 🔄 Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment/darwin-scaffold-studio --replicas=5 -n darwin-scaffold-studio

# Verify
kubectl get pods -n darwin-scaffold-studio
```

### Auto-Scaling (HPA)

**Already configured in `hpa.yaml`:**
- Min: 2 replicas
- Max: 10 replicas
- Target: CPU 70%, Memory 80%

**Monitor HPA:**
```bash
kubectl get hpa -n darwin-scaffold-studio -w
```

---

## 🔧 Updates & Rollbacks

### Rolling Update

```bash
# Update image
kubectl set image deployment/darwin-scaffold-studio \
  darwin-scaffold-studio=ghcr.io/agourakis82/darwin-scaffold-studio:v2.1.0 \
  -n darwin-scaffold-studio

# Monitor rollout
kubectl rollout status deployment/darwin-scaffold-studio -n darwin-scaffold-studio

# Check history
kubectl rollout history deployment/darwin-scaffold-studio -n darwin-scaffold-studio
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/darwin-scaffold-studio -n darwin-scaffold-studio

# Rollback to specific revision
kubectl rollout undo deployment/darwin-scaffold-studio --to-revision=2 -n darwin-scaffold-studio
```

---

## 🐛 Troubleshooting

### Pod not starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n darwin-scaffold-studio

# Check events
kubectl get events -n darwin-scaffold-studio --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n darwin-scaffold-studio

# Check previous logs (if crashed)
kubectl logs <pod-name> -n darwin-scaffold-studio --previous
```

### Service not accessible

```bash
# Check endpoints
kubectl get endpoints darwin-scaffold-studio -n darwin-scaffold-studio

# Describe service
kubectl describe svc darwin-scaffold-studio -n darwin-scaffold-studio

# Test from another pod
kubectl run test-pod --rm -it --image=curlimages/curl -- sh
curl http://darwin-scaffold-studio.darwin-scaffold-studio.svc.cluster.local:8000/health
```

### Resource issues

```bash
# Check resource usage
kubectl top pods -n darwin-scaffold-studio
kubectl top nodes

# Check HPA
kubectl get hpa -n darwin-scaffold-studio

# Increase resources
kubectl edit deployment darwin-scaffold-studio -n darwin-scaffold-studio
# Edit: resources.requests/limits
```

### Persistent Volume issues

```bash
# Check PVCs
kubectl get pvc -n darwin-scaffold-studio

# Check PVs
kubectl get pv

# Describe PVC
kubectl describe pvc <pvc-name> -n darwin-scaffold-studio
```

---

## 🧪 Testing

### Local Testing (Docker Compose)

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Run tests
docker-compose exec darwin-scaffold-studio pytest tests/

# Stop
docker-compose down
```

### Cluster Testing

```bash
# Deploy to dev namespace
kubectl apply -k kubernetes/overlays/dev/

# Run cluster tests
pytest tests/cluster/

# Clean up
kubectl delete namespace darwin-scaffold-studio-dev
```

---

## 🧹 Clean Up

### Remove deployment

```bash
# Delete all resources
kubectl delete namespace darwin-scaffold-studio

# Or delete specific resources
kubectl delete -f .darwin/cluster/k8s/
```

### Remove persistent data

```bash
# List PVCs
kubectl get pvc -n darwin-scaffold-studio

# Delete PVCs
kubectl delete pvc --all -n darwin-scaffold-studio
```

---

## 📞 Support

### Issues

- GitHub Issues: https://github.com/agourakis82/darwin-scaffold-studio/issues
- Email: agourakis@agourakis.med.br

### Documentation

- Architecture: [ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
- Darwin Agents: [DARWIN_AGENTS.md](../agents/DARWIN_AGENTS.md)
- API Reference: http://localhost:8000/docs

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**

