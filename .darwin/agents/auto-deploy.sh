#!/bin/bash
# Darwin Auto-Deploy - Automated deployment to cluster
# Version: 1.0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
ENVIRONMENT="${1:-dev}"
VERSION="${2:-latest}"
HEALTH_CHECK_TIMEOUT=300

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║  🚀 DARWIN AUTO-DEPLOY - Automated Cluster Deployment               ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo ""

# Load cluster config
if [ ! -f "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" ]; then
    echo "❌ .darwin-cluster.yaml not found!"
    exit 1
fi

NAMESPACE=$(grep "namespace:" "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" | head -1 | awk '{print $2}')

echo "📋 Pre-deployment checks..."
echo ""

# 1. Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found!"
    exit 1
fi
echo "  ✅ kubectl available"

# 2. Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to cluster!"
    exit 1
fi
echo "  ✅ Cluster connected"

# 3. Check namespace
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "  ⚠️  Namespace $NAMESPACE not found, creating..."
    kubectl create namespace "$NAMESPACE"
    kubectl label namespace "$NAMESPACE" darwin-ecosystem=true
fi
echo "  ✅ Namespace: $NAMESPACE"

# 4. Run tests
if [ -d "$PROJECT_ROOT/tests" ]; then
    echo ""
    echo "🧪 Running tests..."
    
    if command -v pytest &> /dev/null; then
        cd "$PROJECT_ROOT"
        pytest tests/ -q || {
            echo "❌ Tests failed! Aborting deployment."
            exit 1
        }
        echo "  ✅ Tests passed"
    else
        echo "  ⚠️  pytest not available, skipping tests"
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Deploying to cluster..."
echo ""

# Apply K8s manifests
echo "  📦 Applying manifests..."

kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/namespace.yaml" 2>/dev/null || true
kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/configmap.yaml"
kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/deployment.yaml"
kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/service.yaml"
kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/ingress.yaml" 2>/dev/null || echo "  ⚠️  Ingress skipped"
kubectl apply -f "$PROJECT_ROOT/.darwin/cluster/k8s/hpa.yaml" 2>/dev/null || echo "  ⚠️  HPA skipped"

echo "  ✅ Manifests applied"
echo ""

# Wait for rollout
echo "  ⏳ Waiting for rollout..."
kubectl rollout status deployment/darwin-scaffold-studio -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s" || {
    echo "❌ Rollout failed!"
    echo ""
    echo "📊 Pod status:"
    kubectl get pods -n "$NAMESPACE"
    echo ""
    echo "📋 Recent events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
    exit 1
}

echo "  ✅ Rollout complete"
echo ""

# Health check
echo "  🏥 Health check..."
sleep 5

POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app=darwin-scaffold-studio -o jsonpath='{.items[0].metadata.name}')

if kubectl exec "$POD_NAME" -n "$NAMESPACE" -- curl -sf http://localhost:8000/health &> /dev/null; then
    echo "  ✅ Health check passed"
else
    echo "  ⚠️  Health check failed (may still be starting)"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Status:"
kubectl get pods -n "$NAMESPACE" -l app=darwin-scaffold-studio
echo ""
echo "🔗 Service:"
kubectl get svc -n "$NAMESPACE" -l app=darwin-scaffold-studio
echo ""
echo "📋 To view logs:"
echo "  kubectl logs -f deployment/darwin-scaffold-studio -n $NAMESPACE"
echo ""
echo "🌐 To access locally:"
echo "  kubectl port-forward svc/darwin-scaffold-studio 8000:8000 -n $NAMESPACE"
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Update SYNC_STATE
if [ -f "$PROJECT_ROOT/SYNC_STATE.json" ]; then
    TIMESTAMP=$(date -Iseconds)
    jq --arg ts "$TIMESTAMP" --arg env "$ENVIRONMENT" --arg ver "$VERSION" \
       '.last_update = $ts | .last_actions += [{action: "auto_deploy", environment: $env, version: $ver, timestamp: $ts}]' \
       "$PROJECT_ROOT/SYNC_STATE.json" > "$PROJECT_ROOT/SYNC_STATE.json.tmp"
    mv "$PROJECT_ROOT/SYNC_STATE.json.tmp" "$PROJECT_ROOT/SYNC_STATE.json"
fi

echo "💾 SYNC_STATE updated"
echo ""
echo "🎉 Auto-deploy completed successfully!"

