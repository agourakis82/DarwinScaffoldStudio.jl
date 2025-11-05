#!/bin/bash
# Darwin Omniscient Agent - Cross-repo context loader
# Version: 1.0.0
# Description: Loads context from ALL Darwin repos for AI agents

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GLOBAL_DARWIN="${HOME}/.darwin-global"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║  🧠 DARWIN OMNISCIENT AGENT - Cross-Repo Context Loader             ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Create global Darwin directory if not exists
mkdir -p "$GLOBAL_DARWIN"

echo "📊 Loading global Darwin state..."
echo ""

# Find all Darwin repos
DARWIN_REPOS=()
for repo in "$HOME/workspace"/darwin-*; do
    if [ -d "$repo" ]; then
        DARWIN_REPOS+=("$repo")
    fi
done

# Also check kec-biomaterials-scaffolds (meta-repo)
if [ -d "$HOME/workspace/kec-biomaterials-scaffolds" ]; then
    DARWIN_REPOS+=("$HOME/workspace/kec-biomaterials-scaffolds")
fi

echo "🔍 Found ${#DARWIN_REPOS[@]} Darwin repositories:"
echo ""

# Load SYNC_STATE from each repo
for repo in "${DARWIN_REPOS[@]}"; do
    repo_name=$(basename "$repo")
    echo "  📁 $repo_name"
    
    # Load SYNC_STATE
    if [ -f "$repo/SYNC_STATE.json" ]; then
        echo "    ✓ SYNC_STATE loaded"
        
        # Check for active agents
        active_agents=$(jq -r '.active_agents | length' "$repo/SYNC_STATE.json" 2>/dev/null || echo "0")
        if [ "$active_agents" -gt 0 ]; then
            echo "    ⚠️  $active_agents active agent(s)!"
        fi
        
        # Check for locks
        locks=$(jq -r '.locks | length' "$repo/SYNC_STATE.json" 2>/dev/null || echo "0")
        if [ "$locks" -gt 0 ]; then
            echo "    🔒 $locks file lock(s)"
        fi
    fi
    
    # Load EXECUTION_LOG (last 10 lines)
    if [ -f "$repo/EXECUTION_LOG.md" ]; then
        echo "    ✓ EXECUTION_LOG loaded (recent activity)"
    fi
    
    # Load Darwin cluster config
    if [ -f "$repo/.darwin/configs/.darwin-cluster.yaml" ]; then
        echo "    ☸️  Cluster config available"
    fi
    
    echo ""
done

echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Load current project info
CURRENT_PROJECT=$(basename "$PROJECT_ROOT")
echo "📍 Current project: $CURRENT_PROJECT"
echo ""

# Load cluster config
if [ -f "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" ]; then
    echo "☸️  Cluster Configuration:"
    echo ""
    
    namespace=$(grep "namespace:" "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" | head -1 | awk '{print $2}')
    provider=$(grep "provider:" "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" | head -1 | awk '{print $2}')
    
    echo "  Namespace: $namespace"
    echo "  Provider: $provider"
    echo ""
fi

# Load Darwin memory config
if [ -f "$PROJECT_ROOT/.darwin/configs/.darwin-memory-config.json" ]; then
    echo "🧠 Darwin Memory: Enabled"
    echo ""
fi

# Show available commands
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Darwin Omniscient Agent ready!"
echo ""
echo "📋 Available commands:"
echo ""
echo "  🚀 CLUSTER:"
echo "    darwin-cluster status              # Check cluster status"
echo "    darwin-cluster deploy              # Deploy to cluster"
echo "    darwin-cluster logs                # View logs"
echo "    kubectl get pods -n $namespace     # K8s pods"
echo ""
echo "  🤖 AGENTS:"
echo "    ./.darwin/agents/sync-check.sh     # Check synchronization"
echo "    ./.darwin/agents/auto-deploy.sh    # Auto deploy"
echo ""
echo "  🧠 MEMORY:"
echo "    darwin-memory search <query>       # Search Darwin RAG++"
echo "    darwin-memory save <text>          # Save to memory"
echo ""
echo "  📊 MONITORING:"
echo "    kubectl top pods -n $namespace     # Resource usage"
echo "    kubectl logs -f deployment/darwin-scaffold-studio -n $namespace"
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Save context to global Darwin
cat > "$GLOBAL_DARWIN/last-context.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "project": "$CURRENT_PROJECT",
  "repos_found": ${#DARWIN_REPOS[@]},
  "cluster_configured": $([ -f "$PROJECT_ROOT/.darwin/configs/.darwin-cluster.yaml" ] && echo "true" || echo "false")
}
EOF

echo "💾 Context saved to: $GLOBAL_DARWIN/last-context.json"
echo ""

