#!/bin/bash
# Darwin Sync Check - Verify repository synchronization
# Version: 1.0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GLOBAL_SYNC="${HOME}/.darwin-sync"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║  🔄 DARWIN SYNC CHECK - Repository Synchronization                  ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Create sync directory if not exists
mkdir -p "$GLOBAL_SYNC"

# Check SYNC_STATE.json
if [ ! -f "$PROJECT_ROOT/SYNC_STATE.json" ]; then
    echo "❌ SYNC_STATE.json not found!"
    echo "   Creating template..."
    
    cat > "$PROJECT_ROOT/SYNC_STATE.json" << 'EOF'
{
  "last_update": "",
  "active_agents": [],
  "locks": {},
  "last_actions": [],
  "version": "1.0.0"
}
EOF
    
    echo "✅ SYNC_STATE.json created"
    echo ""
fi

# Read SYNC_STATE
echo "📊 Synchronization Status:"
echo ""

# Check active agents
active_agents=$(jq -r '.active_agents | length' "$PROJECT_ROOT/SYNC_STATE.json" 2>/dev/null || echo "0")
if [ "$active_agents" -gt 0 ]; then
    echo "  ⚠️  Active agents: $active_agents"
    jq -r '.active_agents[] | "    - \(.agent_id) (since \(.timestamp))"' "$PROJECT_ROOT/SYNC_STATE.json"
else
    echo "  ✅ No active agents"
fi
echo ""

# Check locks
locks=$(jq -r '.locks | length' "$PROJECT_ROOT/SYNC_STATE.json" 2>/dev/null || echo "0")
if [ "$locks" -gt 0 ]; then
    echo "  🔒 File locks: $locks"
    jq -r '.locks | to_entries[] | "    - \(.key): \(.value.agent_id) (since \(.value.timestamp))"' "$PROJECT_ROOT/SYNC_STATE.json"
    echo ""
    echo "  ⚠️  Files are locked! Check if other agents are active."
else
    echo "  ✅ No file locks"
fi
echo ""

# Check last update
last_update=$(jq -r '.last_update' "$PROJECT_ROOT/SYNC_STATE.json" 2>/dev/null || echo "never")
echo "  📅 Last update: $last_update"
echo ""

# Check for pending changes
if command -v git &> /dev/null; then
    cd "$PROJECT_ROOT"
    
    if [ -d .git ]; then
        uncommitted=$(git status --porcelain | wc -l)
        if [ "$uncommitted" -gt 0 ]; then
            echo "  ⚠️  Uncommitted changes: $uncommitted files"
        else
            echo "  ✅ No uncommitted changes"
        fi
        
        unpushed=$(git log @{u}.. --oneline 2>/dev/null | wc -l || echo "0")
        if [ "$unpushed" -gt 0 ]; then
            echo "  ⚠️  Unpushed commits: $unpushed"
        else
            echo "  ✅ All commits pushed"
        fi
    fi
fi
echo ""

# Sync with global
echo "🔄 Syncing with global Darwin state..."
cp "$PROJECT_ROOT/SYNC_STATE.json" "$GLOBAL_SYNC/$(basename $PROJECT_ROOT)_SYNC_STATE.json" 2>/dev/null || true
echo "  ✅ Synced to: $GLOBAL_SYNC/"
echo ""

echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Sync check complete!"
echo ""

# Exit with warning if locks exist
if [ "$locks" -gt 0 ] || [ "$active_agents" -gt 0 ]; then
    echo "⚠️  WARNING: Other agents active or files locked!"
    echo "   Coordinate before making changes."
    echo ""
    exit 1
fi

exit 0

