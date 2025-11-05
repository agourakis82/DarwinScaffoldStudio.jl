# Darwin Agents Integration

**Version:** 1.0.0  
**Type:** Multi-Agent System  
**Status:** Production

---

## 🎯 Overview

Darwin ecosystem provides AI agents for development automation, context management, and cross-repo synchronization.

**Philosophy:** AI agents help AI agents (and humans!)

---

## 🤖 Available Agents

### 1. Darwin Omniscient Agent 🧠

**Purpose:** Load cross-repo context automatically for AI assistants (ChatGPT, Claude, Cursor)

**Location:** `.darwin/agents/darwin-omniscient-agent.sh`

**Usage:**
```bash
# Run before ANY AI interaction
./.darwin/agents/darwin-omniscient-agent.sh
```

**What it does:**
1. Scans ALL Darwin repos in `~/workspace/`
2. Loads SYNC_STATE.json from each repo
3. Checks for active agents and file locks
4. Loads cluster configurations
5. Provides available commands

**Output:**
```
🧠 DARWIN OMNISCIENT AGENT - Cross-Repo Context Loader

📊 Loading global Darwin state...

🔍 Found 4 Darwin repositories:

  📁 darwin-scaffold-studio
    ✓ SYNC_STATE loaded
    ☸️  Cluster config available

  📁 darwin-scaffold-studio
    ✓ SYNC_STATE loaded
    ⚠️  1 active agent(s)!
    🔒 2 file lock(s)

  📁 darwin-pbpk-platform
    ✓ SYNC_STATE loaded

  📁 kec-biomaterials-scaffolds
    ✓ SYNC_STATE loaded
    ✓ EXECUTION_LOG loaded

══════════════════════════════════════════════════════════════════════

📍 Current project: darwin-scaffold-studio

☸️  Cluster Configuration:
  Namespace: darwin-scaffold-studio
  Provider: k3s

✅ Darwin Omniscient Agent ready!

📋 Available commands:
  🚀 CLUSTER:
    darwin-cluster status
    darwin-cluster deploy
    kubectl get pods -n darwin-scaffold-studio

  🤖 AGENTS:
    ./.darwin/agents/sync-check.sh
    ./.darwin/agents/auto-deploy.sh

  🧠 MEMORY:
    darwin-memory search <query>
    darwin-memory save <text>
```

**When to use:**
- ✅ Before starting ANY development session
- ✅ Before AI interactions (Cursor, ChatGPT, Claude)
- ✅ After switching repos
- ✅ After long breaks

---

### 2. Sync Check Agent 🔄

**Purpose:** Verify repository synchronization and detect conflicts

**Location:** `.darwin/agents/sync-check.sh`

**Usage:**
```bash
./.darwin/agents/sync-check.sh
```

**What it checks:**
- SYNC_STATE.json exists and is valid
- Active agents (who is working?)
- File locks (what's being edited?)
- Last update timestamp
- Uncommitted changes (git status)
- Unpushed commits (git log)

**Output Example:**
```
🔄 DARWIN SYNC CHECK - Repository Synchronization

📊 Synchronization Status:

  ✅ No active agents
  ✅ No file locks
  📅 Last update: 2025-11-05T19:30:00-03:00
  ✅ No uncommitted changes
  ✅ All commits pushed

🔄 Syncing with global Darwin state...
  ✅ Synced to: /home/agourakis82/.darwin-sync/

✅ Sync check complete!
```

**Exit codes:**
- `0`: All clear, safe to work
- `1`: Conflicts detected (locks or active agents)

**When to use:**
- ✅ Before making significant changes
- ✅ If working with multiple AI agents
- ✅ After seeing unusual behavior
- ✅ Before committing changes

---

### 3. Auto-Deploy Agent 🚀

**Purpose:** Automated deployment to Kubernetes cluster

**Location:** `.darwin/agents/auto-deploy.sh`

**Usage:**
```bash
# Deploy to dev
./.darwin/agents/auto-deploy.sh dev

# Deploy to staging
./.darwin/agents/auto-deploy.sh staging

# Deploy to production with specific version
./.darwin/agents/auto-deploy.sh production v2.0.0
```

**What it does:**
1. **Pre-deployment checks:**
   - kubectl available
   - Cluster connected
   - Namespace exists (creates if not)
   - Tests pass (pytest)

2. **Deployment:**
   - Applies K8s manifests
   - Waits for rollout
   - Performs health check

3. **Post-deployment:**
   - Shows pod status
   - Shows service info
   - Provides access commands
   - Updates SYNC_STATE.json

**Output:**
```
🚀 DARWIN AUTO-DEPLOY - Automated Cluster Deployment

Environment: production
Version: v2.0.0

📋 Pre-deployment checks...
  ✅ kubectl available
  ✅ Cluster connected
  ✅ Namespace: darwin-scaffold-studio

🧪 Running tests...
  ✅ Tests passed (42 passed in 5.2s)

══════════════════════════════════════════════════════════════════════

🚀 Deploying to cluster...
  📦 Applying manifests...
  ✅ Manifests applied

  ⏳ Waiting for rollout...
  ✅ Rollout complete

  🏥 Health check...
  ✅ Health check passed

══════════════════════════════════════════════════════════════════════

✅ Deployment complete!

📊 Status:
NAME                          READY   STATUS    AGE
darwin-scaffold-studio-xxxxxxxxx-xxxxx   1/1     Running   30s
darwin-scaffold-studio-xxxxxxxxx-xxxxx   1/1     Running   30s

🔗 Service:
NAME          TYPE        CLUSTER-IP    PORT(S)
darwin-scaffold-studio   ClusterIP   10.43.x.x     8000,50051,9090

📋 To view logs:
  kubectl logs -f deployment/darwin-scaffold-studio -n darwin-scaffold-studio

🌐 To access locally:
  kubectl port-forward svc/darwin-scaffold-studio 8000:8000 -n darwin-scaffold-studio

══════════════════════════════════════════════════════════════════════

🎉 Auto-deploy completed successfully!
```

**When to use:**
- ✅ After code changes
- ✅ For version releases
- ✅ Instead of manual kubectl commands
- ✅ For consistent deployments

---

## 📋 Agent Workflows

### Development Workflow

```bash
# 1. Start session - Load context
./.darwin/agents/darwin-omniscient-agent.sh

# 2. Check synchronization
./.darwin/agents/sync-check.sh

# 3. Make changes
# ... your development work ...

# 4. Test locally
pytest tests/
docker-compose up

# 5. Deploy to dev cluster
./.darwin/agents/auto-deploy.sh dev

# 6. Verify
kubectl logs -f deployment/darwin-scaffold-studio -n darwin-scaffold-studio-dev

# 7. If OK, deploy to production
./.darwin/agents/auto-deploy.sh production v2.1.0
```

### Multi-Agent Collaboration

**Scenario:** 2 AI agents working on different repos

**Agent 1 (Cursor on darwin-scaffold-studio):**
```bash
# Load context
./.darwin/agents/darwin-omniscient-agent.sh
# Sees: Agent 2 active on darwin-scaffold-studio

# Check sync
./.darwin/agents/sync-check.sh
# Status: scaffold_optimizer.py locked by Agent 2

# Coordinate: Work on different files!
```

**Agent 2 (ChatGPT on darwin-scaffold-studio):**
```bash
# Load context
./.darwin/agents/darwin-omniscient-agent.sh
# Sees: Agent 1 active on darwin-scaffold-studio

# Agent 2 updates SYNC_STATE when done
# Agent 1 can now sync and see changes
```

---

## ⚙️ Agent Configuration

### `.darwin/configs/agent-config.yaml`

```yaml
agents:
  omniscient:
    enabled: true
    repos:
      - darwin-scaffold-studio
      - darwin-scaffold-studio
      - darwin-pbpk-platform
      - kec-biomaterials-scaffolds
    memory:
      enabled: true
      backend: qdrant
      endpoint: http://qdrant.darwin-scaffold-studio.svc.cluster.local:6333
  
  sync:
    enabled: true
    interval: 300  # seconds
    conflict_resolution: manual  # manual | auto
    auto_lock_timeout: 1800  # 30 minutes
  
  auto_deploy:
    enabled: true
    environments:
      dev:
        cluster: k3s-dev
        namespace: darwin-scaffold-studio-dev
        health_check_timeout: 180
      
      staging:
        cluster: k3s-staging
        namespace: darwin-scaffold-studio-staging
        health_check_timeout: 300
      
      production:
        cluster: k8s-prod
        namespace: darwin-scaffold-studio
        health_check_timeout: 600
        require_tests: true
        require_approval: false  # Set true for manual approval

logging:
  level: INFO
  file: .darwin/logs/agents.log
  format: json

notifications:
  enabled: false
  slack_webhook: ""
  email: ""
```

---

## 🧠 Integration with Darwin RAG++

### Agent Memory Search

**Purpose:** AI agents can search Darwin RAG++ for context

**Example (Cursor AI):**
```python
# In Cursor, after running omniscient agent
from darwin_core.services.graph_rag import GraphRAG

# Search for deployment info
graphrag = GraphRAG()
answer = graphrag.query(
    "How do I deploy Darwin Core to production cluster?",
    query_type="local"
)

print(answer)
# Returns: Context from CLUSTER_SETUP.md, deployment.yaml, etc
```

### Agent Context Injection

**Purpose:** Inject cross-repo context into AI prompts

**Example:**
```python
from darwin_core.multi_ai.router import MultiAIHub

# Load agent context
agent_context = load_omniscient_context()

# Include in AI request
request = ChatRequest(
    message="How to scale darwin-scaffold-studio?",
    domain=ScientificDomain.DEVOPS,
    context=agent_context  # Includes all repo info!
)

hub = MultiAIHub()
response = await hub.chat_with_routing(request)

# Response is informed by ALL Darwin repos!
```

---

## 🔗 Cross-Repo Communication

### SYNC_STATE.json Format

```json
{
  "last_update": "2025-11-05T19:30:00-03:00",
  "active_agents": [
    {
      "agent_id": "cursor_agent_1",
      "timestamp": "2025-11-05T19:25:00-03:00",
      "repo": "darwin-scaffold-studio"
    }
  ],
  "locks": {
    "app/services/graph_rag.py": {
      "agent_id": "cursor_agent_1",
      "timestamp": "2025-11-05T19:26:00-03:00",
      "expires": "2025-11-05T19:56:00-03:00"
    }
  },
  "last_actions": [
    {
      "action": "auto_deploy",
      "environment": "production",
      "version": "v2.0.0",
      "timestamp": "2025-11-05T19:30:00-03:00"
    }
  ],
  "version": "1.0.0"
}
```

---

## 🛠️ Best Practices

### 1. Always Run Omniscient Agent First

**Why:** Loads cross-repo context for AI

**When:**
- Starting development session
- Switching repos
- Before asking AI for help
- After long breaks

### 2. Check Sync Before Major Changes

**Why:** Avoid conflicts with other agents

**When:**
- Before editing shared files
- Before deployment
- Before refactoring

### 3. Use Auto-Deploy for Consistency

**Why:** Ensures deployments are reproducible

**When:**
- Every deployment
- Instead of manual kubectl
- For CI/CD automation

### 4. Monitor Agent Logs

**Location:** `.darwin/logs/agents.log`

**Why:** Debug agent behavior, track actions

### 5. Keep Agents Updated

**Update agents:**
```bash
# Pull latest from darwin-scaffold-studio
cd ~/workspace/darwin-scaffold-studio
git pull origin main

# Copy agent scripts to global
cp .darwin/agents/* ~/.darwin-global/agents/
```

---

## 🐛 Troubleshooting

### Agent Not Found

```bash
# Make scripts executable
chmod +x .darwin/agents/*.sh

# Verify
ls -lah .darwin/agents/
```

### Context Not Loading

```bash
# Check global Darwin directory
ls -la ~/.darwin-global/

# Verify SYNC_STATE
cat SYNC_STATE.json | jq .

# Create if missing
echo '{"last_update":"","active_agents":[],"locks":{},"last_actions":[],"version":"1.0.0"}' > SYNC_STATE.json
```

### Deployment Failures

```bash
# Check agent logs
tail -f .darwin/logs/agents.log

# Manual deployment
kubectl apply -f .darwin/cluster/k8s/

# Check pod status
kubectl describe pod <pod-name> -n darwin-scaffold-studio
```

### Lock Expired

```bash
# Check locks
jq '.locks' SYNC_STATE.json

# Remove expired locks (30+ min old)
# Edit SYNC_STATE.json manually or:
jq 'del(.locks["file.py"])' SYNC_STATE.json > SYNC_STATE.json.tmp
mv SYNC_STATE.json.tmp SYNC_STATE.json
```

---

## 🎯 Advanced Usage

### Custom Agent Development

**Create your own Darwin agent:**

```bash
#!/bin/bash
# my-custom-agent.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🤖 My Custom Darwin Agent"

# Load Darwin context
source "$SCRIPT_DIR/darwin-omniscient-agent.sh"

# Your custom logic
# ...

# Update SYNC_STATE
jq --arg ts "$(date -Iseconds)" \
   '.last_actions += [{action: "custom_agent", timestamp: $ts}]' \
   "$PROJECT_ROOT/SYNC_STATE.json" > "$PROJECT_ROOT/SYNC_STATE.json.tmp"
mv "$PROJECT_ROOT/SYNC_STATE.json.tmp" "$PROJECT_ROOT/SYNC_STATE.json"
```

### Agent API (Python)

```python
# darwin_core/agents/api.py

from darwin_core.services.semantic_memory_v2 import SemanticMemoryServiceV2

class DarwinAgentAPI:
    """Python API for Darwin agents"""
    
    def __init__(self):
        self.memory = SemanticMemoryServiceV2()
    
    def load_context(self, repo_name: str):
        """Load cross-repo context"""
        return self.memory.search(
            query=f"repo:{repo_name}",
            top_k=10
        )
    
    def check_locks(self, repo_name: str):
        """Check file locks"""
        sync_state = load_sync_state(repo_name)
        return sync_state.get('locks', {})
    
    def deploy(self, environment: str, version: str):
        """Automated deployment"""
        subprocess.run([
            './.darwin/agents/auto-deploy.sh',
            environment,
            version
        ])
```

---

## 📚 Integration Examples

### Example 1: Cursor Agent

```python
# In Cursor workspace
# User opens darwin-scaffold-studio repo

# Cursor runs automatically:
subprocess.run(['./.darwin/agents/darwin-omniscient-agent.sh'])

# Cursor now has context of ALL Darwin repos!
# Can answer: "What's the status of darwin-scaffold-studio?"
```

### Example 2: ChatGPT via MCP

```javascript
// ChatGPT Desktop calls Darwin MCP server

// MCP server runs omniscient agent internally
const context = await execSync('./.darwin/agents/darwin-omniscient-agent.sh');

// Provides context to ChatGPT
// ChatGPT sees: All Darwin repos, locks, active agents
```

### Example 3: CI/CD Pipeline

```yaml
# .github/workflows/cd.yml

jobs:
  deploy:
    steps:
    - name: Load Darwin context
      run: ./.darwin/agents/darwin-omniscient-agent.sh
    
    - name: Check sync
      run: ./.darwin/agents/sync-check.sh
    
    - name: Auto-deploy
      run: ./.darwin/agents/auto-deploy.sh production ${{ github.ref_name }}
```

---

## 🎊 Benefits

### For Developers

- ✅ Context-aware AI assistance
- ✅ Conflict detection
- ✅ Automated deployments
- ✅ Cross-repo visibility

### For AI Agents

- ✅ Complete project context
- ✅ Multi-repo awareness
- ✅ Lock coordination
- ✅ Better recommendations

### For Production

- ✅ Consistent deployments
- ✅ Automated testing
- ✅ Health verification
- ✅ Rollback capability

---

## 📞 Support

**Documentation:**
- Architecture: [ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
- Cluster Setup: [CLUSTER_SETUP.md](../deployment/CLUSTER_SETUP.md)

**Issues:**
- GitHub: https://github.com/agourakis82/darwin-scaffold-studio/issues

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**

