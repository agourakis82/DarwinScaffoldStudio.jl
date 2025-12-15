<script lang="ts">
  const modules = [
    {
      href: '/science/tda',
      name: 'Topological Data Analysis',
      desc: 'Compute persistent homology to analyze scaffold pore connectivity, tunnels, and voids. Visualize persistence diagrams and Betti numbers.',
      color: '#8b5cf6',
      icon: 'share-2',
      features: ['Persistence Diagrams', 'Betti Numbers', 'Interconnectivity Score'],
    },
    {
      href: '/science/twin',
      name: 'Digital Twin Dashboard',
      desc: 'Monitor real-time biosensor data during cell culture. Track pH, O2, glucose, and temperature with anomaly detection and predictive forecasting.',
      color: '#10b981',
      icon: 'activity',
      features: ['Live Sensors', 'Anomaly Detection', '24h Forecasts'],
    },
    {
      href: '/science/pinn',
      name: 'PINN Concentration Fields',
      desc: 'Physics-informed neural network simulation of nutrient transport through the scaffold. Identify hypoxic regions that may limit cell viability.',
      color: '#f59e0b',
      icon: 'grid',
      features: ['3D Heatmaps', 'Time Evolution', 'Hypoxia Detection'],
    },
    {
      href: '/science/growth',
      name: 'Tissue Growth Simulation',
      desc: 'Agent-based model of cell proliferation and differentiation. Watch MSCs differentiate into osteoblasts and osteocytes over time.',
      color: '#ec4899',
      icon: 'git-branch',
      features: ['Cell Animation', 'Population Charts', 'Differentiation Tree'],
    },
    {
      href: '/science/causal',
      name: 'Causal Discovery',
      desc: 'Learn causal DAGs from scaffold-tissue data using PC, FCI, or NOTEARS algorithms. Estimate treatment effects with backdoor adjustment.',
      color: '#3b82f6',
      icon: 'git-merge',
      features: ['DAG Learning', 'Effect Estimation', 'Confounder Control'],
    },
    {
      href: '/science/gnn',
      name: 'GNN Property Prediction',
      desc: 'Graph neural network for predicting scaffold properties from pore network topology. Visualize node importance with 3D heatmaps.',
      color: '#14b8a6',
      icon: 'hexagon',
      features: ['Property Prediction', 'Node Importance', 'GAT/GCN/Transformer'],
    },
    {
      href: '/science/sem',
      name: 'SEM Cell Classification',
      desc: 'AI-powered cell type identification from scanning electron microscopy images. Classify osteoblasts, chondrocytes, fibroblasts, and more.',
      color: '#f97316',
      icon: 'eye',
      features: ['Cell Detection', 'Type Classification', 'Morphology Analysis'],
    },
  ];
</script>

<div class="science-overview">
  <header class="overview-header">
    <h1>Science Modules</h1>
    <p>Advanced computational analysis and simulation tools for scaffold characterization</p>
  </header>

  <div class="modules-grid">
    {#each modules as mod}
      <a href={mod.href} class="module-card">
        <div class="card-header">
          <span class="card-dot" style="background: {mod.color}"></span>
          <h2>{mod.name}</h2>
        </div>
        <p class="card-desc">{mod.desc}</p>
        <div class="card-features">
          {#each mod.features as feature}
            <span class="feature-tag">{feature}</span>
          {/each}
        </div>
        <div class="card-action">
          Open Module
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        </div>
      </a>
    {/each}
  </div>
</div>

<style>
  .science-overview {
    max-width: 900px;
    margin: 0 auto;
  }

  .overview-header {
    text-align: center;
    margin-bottom: 40px;
  }

  .overview-header h1 {
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .overview-header p {
    font-size: 14px;
    color: var(--text-secondary);
  }

  .modules-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
  }

  .module-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    text-decoration: none;
    transition: all var(--transition-base);
    display: flex;
    flex-direction: column;
  }

  .module-card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .card-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .card-header h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .card-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 16px;
    flex: 1;
  }

  .card-features {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
  }

  .feature-tag {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .card-action {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--primary);
  }

  .module-card:hover .card-action svg {
    transform: translateX(4px);
    transition: transform var(--transition-fast);
  }

  @media (max-width: 700px) {
    .modules-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
