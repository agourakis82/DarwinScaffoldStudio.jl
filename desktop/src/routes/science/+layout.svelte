<script lang="ts">
  import { page } from '$app/stores';

  const modules = [
    { href: '/science/tda', name: 'Topology (TDA)', desc: 'Persistent homology analysis', color: '#8b5cf6' },
    { href: '/science/twin', name: 'Digital Twin', desc: 'Real-time biosensor monitoring', color: '#10b981' },
    { href: '/science/pinn', name: 'PINN Fields', desc: 'Nutrient concentration simulation', color: '#f59e0b' },
    { href: '/science/growth', name: 'Tissue Growth', desc: 'Cell differentiation simulation', color: '#ec4899' },
    { href: '/science/causal', name: 'Causal Discovery', desc: 'DAG learning & effect estimation', color: '#3b82f6' },
    { href: '/science/gnn', name: 'GNN Prediction', desc: 'Graph neural network properties', color: '#14b8a6' },
    { href: '/science/sem', name: 'SEM Classification', desc: 'AI cell type identification', color: '#f97316' },
  ];

  $: currentPath = $page.url.pathname;
</script>

<div class="science-layout">
  <nav class="science-nav">
    <div class="nav-header">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 16v-4"></path>
        <path d="M12 8h.01"></path>
      </svg>
      <span>Science Modules</span>
    </div>

    <div class="nav-links">
      {#each modules as mod}
        <a
          href={mod.href}
          class="nav-link"
          class:active={currentPath === mod.href}
        >
          <span class="link-dot" style="background: {mod.color}"></span>
          <div class="link-content">
            <span class="link-name">{mod.name}</span>
            <span class="link-desc">{mod.desc}</span>
          </div>
        </a>
      {/each}
    </div>

    <div class="nav-footer">
      <a href="/" class="back-link">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="19" y1="12" x2="5" y2="12"></line>
          <polyline points="12 19 5 12 12 5"></polyline>
        </svg>
        Back to Studio
      </a>
    </div>
  </nav>

  <div class="science-content">
    <slot />
  </div>
</div>

<style>
  .science-layout {
    display: flex;
    height: 100%;
    gap: 24px;
  }

  .science-nav {
    width: 260px;
    flex-shrink: 0;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
  }

  .nav-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    margin-bottom: 16px;
    color: var(--text-primary);
    font-weight: 600;
    font-size: 14px;
  }

  .nav-links {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .nav-link {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
    text-decoration: none;
    transition: all var(--transition-fast);
  }

  .nav-link:hover {
    background: var(--bg-tertiary);
  }

  .nav-link.active {
    background: var(--bg-tertiary);
    border-left: 3px solid var(--primary);
    margin-left: -3px;
  }

  .link-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
  }

  .link-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .link-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .link-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .nav-footer {
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
    margin-top: 16px;
  }

  .back-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-secondary);
    text-decoration: none;
    transition: all var(--transition-fast);
  }

  .back-link:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .science-content {
    flex: 1;
    overflow: auto;
  }
</style>
