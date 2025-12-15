<script lang="ts">
  import { page } from '$app/stores';
  import { material, tissue } from '$lib/stores/scaffold';

  export let collapsed = false;

  $: currentPath = $page.url.pathname;

  const scienceModules = [
    { href: '/science/tda', name: 'Topology (TDA)', icon: 'share-2', color: '#8b5cf6' },
    { href: '/science/twin', name: 'Digital Twin', icon: 'activity', color: '#10b981' },
    { href: '/science/pinn', name: 'PINN Fields', icon: 'grid', color: '#f59e0b' },
    { href: '/science/growth', name: 'Tissue Growth', icon: 'git-branch', color: '#ec4899' },
  ];

  const materials = [
    { id: 'PCL', name: 'PCL', desc: 'Poly-e-caprolactone' },
    { id: 'PLA', name: 'PLA', desc: 'Poly(lactic acid)' },
    { id: 'PLGA', name: 'PLGA', desc: 'Copolymer' },
    { id: 'Collagen', name: 'Collagen', desc: 'Natural ECM' },
    { id: 'Chitosan', name: 'Chitosan', desc: 'Biopolymer' },
    { id: 'Hydroxyapatite', name: 'HA/TCP', desc: 'Ceramic' },
    { id: 'Ti6Al4V', name: 'Ti6Al4V', desc: 'Titanium alloy' },
  ];

  const tissues = [
    { id: 'bone', name: 'Bone', icon: '(bones)', color: '#f59e0b' },
    { id: 'cartilage', name: 'Cartilage', icon: '(cart)', color: '#3b82f6' },
    { id: 'skin', name: 'Skin', icon: '(skin)', color: '#ec4899' },
    { id: 'vascular', name: 'Vascular', icon: '(vasc)', color: '#ef4444' },
    { id: 'neural', name: 'Neural', icon: '(neur)', color: '#a855f7' },
  ];

  const tools = [
    { id: 'add', name: 'Add Material', icon: 'plus-circle' },
    { id: 'remove', name: 'Remove Material', icon: 'minus-circle' },
    { id: 'smooth', name: 'Smooth Surface', icon: 'layers' },
    { id: 'erode', name: 'Erode', icon: 'minimize' },
    { id: 'dilate', name: 'Dilate', icon: 'maximize' },
  ];

  function toggleCollapse() {
    collapsed = !collapsed;
  }
</script>

<aside class="sidebar" class:collapsed>
  <button class="collapse-btn" on:click={toggleCollapse}>
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      {#if collapsed}
        <polyline points="9 18 15 12 9 6"></polyline>
      {:else}
        <polyline points="15 18 9 12 15 6"></polyline>
      {/if}
    </svg>
  </button>

  {#if !collapsed}
    <div class="sidebar-content">
      <!-- Material Selector -->
      <section class="sidebar-section">
        <h3 class="section-title">Material</h3>
        <div class="material-grid">
          {#each materials as mat}
            <button
              class="material-chip"
              class:selected={$material === mat.id}
              on:click={() => material.set(mat.id)}
              title={mat.desc}
            >
              {mat.name}
            </button>
          {/each}
        </div>
      </section>

      <!-- Tissue Selector -->
      <section class="sidebar-section">
        <h3 class="section-title">Target Tissue</h3>
        <div class="tissue-list">
          {#each tissues as t}
            <button
              class="tissue-option"
              class:selected={$tissue === t.id}
              on:click={() => tissue.set(t.id)}
            >
              <span class="tissue-dot" style="background: {t.color}"></span>
              <span class="tissue-name">{t.name}</span>
            </button>
          {/each}
        </div>
      </section>

      <!-- Tools -->
      <section class="sidebar-section">
        <h3 class="section-title">Tools</h3>
        <div class="tool-list">
          {#each tools as tool}
            <button class="tool-btn" title={tool.name}>
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                {#if tool.icon === 'plus-circle'}
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="16"></line>
                  <line x1="8" y1="12" x2="16" y2="12"></line>
                {:else if tool.icon === 'minus-circle'}
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="8" y1="12" x2="16" y2="12"></line>
                {:else if tool.icon === 'layers'}
                  <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                  <polyline points="2 17 12 22 22 17"></polyline>
                  <polyline points="2 12 12 17 22 12"></polyline>
                {:else if tool.icon === 'minimize'}
                  <polyline points="4 14 10 14 10 20"></polyline>
                  <polyline points="20 10 14 10 14 4"></polyline>
                  <line x1="14" y1="10" x2="21" y2="3"></line>
                  <line x1="3" y1="21" x2="10" y2="14"></line>
                {:else if tool.icon === 'maximize'}
                  <polyline points="15 3 21 3 21 9"></polyline>
                  <polyline points="9 21 3 21 3 15"></polyline>
                  <line x1="21" y1="3" x2="14" y2="10"></line>
                  <line x1="3" y1="21" x2="10" y2="14"></line>
                {/if}
              </svg>
              <span>{tool.name}</span>
            </button>
          {/each}
        </div>
      </section>

      <!-- Science Modules -->
      <section class="sidebar-section">
        <h3 class="section-title">Science</h3>
        <div class="science-list">
          {#each scienceModules as mod}
            <a
              href={mod.href}
              class="science-link"
              class:active={currentPath.startsWith(mod.href)}
            >
              <span class="science-dot" style="background: {mod.color}"></span>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                {#if mod.icon === 'share-2'}
                  <circle cx="18" cy="5" r="3"></circle>
                  <circle cx="6" cy="12" r="3"></circle>
                  <circle cx="18" cy="19" r="3"></circle>
                  <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                  <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                {:else if mod.icon === 'activity'}
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                {:else if mod.icon === 'grid'}
                  <rect x="3" y="3" width="7" height="7"></rect>
                  <rect x="14" y="3" width="7" height="7"></rect>
                  <rect x="14" y="14" width="7" height="7"></rect>
                  <rect x="3" y="14" width="7" height="7"></rect>
                {:else if mod.icon === 'git-branch'}
                  <line x1="6" y1="3" x2="6" y2="15"></line>
                  <circle cx="18" cy="6" r="3"></circle>
                  <circle cx="6" cy="18" r="3"></circle>
                  <path d="M18 9a9 9 0 0 1-9 9"></path>
                {/if}
              </svg>
              <span class="science-name">{mod.name}</span>
            </a>
          {/each}
        </div>
      </section>
    </div>
  {/if}
</aside>

<style>
  .sidebar {
    position: fixed;
    left: 0;
    top: var(--topbar-height);
    bottom: var(--statusbar-height);
    width: var(--sidebar-width);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    transition: width var(--transition-base);
    z-index: 10;
  }

  .sidebar.collapsed {
    width: 64px;
  }

  .collapse-btn {
    position: absolute;
    right: -12px;
    top: 20px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 11;
    transition: all var(--transition-fast);
  }

  .collapse-btn:hover {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
  }

  .sidebar-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .sidebar-section {
    margin-bottom: 24px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  .material-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .material-chip {
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 16px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .material-chip:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
  }

  .material-chip.selected {
    background: rgba(74, 158, 255, 0.15);
    border-color: var(--primary);
    color: var(--primary);
  }

  .tissue-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .tissue-option {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
    text-align: left;
  }

  .tissue-option:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tissue-option.selected {
    background: var(--bg-tertiary);
    border-color: var(--border-color);
    color: var(--text-primary);
  }

  .tissue-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .tissue-name {
    font-size: 13px;
    font-weight: 500;
  }

  .tool-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .tool-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
    font-size: 13px;
  }

  .tool-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .science-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .science-link {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: transparent;
    color: var(--text-secondary);
    text-decoration: none;
    transition: all var(--transition-fast);
  }

  .science-link:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .science-link.active {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-left: 2px solid var(--primary);
  }

  .science-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .science-name {
    font-size: 13px;
    font-weight: 500;
  }
</style>
