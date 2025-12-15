<script lang="ts">
  import { page } from '$app/stores';
  import { juliaConnected } from '$lib/stores/julia';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ openSettings: void }>();

  const navItems = [
    { href: '/', label: 'Home', icon: 'home' },
    { href: '/analyze', label: 'Analyze', icon: 'scan' },
    { href: '/design', label: 'Design', icon: 'cube' },
    { href: '/fabricate', label: 'Fabricate', icon: 'printer' },
  ];

  $: currentPath = $page.url.pathname;

  function handleOpenSettings() {
    dispatch('openSettings');
  }
</script>

<header class="topbar">
  <div class="logo">
    <div class="logo-icon">
      <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
        <polyline points="7.5 4.21 12 6.81 16.5 4.21"></polyline>
        <polyline points="7.5 19.79 7.5 14.6 3 12"></polyline>
        <polyline points="21 12 16.5 14.6 16.5 19.79"></polyline>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
        <line x1="12" y1="22.08" x2="12" y2="12"></line>
      </svg>
    </div>
    <span class="logo-text">Darwin Scaffold Studio</span>
  </div>

  <nav class="nav-tabs">
    {#each navItems as item}
      <a
        href={item.href}
        class="nav-tab"
        class:active={currentPath === item.href || (item.href !== '/' && currentPath.startsWith(item.href))}
      >
        {item.label}
      </a>
    {/each}
  </nav>

  <div class="topbar-actions">
    <div class="connection-status" class:connected={$juliaConnected}>
      <span class="status-dot"></span>
      <span class="status-text">{$juliaConnected ? 'Connected' : 'Disconnected'}</span>
    </div>

    <button class="icon-btn" title="Settings (Ctrl+,)" on:click={handleOpenSettings}>
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
    </button>
  </div>
</header>

<style>
  .topbar {
    display: flex;
    align-items: center;
    height: var(--topbar-height);
    padding: 0 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    gap: 24px;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }

  .logo-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 8px;
    color: white;
  }

  .logo-text {
    font-weight: 600;
    font-size: 16px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .nav-tabs {
    display: flex;
    gap: 4px;
    flex: 1;
    justify-content: center;
  }

  .nav-tab {
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: 6px;
    transition: all var(--transition-fast);
  }

  .nav-tab:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  .nav-tab.active {
    color: var(--primary);
    background: rgba(74, 158, 255, 0.1);
  }

  .topbar-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    background: var(--bg-tertiary);
    font-size: 12px;
    color: var(--text-secondary);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--error);
    animation: pulse 2s infinite;
  }

  .connection-status.connected .status-dot {
    background: var(--success);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
</style>
