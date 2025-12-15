<script lang="ts">
  import '../app.css';
  import Sidebar from '$lib/components/layout/Sidebar.svelte';
  import TopBar from '$lib/components/layout/TopBar.svelte';
  import StatusBar from '$lib/components/layout/StatusBar.svelte';
  import ToastContainer from '$lib/components/shared/ToastContainer.svelte';
  import SettingsPanel from '$lib/components/settings/SettingsPanel.svelte';
  import { juliaConnected } from '$lib/stores/julia';
  import { settings, resolvedTheme } from '$lib/stores/settings';
  import { showError, showSuccess } from '$lib/stores/toast';
  import { onMount } from 'svelte';

  let sidebarCollapsed = false;
  let showSettings = false;

  onMount(() => {
    // Apply theme on mount
    applyTheme($resolvedTheme);

    // Check Julia server connection on mount
    checkJuliaConnection();

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', handleSystemThemeChange);

    return () => {
      mediaQuery.removeEventListener('change', handleSystemThemeChange);
    };
  });

  function handleSystemThemeChange() {
    if ($settings.theme === 'system') {
      applyTheme($resolvedTheme);
    }
  }

  function applyTheme(theme: 'dark' | 'light') {
    document.documentElement.setAttribute('data-theme', theme);
  }

  async function checkJuliaConnection() {
    try {
      const response = await fetch('http://localhost:8081/health');
      if (response.ok) {
        juliaConnected.set(true);
        showSuccess('Connected', 'Julia server is online');
      } else {
        juliaConnected.set(false);
      }
    } catch {
      juliaConnected.set(false);
      showError('Connection Failed', 'Could not connect to Julia server');
    }
  }

  function handleOpenSettings() {
    showSettings = true;
  }

  // Reactive theme application
  $: if (typeof window !== 'undefined') {
    applyTheme($resolvedTheme);
  }
</script>

<svelte:window on:keydown={(e) => {
  if (e.key === ',' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    showSettings = !showSettings;
  }
  if (e.key === 'Escape' && showSettings) {
    showSettings = false;
  }
}} />

<div class="app-container">
  <TopBar on:openSettings={handleOpenSettings} />

  <div class="main-content">
    <Sidebar bind:collapsed={sidebarCollapsed} />

    <main class="workspace" class:sidebar-collapsed={sidebarCollapsed}>
      <slot />
    </main>
  </div>

  <StatusBar />

  <!-- Settings Modal -->
  {#if showSettings}
    <div class="modal-overlay" on:click={() => showSettings = false} on:keydown={(e) => e.key === 'Escape' && (showSettings = false)} role="button" tabindex="0">
      <div class="modal-content" on:click|stopPropagation on:keydown|stopPropagation role="dialog" aria-modal="true">
        <SettingsPanel on:close={() => showSettings = false} />
      </div>
    </div>
  {/if}

  <!-- Toast Notifications -->
  <ToastContainer />
</div>

<style>
  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .workspace {
    flex: 1;
    overflow: auto;
    padding: 16px;
    margin-left: var(--sidebar-width);
    transition: margin-left var(--transition-base);
  }

  .workspace.sidebar-collapsed {
    margin-left: 64px;
  }

  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.2s ease-out;
  }

  .modal-content {
    width: 480px;
    max-height: 80vh;
    background: var(--bg-primary);
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    overflow: hidden;
    animation: slideIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: scale(0.95) translateY(-10px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }
</style>
