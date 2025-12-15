<script lang="ts">
  import { settings, setTheme, setLanguage, resolvedTheme } from '$lib/stores/settings';
  import { juliaStatus } from '$lib/stores/julia';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{ close: void }>();

  // Local state for form inputs
  let juliaUrl = $settings.juliaServerUrl;
  let ollamaUrl = $settings.ollamaServerUrl;
  let ollamaModel = $settings.ai.ollamaModel;

  // Helper functions for type-safe event handlers
  function handleLanguageChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    setLanguage(target.value as 'en' | 'pt' | 'es');
  }

  function handleDefaultAgentChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    settings.update((s) => ({ ...s, ai: { ...s.ai, defaultAgent: target.value as 'design' | 'analysis' | 'synthesis' } }));
  }

  function handleExportFormatChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    settings.update((s) => ({ ...s, export: { ...s.export, defaultFormat: target.value as 'stl' | 'gcode' | 'obj' } }));
  }

  function handleExportQualityChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    settings.update((s) => ({ ...s, export: { ...s.export, defaultQuality: target.value as 'low' | 'medium' | 'high' } }));
  }

  // Available models
  const ollamaModels = [
    'qwen2.5:7b',
    'qwen2.5:14b',
    'llama3.2:3b',
    'llama3.2:7b',
    'mistral:7b',
    'codellama:7b',
  ];

  function handleThemeChange(theme: 'dark' | 'light' | 'system') {
    setTheme(theme);
    applyTheme(theme === 'system' ? $resolvedTheme : theme);
  }

  function applyTheme(theme: 'dark' | 'light') {
    document.documentElement.setAttribute('data-theme', theme);
  }

  function saveServerSettings() {
    settings.update((s) => ({
      ...s,
      juliaServerUrl: juliaUrl,
      ollamaServerUrl: ollamaUrl,
    }));
  }

  function saveAISettings() {
    settings.update((s) => ({
      ...s,
      ai: { ...s.ai, ollamaModel },
    }));
  }

  function toggleVisualization(key: keyof typeof $settings.visualization) {
    settings.update((s) => ({
      ...s,
      visualization: {
        ...s.visualization,
        [key]: !s.visualization[key],
      },
    }));
  }

  function resetSettings() {
    if (confirm('Reset all settings to defaults?')) {
      settings.reset();
      juliaUrl = $settings.juliaServerUrl;
      ollamaUrl = $settings.ollamaServerUrl;
      ollamaModel = $settings.ai.ollamaModel;
    }
  }
</script>

<div class="settings-panel">
  <div class="settings-header">
    <h2>Settings</h2>
    <button class="close-btn" on:click={() => dispatch('close')}>
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
    </button>
  </div>

  <div class="settings-content">
    <!-- Appearance -->
    <section class="settings-section">
      <h3>Appearance</h3>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Theme</span>
          <span class="setting-desc">Choose your preferred color scheme</span>
        </div>
        <div class="theme-buttons">
          <button
            class="theme-btn"
            class:active={$settings.theme === 'dark'}
            on:click={() => handleThemeChange('dark')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
            Dark
          </button>
          <button
            class="theme-btn"
            class:active={$settings.theme === 'light'}
            on:click={() => handleThemeChange('light')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="5"></circle>
              <line x1="12" y1="1" x2="12" y2="3"></line>
              <line x1="12" y1="21" x2="12" y2="23"></line>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
              <line x1="1" y1="12" x2="3" y2="12"></line>
              <line x1="21" y1="12" x2="23" y2="12"></line>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
            Light
          </button>
          <button
            class="theme-btn"
            class:active={$settings.theme === 'system'}
            on:click={() => handleThemeChange('system')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
              <line x1="8" y1="21" x2="16" y2="21"></line>
              <line x1="12" y1="17" x2="12" y2="21"></line>
            </svg>
            System
          </button>
        </div>
      </div>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Language</span>
          <span class="setting-desc">Interface language</span>
        </div>
        <select class="setting-select" value={$settings.language} on:change={handleLanguageChange}>
          <option value="en">English</option>
          <option value="pt">Portugues</option>
          <option value="es">Espanol</option>
        </select>
      </div>
    </section>

    <!-- Server Connections -->
    <section class="settings-section">
      <h3>Connections</h3>

      <div class="setting-item vertical">
        <div class="setting-info">
          <span class="setting-label">Julia Server URL</span>
          <span class="setting-desc">DarwinScaffoldStudio backend</span>
        </div>
        <div class="input-with-status">
          <input
            type="text"
            class="setting-input"
            bind:value={juliaUrl}
            on:blur={saveServerSettings}
            placeholder="http://localhost:8081"
          />
          <span class="status-dot" class:connected={$juliaStatus === 'connected'}></span>
        </div>
      </div>

      <div class="setting-item vertical">
        <div class="setting-info">
          <span class="setting-label">Ollama Server URL</span>
          <span class="setting-desc">Local LLM server</span>
        </div>
        <input
          type="text"
          class="setting-input"
          bind:value={ollamaUrl}
          on:blur={saveServerSettings}
          placeholder="http://localhost:11434"
        />
      </div>

      <label class="checkbox-setting">
        <input
          type="checkbox"
          checked={$settings.autoConnect}
          on:change={() => settings.update((s) => ({ ...s, autoConnect: !s.autoConnect }))}
        />
        <span>Auto-connect on startup</span>
      </label>
    </section>

    <!-- AI Settings -->
    <section class="settings-section">
      <h3>AI Assistant</h3>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Default Model</span>
          <span class="setting-desc">Ollama model for AI agents</span>
        </div>
        <select class="setting-select" bind:value={ollamaModel} on:change={saveAISettings}>
          {#each ollamaModels as model}
            <option value={model}>{model}</option>
          {/each}
        </select>
      </div>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Default Agent</span>
          <span class="setting-desc">Agent to use for new chats</span>
        </div>
        <select
          class="setting-select"
          value={$settings.ai.defaultAgent}
          on:change={handleDefaultAgentChange}
        >
          <option value="design">Design Assistant</option>
          <option value="analysis">Analysis Expert</option>
          <option value="synthesis">Synthesis Guide</option>
        </select>
      </div>

      <label class="checkbox-setting">
        <input
          type="checkbox"
          checked={$settings.ai.streamResponses}
          on:change={() => settings.update((s) => ({ ...s, ai: { ...s.ai, streamResponses: !s.ai.streamResponses } }))}
        />
        <span>Stream AI responses</span>
      </label>
    </section>

    <!-- Visualization -->
    <section class="settings-section">
      <h3>3D Visualization</h3>

      <label class="checkbox-setting">
        <input
          type="checkbox"
          checked={$settings.visualization.showGrid}
          on:change={() => toggleVisualization('showGrid')}
        />
        <span>Show grid</span>
      </label>

      <label class="checkbox-setting">
        <input
          type="checkbox"
          checked={$settings.visualization.showAxes}
          on:change={() => toggleVisualization('showAxes')}
        />
        <span>Show axes</span>
      </label>

      <label class="checkbox-setting">
        <input
          type="checkbox"
          checked={$settings.visualization.antialiasing}
          on:change={() => toggleVisualization('antialiasing')}
        />
        <span>Antialiasing (requires restart)</span>
      </label>
    </section>

    <!-- Export -->
    <section class="settings-section">
      <h3>Export Defaults</h3>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Default Format</span>
        </div>
        <select
          class="setting-select"
          value={$settings.export.defaultFormat}
          on:change={handleExportFormatChange}
        >
          <option value="stl">STL</option>
          <option value="gcode">G-Code</option>
          <option value="obj">OBJ</option>
        </select>
      </div>

      <div class="setting-item">
        <div class="setting-info">
          <span class="setting-label">Default Quality</span>
        </div>
        <select
          class="setting-select"
          value={$settings.export.defaultQuality}
          on:change={handleExportQualityChange}
        >
          <option value="low">Low (Fast)</option>
          <option value="medium">Medium</option>
          <option value="high">High (Slow)</option>
        </select>
      </div>
    </section>

    <!-- Danger Zone -->
    <section class="settings-section danger">
      <h3>Reset</h3>
      <button class="reset-btn" on:click={resetSettings}>
        Reset All Settings
      </button>
    </section>
  </div>

  <div class="settings-footer">
    <span class="version">Darwin Scaffold Studio v2.3.1</span>
  </div>
</div>

<style>
  .settings-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary);
  }

  .settings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .settings-header h2 {
    font-size: 18px;
    font-weight: 600;
    margin: 0;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 6px;
    transition: all var(--transition-fast);
  }

  .close-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .settings-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .settings-section {
    margin-bottom: 28px;
  }

  .settings-section h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin: 0 0 16px 0;
  }

  .setting-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .setting-item.vertical {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
  }

  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .setting-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .theme-buttons {
    display: flex;
    gap: 6px;
  }

  .theme-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .theme-btn:hover {
    border-color: var(--border-color);
  }

  .theme-btn.active {
    border-color: var(--primary);
    background: rgba(74, 158, 255, 0.1);
    color: var(--primary);
  }

  .setting-select {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    min-width: 140px;
  }

  .setting-select:focus {
    outline: none;
    border-color: var(--primary);
  }

  .setting-input {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .setting-input:focus {
    outline: none;
    border-color: var(--primary);
  }

  .input-with-status {
    position: relative;
  }

  .input-with-status input {
    padding-right: 32px;
  }

  .status-dot {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--error);
  }

  .status-dot.connected {
    background: var(--success);
  }

  .checkbox-setting {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    cursor: pointer;
  }

  .checkbox-setting input {
    width: 16px;
    height: 16px;
    accent-color: var(--primary);
  }

  .checkbox-setting span {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .settings-section.danger {
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
  }

  .reset-btn {
    padding: 10px 16px;
    background: transparent;
    border: 1px solid var(--error);
    border-radius: 6px;
    color: var(--error);
    font-size: 12px;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .reset-btn:hover {
    background: rgba(239, 68, 68, 0.1);
  }

  .settings-footer {
    padding: 12px 20px;
    border-top: 1px solid var(--border-color);
    text-align: center;
  }

  .version {
    font-size: 11px;
    color: var(--text-muted);
  }
</style>
