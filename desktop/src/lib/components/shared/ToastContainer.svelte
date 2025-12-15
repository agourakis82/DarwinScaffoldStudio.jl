<script lang="ts">
  import { toasts } from '$lib/stores/toast';
  import { flip } from 'svelte/animate';
  import { fly } from 'svelte/transition';

  const icons = {
    success: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`,
    error: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`,
    warning: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
    info: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`,
  };
</script>

<div class="toast-container" aria-live="polite">
  {#each $toasts as toast (toast.id)}
    <div
      class="toast toast-{toast.type}"
      animate:flip={{ duration: 200 }}
      in:fly={{ x: 300, duration: 300 }}
      out:fly={{ x: 300, duration: 200 }}
      role="alert"
    >
      <div class="toast-icon">
        {@html icons[toast.type]}
      </div>
      <div class="toast-content">
        <span class="toast-title">{toast.title}</span>
        {#if toast.message}
          <span class="toast-message">{toast.message}</span>
        {/if}
      </div>
      {#if toast.dismissible}
        <button class="toast-dismiss" on:click={() => toasts.dismiss(toast.id)} aria-label="Dismiss">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      {/if}
    </div>
  {/each}
</div>

<style>
  .toast-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column-reverse;
    gap: 10px;
    max-width: 360px;
    pointer-events: none;
  }

  .toast {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    pointer-events: auto;
  }

  .toast-success {
    border-left: 3px solid var(--success);
  }

  .toast-error {
    border-left: 3px solid var(--error);
  }

  .toast-warning {
    border-left: 3px solid var(--warning);
  }

  .toast-info {
    border-left: 3px solid var(--primary);
  }

  .toast-icon {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .toast-success .toast-icon {
    color: var(--success);
  }

  .toast-error .toast-icon {
    color: var(--error);
  }

  .toast-warning .toast-icon {
    color: var(--warning);
  }

  .toast-info .toast-icon {
    color: var(--primary);
  }

  .toast-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .toast-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .toast-message {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .toast-dismiss {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 4px;
    transition: all var(--transition-fast);
    margin: -4px -4px -4px 0;
  }

  .toast-dismiss:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
</style>
