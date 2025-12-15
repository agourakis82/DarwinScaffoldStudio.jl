<script lang="ts">
  import type { Anomaly } from '$lib/stores/digitalTwin';

  export let anomalies: Anomaly[] = [];

  $: criticalCount = anomalies.filter(a => a.severity === 'critical').length;
  $: warningCount = anomalies.filter(a => a.severity === 'warning').length;

  function formatTime(date: Date): string {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
</script>

<div class="anomaly-indicator" class:has-critical={criticalCount > 0} class:has-warning={warningCount > 0 && criticalCount === 0}>
  <div class="indicator-header">
    <div class="indicator-icon">
      {#if criticalCount > 0}
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="15" y1="9" x2="9" y2="15"></line>
          <line x1="9" y1="9" x2="15" y2="15"></line>
        </svg>
      {:else if warningCount > 0}
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
      {:else}
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
          <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
      {/if}
    </div>
    <div class="indicator-status">
      {#if criticalCount > 0}
        <span class="status-text critical">{criticalCount} Critical</span>
      {:else if warningCount > 0}
        <span class="status-text warning">{warningCount} Warning{warningCount > 1 ? 's' : ''}</span>
      {:else}
        <span class="status-text normal">All Normal</span>
      {/if}
    </div>
  </div>

  {#if anomalies.length > 0}
    <div class="anomaly-list">
      {#each anomalies.slice(-5) as anomaly}
        <div class="anomaly-item" class:critical={anomaly.severity === 'critical'}>
          <span class="anomaly-sensor">{anomaly.sensor}</span>
          <span class="anomaly-value">{anomaly.value.toFixed(2)}</span>
          <span class="anomaly-range">
            (range: {anomaly.threshold.min}-{anomaly.threshold.max})
          </span>
          <span class="anomaly-time">{formatTime(anomaly.timestamp)}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .anomaly-indicator {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
    border-left: 3px solid var(--success);
  }

  .anomaly-indicator.has-warning {
    border-left-color: var(--warning);
  }

  .anomaly-indicator.has-critical {
    border-left-color: var(--error);
    animation: pulse-border 2s infinite;
  }

  @keyframes pulse-border {
    0%, 100% { border-left-color: var(--error); }
    50% { border-left-color: rgba(239, 68, 68, 0.5); }
  }

  .indicator-header {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .indicator-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: var(--bg-tertiary);
  }

  .anomaly-indicator:not(.has-warning):not(.has-critical) .indicator-icon {
    color: var(--success);
  }

  .anomaly-indicator.has-warning .indicator-icon {
    color: var(--warning);
    background: rgba(245, 158, 11, 0.1);
  }

  .anomaly-indicator.has-critical .indicator-icon {
    color: var(--error);
    background: rgba(239, 68, 68, 0.1);
  }

  .indicator-status {
    flex: 1;
  }

  .status-text {
    font-size: 14px;
    font-weight: 600;
  }

  .status-text.normal {
    color: var(--success);
  }

  .status-text.warning {
    color: var(--warning);
  }

  .status-text.critical {
    color: var(--error);
  }

  .anomaly-list {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .anomaly-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .anomaly-item.critical {
    background: rgba(239, 68, 68, 0.1);
  }

  .anomaly-sensor {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 80px;
  }

  .anomaly-value {
    color: var(--warning);
    font-weight: 500;
  }

  .anomaly-item.critical .anomaly-value {
    color: var(--error);
  }

  .anomaly-range {
    color: var(--text-muted);
    flex: 1;
  }

  .anomaly-time {
    color: var(--text-muted);
  }
</style>
