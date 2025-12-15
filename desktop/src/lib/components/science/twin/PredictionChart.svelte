<script lang="ts">
  import type { Prediction } from '$lib/stores/digitalTwin';
  import { SENSOR_RANGES } from '$lib/stores/digitalTwin';

  export let predictions: Prediction[] = [];
  export let sensor: 'pH' | 'O2' | 'glucose' = 'pH';

  $: range = SENSOR_RANGES[sensor];
  $: sortedPredictions = [...predictions].sort((a, b) => a.hoursAhead - b.hoursAhead);

  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'var(--success)';
    if (confidence >= 0.6) return 'var(--warning)';
    return 'var(--error)';
  }

  function getValueStatus(value: number): 'normal' | 'warning' | 'critical' {
    if (value >= range.min && value <= range.max) return 'normal';
    const deviation = value < range.min ? range.min - value : value - range.max;
    const maxDeviation = (range.max - range.min) / 2;
    return deviation > maxDeviation ? 'critical' : 'warning';
  }
</script>

<div class="prediction-chart">
  <h3 class="title">
    {sensor} Forecast
    <span class="subtitle">Next 24 hours</span>
  </h3>

  <div class="predictions">
    {#each sortedPredictions as pred}
      {@const status = getValueStatus(pred[sensor])}
      <div class="prediction-card" class:warning={status === 'warning'} class:critical={status === 'critical'}>
        <div class="time-label">+{pred.hoursAhead}h</div>
        <div class="value" class:warning={status === 'warning'} class:critical={status === 'critical'}>
          {pred[sensor].toFixed(sensor === 'pH' ? 2 : 1)}
          {#if range.unit}<span class="unit">{range.unit}</span>{/if}
        </div>
        <div class="confidence">
          <div class="confidence-bar" style="width: {pred.confidence * 100}%; background: {getConfidenceColor(pred.confidence)}"></div>
        </div>
        <div class="confidence-label">{Math.round(pred.confidence * 100)}% conf.</div>
      </div>
    {/each}
  </div>

  {#if sortedPredictions.some(p => getValueStatus(p[sensor]) !== 'normal')}
    <div class="alert">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
      <span>Predicted values may exceed optimal range</span>
    </div>
  {/if}
</div>

<style>
  .prediction-chart {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .subtitle {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-muted);
  }

  .predictions {
    display: flex;
    gap: 12px;
  }

  .prediction-card {
    flex: 1;
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    transition: all var(--transition-fast);
  }

  .prediction-card.warning {
    border: 1px solid var(--warning);
  }

  .prediction-card.critical {
    border: 1px solid var(--error);
  }

  .time-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .value {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .value.warning {
    color: var(--warning);
  }

  .value.critical {
    color: var(--error);
  }

  .unit {
    font-size: 12px;
    font-weight: 400;
    color: var(--text-muted);
  }

  .confidence {
    height: 3px;
    background: var(--bg-secondary);
    border-radius: 2px;
    margin: 10px 0 4px 0;
    overflow: hidden;
  }

  .confidence-bar {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease-out;
  }

  .confidence-label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .alert {
    margin-top: 12px;
    padding: 10px 12px;
    background: rgba(245, 158, 11, 0.1);
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--warning);
  }
</style>
