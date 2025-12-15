<script lang="ts">
  import type { GNNPrediction } from '$lib/stores/gnn';

  export let predictions: GNNPrediction[] = [];
  export let isLoading: boolean = false;
</script>

<div class="prediction-card">
  <h3 class="title">Property Predictions</h3>

  {#if isLoading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Running GNN inference...</span>
    </div>
  {:else if predictions.length > 0}
    <div class="predictions-list">
      {#each predictions as prediction}
        <div class="prediction-item">
          <div class="property-info">
            <span class="property-name">{prediction.property}</span>
            {#if prediction.confidence !== undefined}
              <span class="confidence" class:high={prediction.confidence >= 0.8} class:medium={prediction.confidence >= 0.5 && prediction.confidence < 0.8} class:low={prediction.confidence < 0.5}>
                {(prediction.confidence * 100).toFixed(0)}% conf.
              </span>
            {/if}
          </div>
          <div class="property-value">
            <span class="value">{prediction.value.toFixed(3)}</span>
            <span class="unit">{prediction.unit}</span>
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="no-predictions">
      <p>No predictions yet</p>
      <p class="hint">Click "Predict Properties" to run GNN inference</p>
    </div>
  {/if}
</div>

<style>
  .prediction-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .loading {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 24px;
    color: var(--text-muted);
    font-size: 13px;
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-color);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .predictions-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .prediction-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .property-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .property-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .confidence {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 4px;
    width: fit-content;
  }

  .confidence.high {
    background: rgba(16, 185, 129, 0.15);
    color: var(--success);
  }

  .confidence.medium {
    background: rgba(245, 158, 11, 0.15);
    color: var(--warning);
  }

  .confidence.low {
    background: rgba(239, 68, 68, 0.15);
    color: var(--error);
  }

  .property-value {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .value {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .unit {
    font-size: 12px;
    color: var(--text-muted);
  }

  .no-predictions {
    padding: 24px;
    text-align: center;
    color: var(--text-muted);
  }

  .no-predictions p {
    margin: 0;
    font-size: 13px;
  }

  .no-predictions .hint {
    font-size: 12px;
    margin-top: 4px;
  }
</style>
