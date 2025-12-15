<script lang="ts">
  import type { CellType, DetectedCell } from '$lib/stores/sem';
  import { CELL_TYPE_INFO, getConfidenceLevel } from '$lib/stores/sem';

  export let cells: DetectedCell[] = [];
  export let stats: Record<CellType, number>;

  $: totalCells = cells.length;
  $: avgConfidence = cells.length > 0
    ? cells.reduce((sum, c) => sum + c.classification.confidence, 0) / cells.length
    : 0;
  $: sortedTypes = Object.entries(stats)
    .filter(([_, count]) => count > 0)
    .sort((a, b) => b[1] - a[1]) as [CellType, number][];
</script>

<div class="stats-panel">
  <h3 class="title">Cell Classification Results</h3>

  {#if totalCells > 0}
    <div class="summary">
      <div class="summary-item">
        <span class="summary-value">{totalCells}</span>
        <span class="summary-label">Cells Detected</span>
      </div>
      <div class="summary-item">
        <span class="summary-value" class:high={getConfidenceLevel(avgConfidence) === 'high'} class:medium={getConfidenceLevel(avgConfidence) === 'medium'} class:low={getConfidenceLevel(avgConfidence) === 'low'}>
          {(avgConfidence * 100).toFixed(1)}%
        </span>
        <span class="summary-label">Avg. Confidence</span>
      </div>
    </div>

    <div class="type-breakdown">
      <h4>Cell Type Distribution</h4>
      {#each sortedTypes as [type, count]}
        {@const info = CELL_TYPE_INFO[type]}
        {@const percentage = (count / totalCells) * 100}
        <div class="type-row">
          <div class="type-info">
            <span class="type-dot" style="background: {info.color}"></span>
            <span class="type-name">{info.label}</span>
            <span class="type-count">{count}</span>
          </div>
          <div class="type-bar-container">
            <div class="type-bar" style="width: {percentage}%; background: {info.color}"></div>
          </div>
          <span class="type-percentage">{percentage.toFixed(1)}%</span>
        </div>
      {/each}
    </div>

    <div class="confidence-breakdown">
      <h4>Confidence Levels</h4>
      {#if totalCells > 0}
        {@const highConf = cells.filter(c => getConfidenceLevel(c.classification.confidence) === 'high').length}
        {@const medConf = cells.filter(c => getConfidenceLevel(c.classification.confidence) === 'medium').length}
        {@const lowConf = cells.filter(c => getConfidenceLevel(c.classification.confidence) === 'low').length}
        <div class="confidence-bars">
        <div class="conf-item">
          <div class="conf-bar high" style="height: {(highConf / totalCells) * 100}%"></div>
          <span class="conf-count">{highConf}</span>
          <span class="conf-label">High</span>
        </div>
        <div class="conf-item">
          <div class="conf-bar medium" style="height: {(medConf / totalCells) * 100}%"></div>
          <span class="conf-count">{medConf}</span>
          <span class="conf-label">Medium</span>
        </div>
        <div class="conf-item">
          <div class="conf-bar low" style="height: {(lowConf / totalCells) * 100}%"></div>
          <span class="conf-count">{lowConf}</span>
          <span class="conf-label">Low</span>
        </div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="no-data">
      <p>No cells classified yet</p>
      <p class="hint">Upload an SEM image and click "Classify Cells"</p>
    </div>
  {/if}
</div>

<style>
  .stats-panel {
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

  .summary {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
  }

  .summary-item {
    flex: 1;
    text-align: center;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 10px;
  }

  .summary-value {
    display: block;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .summary-value.high { color: var(--success); }
  .summary-value.medium { color: var(--warning); }
  .summary-value.low { color: var(--error); }

  .summary-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .type-breakdown, .confidence-breakdown {
    margin-bottom: 20px;
  }

  .type-breakdown h4, .confidence-breakdown h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 12px 0;
  }

  .type-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }

  .type-info {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 120px;
  }

  .type-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .type-name {
    font-size: 12px;
    color: var(--text-secondary);
    flex: 1;
  }

  .type-count {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .type-bar-container {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .type-bar {
    height: 100%;
    border-radius: 4px;
    transition: width var(--transition-normal);
  }

  .type-percentage {
    width: 50px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
    text-align: right;
  }

  .confidence-bars {
    display: flex;
    justify-content: center;
    gap: 32px;
    height: 120px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 10px;
  }

  .conf-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    width: 48px;
  }

  .conf-bar {
    width: 32px;
    min-height: 4px;
    border-radius: 4px 4px 0 0;
    transition: height var(--transition-normal);
  }

  .conf-bar.high { background: var(--success); }
  .conf-bar.medium { background: var(--warning); }
  .conf-bar.low { background: var(--error); }

  .conf-count {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 8px;
  }

  .conf-label {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .no-data {
    padding: 32px;
    text-align: center;
    color: var(--text-muted);
  }

  .no-data p {
    margin: 0;
    font-size: 13px;
  }

  .no-data .hint {
    font-size: 12px;
    margin-top: 4px;
  }
</style>
