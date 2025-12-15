<script lang="ts">
  import type { DetectedCell, CellType } from '$lib/stores/sem';
  import { CELL_TYPE_INFO, getConfidenceLevel } from '$lib/stores/sem';

  export let cell: DetectedCell | null = null;

  function getProb(type: string): number {
    if (!cell) return 0;
    return cell.classification.allProbabilities[type as CellType] ?? 0;
  }
</script>

<div class="detail-card">
  <h3 class="title">Cell Details</h3>

  {#if cell}
    {@const info = CELL_TYPE_INFO[cell.classification.predictedType]}
    <div class="cell-info">
      <div class="cell-header">
        <span class="cell-id">Cell #{cell.id}</span>
        <span class="cell-type" style="background: {info.color}">{info.label}</span>
      </div>

      <div class="confidence-section">
        <span class="conf-label">Confidence</span>
        <div class="conf-display">
          <div class="conf-bar-bg">
            <div
              class="conf-bar"
              class:high={getConfidenceLevel(cell.classification.confidence) === 'high'}
              class:medium={getConfidenceLevel(cell.classification.confidence) === 'medium'}
              class:low={getConfidenceLevel(cell.classification.confidence) === 'low'}
              style="width: {cell.classification.confidence * 100}%"
            ></div>
          </div>
          <span class="conf-value">{(cell.classification.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div class="probabilities">
        <span class="section-label">All Probabilities</span>
        {#each Object.entries(CELL_TYPE_INFO) as [type, typeInfo]}
          <div class="prob-row">
            <span class="prob-dot" style="background: {typeInfo.color}"></span>
            <span class="prob-name">{typeInfo.label}</span>
            <span class="prob-value">{(getProb(type) * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>

      <div class="morphology">
        <span class="section-label">Morphology Features</span>
        <div class="morph-grid">
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.area.toFixed(0)}</span>
            <span class="morph-label">Area (px^2)</span>
          </div>
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.perimeter.toFixed(0)}</span>
            <span class="morph-label">Perimeter (px)</span>
          </div>
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.aspectRatio.toFixed(2)}</span>
            <span class="morph-label">Aspect Ratio</span>
          </div>
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.circularity.toFixed(2)}</span>
            <span class="morph-label">Circularity</span>
          </div>
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.textureVariance.toFixed(2)}</span>
            <span class="morph-label">Texture Var.</span>
          </div>
          <div class="morph-item">
            <span class="morph-value">{cell.morphology.neighborCount}</span>
            <span class="morph-label">Neighbors</span>
          </div>
        </div>
      </div>

      <div class="location">
        <span class="section-label">Location</span>
        <div class="loc-display">
          <span>Centroid: ({cell.centroid.x.toFixed(0)}, {cell.centroid.y.toFixed(0)})</span>
          <span>Size: {cell.boundingBox.width.toFixed(0)} x {cell.boundingBox.height.toFixed(0)} px</span>
        </div>
      </div>
    </div>
  {:else}
    <div class="no-selection">
      <p>No cell selected</p>
      <p class="hint">Click on a cell in the image to view its details</p>
    </div>
  {/if}
</div>

<style>
  .detail-card {
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

  .cell-info {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .cell-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .cell-id {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .cell-type {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    color: white;
  }

  .confidence-section {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .conf-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .conf-display {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .conf-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
  }

  .conf-bar {
    height: 100%;
    border-radius: 4px;
    transition: width var(--transition-normal);
  }

  .conf-bar.high { background: var(--success); }
  .conf-bar.medium { background: var(--warning); }
  .conf-bar.low { background: var(--error); }

  .conf-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    width: 50px;
    text-align: right;
  }

  .section-label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .probabilities {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .prob-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
  }

  .prob-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .prob-name {
    flex: 1;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .prob-value {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
    font-family: var(--font-mono);
  }

  .morphology {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .morph-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .morph-item {
    text-align: center;
    padding: 8px;
    background: var(--bg-primary);
    border-radius: 6px;
  }

  .morph-value {
    display: block;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .morph-label {
    display: block;
    font-size: 9px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .location {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .loc-display {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .loc-display span {
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }

  .no-selection {
    padding: 32px;
    text-align: center;
    color: var(--text-muted);
  }

  .no-selection p {
    margin: 0;
    font-size: 13px;
  }

  .no-selection .hint {
    font-size: 12px;
    margin-top: 4px;
  }
</style>
