<script lang="ts">
  import { CellType, CELL_COLORS, differentiationProgress, type PopulationSnapshot } from '$lib/stores/tissueGrowth';

  export let currentSnapshot: PopulationSnapshot | null = null;

  $: progress = currentSnapshot ? differentiationProgress(currentSnapshot) * 100 : 0;

  const stages = [
    { type: CellType.MSC, name: 'Stem Cell', desc: 'Mesenchymal' },
    { type: CellType.PREOSTEOBLAST, name: 'Pre-Osteoblast', desc: 'Committed' },
    { type: CellType.OSTEOBLAST, name: 'Osteoblast', desc: 'Matrix-producing' },
    { type: CellType.OSTEOCYTE, name: 'Osteocyte', desc: 'Mature bone cell' },
  ];

  function getCount(type: CellType): number {
    return currentSnapshot?.counts[type] || 0;
  }

  function getPercentage(type: CellType): number {
    if (!currentSnapshot || currentSnapshot.totalCells === 0) return 0;
    return (getCount(type) / currentSnapshot.totalCells) * 100;
  }
</script>

<div class="differentiation-tree">
  <h3 class="title">Differentiation Pathway</h3>

  <div class="progress-bar">
    <div class="progress-fill" style="width: {progress}%"></div>
    <span class="progress-label">{progress.toFixed(0)}% Differentiated</span>
  </div>

  <div class="pathway">
    {#each stages as stage, i}
      {@const count = getCount(stage.type)}
      {@const pct = getPercentage(stage.type)}
      <div class="stage" class:active={pct > 0}>
        <div class="stage-icon" style="border-color: {CELL_COLORS[stage.type]}">
          <div class="icon-fill" style="background: {CELL_COLORS[stage.type]}; height: {Math.min(100, pct)}%"></div>
        </div>
        <div class="stage-info">
          <span class="stage-name">{stage.name}</span>
          <span class="stage-desc">{stage.desc}</span>
          <span class="stage-count">{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
        </div>
        {#if i < stages.length - 1}
          <div class="arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>
        {/if}
      </div>
    {/each}
  </div>

  <div class="markers">
    <h4>Key Markers</h4>
    <div class="marker-list">
      <div class="marker">
        <span class="marker-dot" style="background: {CELL_COLORS[CellType.MSC]}"></span>
        <span class="marker-label">CD73+, CD90+, CD105+</span>
      </div>
      <div class="marker">
        <span class="marker-dot" style="background: {CELL_COLORS[CellType.PREOSTEOBLAST]}"></span>
        <span class="marker-label">RUNX2+, ALP+</span>
      </div>
      <div class="marker">
        <span class="marker-dot" style="background: {CELL_COLORS[CellType.OSTEOBLAST]}"></span>
        <span class="marker-label">OSX+, COL1A1+</span>
      </div>
      <div class="marker">
        <span class="marker-dot" style="background: {CELL_COLORS[CellType.OSTEOCYTE]}"></span>
        <span class="marker-label">SOST+, DMP1+</span>
      </div>
    </div>
  </div>
</div>

<style>
  .differentiation-tree {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .progress-bar {
    position: relative;
    height: 24px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981, #f59e0b);
    border-radius: 12px;
    transition: width 0.5s ease-out;
  }

  .progress-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .pathway {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 20px;
  }

  .stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    opacity: 0.5;
    transition: opacity var(--transition-fast);
  }

  .stage.active {
    opacity: 1;
  }

  .stage-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    border: 3px solid var(--border-color);
    overflow: hidden;
    display: flex;
    align-items: flex-end;
    margin-bottom: 8px;
  }

  .icon-fill {
    width: 100%;
    transition: height 0.5s ease-out;
  }

  .stage-info {
    text-align: center;
  }

  .stage-name {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stage-desc {
    display: block;
    font-size: 9px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .stage-count {
    display: block;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .arrow {
    position: absolute;
    right: -16px;
    top: 16px;
    color: var(--border-color);
  }

  .markers {
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
  }

  .markers h4 {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 10px 0;
  }

  .marker-list {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .marker {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .marker-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .marker-label {
    font-size: 10px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
</style>
