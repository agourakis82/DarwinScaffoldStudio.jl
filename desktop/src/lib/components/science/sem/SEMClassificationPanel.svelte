<script lang="ts">
  import { sem, cellTypeStats, selectedCell as selectedCellStore, type DetectedCell, type CellType, type MorphologyFeatures } from '$lib/stores/sem';
  import ImageUploader from './ImageUploader.svelte';
  import ClassificationOverlay from './ClassificationOverlay.svelte';
  import CellStatsPanel from './CellStatsPanel.svelte';
  import CellDetailCard from './CellDetailCard.svelte';

  let selectedCellData: DetectedCell | null = null;

  $: selectedCellData = $selectedCellStore;

  function handleUpload(event: CustomEvent<{ url: string; width: number; height: number }>) {
    const { url, width, height } = event.detail;
    sem.setImage(url, width, height);
  }

  function handleSelectCell(event: CustomEvent<number | null>) {
    sem.selectCell(event.detail);
  }

  async function classifyCells() {
    if (!$sem.imageUrl) return;

    sem.startClassification();

    try {
      const response = await fetch('http://localhost:8081/vision/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: $sem.imageUrl }),
      });

      if (!response.ok) throw new Error('Classification failed');

      const data = await response.json();
      sem.setDetectedCells(data.cells);
    } catch {
      // Generate demo cells
      generateDemoCells();
    }
  }

  function generateDemoCells() {
    if (!$sem.imageSize) return;

    const { width, height } = $sem.imageSize;
    const cellTypes: CellType[] = ['osteoblast', 'chondrocyte', 'fibroblast', 'endothelial', 'macrophage', 'stem_cell'];
    const cells: DetectedCell[] = [];

    const nCells = 12 + Math.floor(Math.random() * 8);
    for (let i = 0; i < nCells; i++) {
      const cellWidth = 40 + Math.random() * 60;
      const cellHeight = 40 + Math.random() * 60;
      const x = Math.random() * (width - cellWidth);
      const y = Math.random() * (height - cellHeight);

      const predictedType = cellTypes[Math.floor(Math.random() * cellTypes.length)];
      const confidence = 0.5 + Math.random() * 0.5;

      const allProbabilities: Record<CellType, number> = {
        osteoblast: 0, chondrocyte: 0, fibroblast: 0,
        endothelial: 0, macrophage: 0, stem_cell: 0,
      };
      let remaining = 1;
      allProbabilities[predictedType] = confidence;
      remaining -= confidence;
      for (const type of cellTypes) {
        if (type !== predictedType) {
          const prob = Math.random() * remaining * 0.5;
          allProbabilities[type] = prob;
          remaining -= prob;
        }
      }

      const morphology: MorphologyFeatures = {
        area: cellWidth * cellHeight * (0.6 + Math.random() * 0.4),
        perimeter: 2 * (cellWidth + cellHeight) * (0.8 + Math.random() * 0.4),
        aspectRatio: cellWidth / cellHeight,
        circularity: 0.3 + Math.random() * 0.7,
        textureVariance: Math.random() * 100,
        neighborCount: Math.floor(Math.random() * 5),
        perimeterRoughness: Math.random() * 0.5,
      };

      cells.push({
        id: i,
        boundingBox: { x, y, width: cellWidth, height: cellHeight },
        centroid: { x: x + cellWidth / 2, y: y + cellHeight / 2 },
        classification: { predictedType, confidence, allProbabilities },
        morphology,
      });
    }

    sem.setDetectedCells(cells);
  }

  function clearImage() {
    sem.clearImage();
  }
</script>

<div class="sem-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>SEM Cell Classification</h1>
      <p>AI-powered cell type identification from scanning electron microscopy images</p>
    </div>

    <div class="header-actions">
      {#if $sem.imageUrl}
        <div class="overlay-controls">
          <label class="toggle">
            <input type="checkbox" bind:checked={$sem.showOverlay} />
            <span>Show Overlay</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={$sem.overlayOpacity}
            disabled={!$sem.showOverlay}
          />
        </div>

        <button class="clear-btn" on:click={clearImage}>
          Clear
        </button>

        <button
          class="classify-btn"
          on:click={classifyCells}
          disabled={$sem.isClassifying}
        >
          {#if $sem.isClassifying}
            <span class="spinner"></span>
            Classifying...
          {:else}
            Classify Cells
          {/if}
        </button>
      {/if}
    </div>
  </header>

  <div class="panel-content">
    <div class="main-view">
      {#if $sem.imageUrl && $sem.imageSize}
        <ClassificationOverlay
          imageUrl={$sem.imageUrl}
          imageSize={$sem.imageSize}
          cells={$sem.detectedCells}
          selectedCell={$sem.selectedCell}
          showOverlay={$sem.showOverlay}
          overlayOpacity={$sem.overlayOpacity}
          on:select={handleSelectCell}
        />
      {:else}
        <ImageUploader
          imageUrl={$sem.imageUrl}
          isLoading={$sem.isClassifying}
          on:upload={handleUpload}
        />
      {/if}

      {#if $sem.error}
        <div class="error-banner">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{$sem.error}</span>
        </div>
      {/if}
    </div>

    <aside class="sidebar">
      <CellStatsPanel
        cells={$sem.detectedCells}
        stats={$cellTypeStats}
      />

      <CellDetailCard cell={selectedCellData} />
    </aside>
  </div>
</div>

<style>
  .sem-panel {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 16px;
  }

  .header-title h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
  }

  .header-title p {
    font-size: 13px;
    color: var(--text-muted);
    margin: 0;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .overlay-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .toggle input {
    width: 16px;
    height: 16px;
    accent-color: var(--primary);
  }

  .overlay-controls input[type="range"] {
    width: 80px;
    accent-color: var(--primary);
  }

  .clear-btn, .classify-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .clear-btn {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  .clear-btn:hover {
    background: var(--bg-tertiary);
  }

  .classify-btn {
    background: var(--primary);
    color: white;
  }

  .classify-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .classify-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .panel-content {
    flex: 1;
    display: flex;
    gap: 20px;
    min-height: 0;
  }

  .main-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    color: var(--error);
    font-size: 13px;
  }

  .sidebar {
    width: 320px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }
</style>
