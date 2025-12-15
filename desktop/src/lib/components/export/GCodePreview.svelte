<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { juliaApi } from '$lib/services/julia-api';

  export let gcodeId: string | null = null;
  export let totalLayers: number = 0;

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let currentLayer = 0;
  let isLoading = false;
  let paths: Array<{ type: 'travel' | 'extrude'; points: [number, number][] }> = [];
  let error: string | null = null;

  // Viewport
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  // Colors
  const COLORS = {
    background: '#1a1a2e',
    grid: '#2a2a4a',
    travel: '#4a4a6a',
    extrude: '#4a9eff',
    extrudeHighlight: '#00d4aa',
  };

  onMount(() => {
    ctx = canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    if (gcodeId && totalLayers > 0) {
      loadLayer(0);
    }
  });

  onDestroy(() => {
    window.removeEventListener('resize', resizeCanvas);
  });

  function resizeCanvas() {
    const rect = canvas.parentElement?.getBoundingClientRect();
    if (rect) {
      canvas.width = rect.width;
      canvas.height = rect.height;
      draw();
    }
  }

  async function loadLayer(layer: number) {
    if (!gcodeId || layer < 0 || layer >= totalLayers) return;

    isLoading = true;
    error = null;

    try {
      const result = await juliaApi.getGCodePreview(gcodeId, layer);
      if (result.success && result.data) {
        paths = result.data.paths;
        currentLayer = layer;
        centerView();
        draw();
      } else {
        error = result.error || 'Failed to load layer';
      }
    } catch (e) {
      error = `Error: ${e}`;
    }

    isLoading = false;
  }

  function centerView() {
    if (paths.length === 0) return;

    // Find bounds
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const path of paths) {
      for (const [x, y] of path.points) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }

    const width = maxX - minX;
    const height = maxY - minY;
    const padding = 40;

    scale = Math.min(
      (canvas.width - padding * 2) / width,
      (canvas.height - padding * 2) / height
    );

    offsetX = canvas.width / 2 - (minX + width / 2) * scale;
    offsetY = canvas.height / 2 - (minY + height / 2) * scale;
  }

  function draw() {
    if (!ctx) return;

    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid();

    // Draw paths
    for (const path of paths) {
      if (path.points.length < 2) continue;

      ctx.beginPath();
      ctx.strokeStyle = path.type === 'travel' ? COLORS.travel : COLORS.extrude;
      ctx.lineWidth = path.type === 'travel' ? 0.5 : 2;

      const [startX, startY] = path.points[0];
      ctx.moveTo(startX * scale + offsetX, startY * scale + offsetY);

      for (let i = 1; i < path.points.length; i++) {
        const [x, y] = path.points[i];
        ctx.lineTo(x * scale + offsetX, y * scale + offsetY);
      }

      ctx.stroke();
    }

    // Draw layer info
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.fillText(`Layer ${currentLayer + 1} / ${totalLayers}`, 10, 20);
  }

  function drawGrid() {
    if (!ctx) return;

    const gridSize = 10 * scale;
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;

    // Vertical lines
    for (let x = offsetX % gridSize; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = offsetY % gridSize; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const mouseX = e.offsetX;
    const mouseY = e.offsetY;

    // Zoom towards mouse position
    offsetX = mouseX - (mouseX - offsetX) * zoomFactor;
    offsetY = mouseY - (mouseY - offsetY) * zoomFactor;
    scale *= zoomFactor;

    draw();
  }

  function handleMouseDown(e: MouseEvent) {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging) return;
    offsetX += e.clientX - lastMouseX;
    offsetY += e.clientY - lastMouseY;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    draw();
  }

  function handleMouseUp() {
    isDragging = false;
  }

  function prevLayer() {
    if (currentLayer > 0) loadLayer(currentLayer - 1);
  }

  function nextLayer() {
    if (currentLayer < totalLayers - 1) loadLayer(currentLayer + 1);
  }

  function handleSliderChange(e: Event) {
    const target = e.target as HTMLInputElement;
    loadLayer(parseInt(target.value));
  }

  $: if (gcodeId && totalLayers > 0) {
    loadLayer(0);
  }
</script>

<div class="gcode-preview">
  <div class="preview-header">
    <h4>G-Code Preview</h4>
    <div class="layer-controls">
      <button class="nav-btn" on:click={prevLayer} disabled={currentLayer === 0}>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="15 18 9 12 15 6"></polyline>
        </svg>
      </button>
      <span class="layer-info">Layer {currentLayer + 1} / {totalLayers}</span>
      <button class="nav-btn" on:click={nextLayer} disabled={currentLayer >= totalLayers - 1}>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="9 18 15 12 9 6"></polyline>
        </svg>
      </button>
    </div>
  </div>

  <div class="canvas-container">
    <canvas
      bind:this={canvas}
      on:wheel={handleWheel}
      on:mousedown={handleMouseDown}
      on:mousemove={handleMouseMove}
      on:mouseup={handleMouseUp}
      on:mouseleave={handleMouseUp}
    />

    {#if isLoading}
      <div class="loading-overlay">
        <div class="spinner"></div>
        <span>Loading layer...</span>
      </div>
    {/if}

    {#if error}
      <div class="error-overlay">
        <span>{error}</span>
      </div>
    {/if}

    {#if !gcodeId}
      <div class="empty-state">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        <span>Generate G-code to preview layers</span>
      </div>
    {/if}
  </div>

  {#if totalLayers > 0}
    <div class="layer-slider">
      <input
        type="range"
        min="0"
        max={totalLayers - 1}
        value={currentLayer}
        on:input={handleSliderChange}
      />
    </div>
  {/if}

  <div class="legend">
    <div class="legend-item">
      <span class="legend-color travel"></span>
      <span>Travel</span>
    </div>
    <div class="legend-item">
      <span class="legend-color extrude"></span>
      <span>Extrude</span>
    </div>
  </div>
</div>

<style>
  .gcode-preview {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary);
    border-radius: 12px;
    overflow: hidden;
  }

  .preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .preview-header h4 {
    font-size: 13px;
    font-weight: 600;
    margin: 0;
  }

  .layer-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .nav-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .nav-btn:hover:not(:disabled) {
    border-color: var(--primary);
    color: var(--primary);
  }

  .nav-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .layer-info {
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
    min-width: 100px;
    text-align: center;
  }

  .canvas-container {
    flex: 1;
    position: relative;
    min-height: 200px;
  }

  canvas {
    width: 100%;
    height: 100%;
    cursor: grab;
  }

  canvas:active {
    cursor: grabbing;
  }

  .loading-overlay,
  .error-overlay,
  .empty-state {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: rgba(13, 17, 23, 0.9);
  }

  .loading-overlay span,
  .error-overlay span,
  .empty-state span {
    font-size: 13px;
    color: var(--text-muted);
  }

  .error-overlay span {
    color: var(--error);
  }

  .empty-state svg {
    color: var(--text-muted);
    opacity: 0.5;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-color);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .layer-slider {
    padding: 8px 16px;
  }

  .layer-slider input {
    width: 100%;
    height: 4px;
    appearance: none;
    background: var(--bg-tertiary);
    border-radius: 2px;
    cursor: pointer;
  }

  .layer-slider input::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    background: var(--primary);
    border-radius: 50%;
    cursor: pointer;
    transition: transform var(--transition-fast);
  }

  .layer-slider input::-webkit-slider-thumb:hover {
    transform: scale(1.2);
  }

  .legend {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    border-top: 1px solid var(--border-color);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .legend-color {
    width: 12px;
    height: 3px;
    border-radius: 1px;
  }

  .legend-color.travel {
    background: #4a4a6a;
  }

  .legend-color.extrude {
    background: #4a9eff;
  }
</style>
