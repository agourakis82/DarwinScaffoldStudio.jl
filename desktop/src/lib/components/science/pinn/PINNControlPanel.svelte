<script lang="ts">
  import { onMount } from 'svelte';
  import { pinn, DEFAULT_BOUNDARY_CONDITIONS, type PINNBoundaryConditions } from '$lib/stores/pinn';
  import ConcentrationHeatmap3D from './ConcentrationHeatmap3D.svelte';
  import TimeSlider from './TimeSlider.svelte';
  import HypoxiaOverlay from './HypoxiaOverlay.svelte';

  let boundaryConditions: PINNBoundaryConditions = { ...DEFAULT_BOUNDARY_CONDITIONS };
  let showHypoxia = true;
  let opacity = 0.6;
  let playInterval: ReturnType<typeof setInterval> | null = null;
  let isPlaying = false;

  async function runSimulation() {
    pinn.startSimulation();

    try {
      const response = await fetch('http://localhost:8081/pinn/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(boundaryConditions),
      });

      if (!response.ok) throw new Error('Simulation failed');

      const data = await response.json();

      pinn.setResults(
        data.simulation_id,
        new Float32Array(data.concentration_field),
        data.dimensions,
        data.time_points,
        data.hypoxic_volume,
        data.hypoxic_regions
      );
    } catch {
      // Generate demo simulation
      generateDemoSimulation();
    }
  }

  function generateDemoSimulation() {
    const nx = 32, ny = 32, nz = 32, nt = 10;
    const timePoints = Array.from({ length: nt }, (_, i) => i * 2.4); // 0-24 hours
    const field = new Float32Array(nx * ny * nz * nt);

    // Generate time-varying concentration field
    for (let t = 0; t < nt; t++) {
      const timeFactor = 1 - (t / nt) * 0.3; // Gradual depletion over time

      for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            const cx = (x - nx/2) / (nx/2);
            const cy = (y - ny/2) / (ny/2);
            const cz = (z - nz/2) / (nz/2);

            const dist = Math.sqrt(cx*cx + cy*cy + cz*cz) / Math.sqrt(3);
            const base = 160 * dist * timeFactor;
            const noise = (Math.random() - 0.5) * 10;
            const value = Math.max(0, Math.min(160, base + noise));

            const idx = t * (nx * ny * nz) + x + y * nx + z * nx * ny;
            field[idx] = value;
          }
        }
      }
    }

    // Calculate hypoxic volume for last time point
    let hypoxicCount = 0;
    const totalVoxels = nx * ny * nz;
    const lastTimeOffset = (nt - 1) * totalVoxels;

    for (let i = 0; i < totalVoxels; i++) {
      if (field[lastTimeOffset + i] < 10) hypoxicCount++;
    }

    const hypoxicVolume = (hypoxicCount / totalVoxels) * 100;

    pinn.setResults(
      'demo-sim',
      field,
      { nx, ny, nz, nt },
      timePoints,
      hypoxicVolume,
      [
        { x: 16, y: 16, z: 16, volume: Math.round(hypoxicCount * 0.6), minO2: 3.2 },
        { x: 10, y: 20, z: 12, volume: Math.round(hypoxicCount * 0.4), minO2: 5.1 },
      ]
    );
  }

  function handleTimeChange(e: CustomEvent<number>) {
    pinn.setTimeIndex(e.detail);
  }

  function handlePlay() {
    isPlaying = true;
    playInterval = setInterval(() => {
      const current = $pinn.currentTimeIndex;
      if (current >= $pinn.dimensions.nt - 1) {
        pinn.setTimeIndex(0);
      } else {
        pinn.setTimeIndex(current + 1);
      }
    }, 500);
  }

  function handlePause() {
    isPlaying = false;
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }

  // Run demo simulation on mount
  onMount(() => {
    if (!$pinn.concentrationField) {
      generateDemoSimulation();
    }
  });
</script>

<div class="pinn-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>PINN Nutrient Transport</h1>
      <p>Physics-informed neural network simulation of oxygen diffusion</p>
    </div>

    <button
      class="run-btn"
      on:click={runSimulation}
      disabled={$pinn.isComputing}
    >
      {#if $pinn.isComputing}
        <span class="spinner"></span>
        Computing...
      {:else}
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Run Simulation
      {/if}
    </button>
  </header>

  <div class="panel-content">
    <div class="main-view">
      <div class="viewer-container">
        <ConcentrationHeatmap3D
          concentrationData={$pinn.concentrationField}
          dimensions={$pinn.dimensions}
          timeIndex={$pinn.currentTimeIndex}
          {showHypoxia}
          {opacity}
        />
      </div>

      <div class="time-controls">
        <TimeSlider
          value={$pinn.currentTimeIndex}
          min={0}
          max={Math.max(0, $pinn.dimensions.nt - 1)}
          timePoints={$pinn.timePoints}
          {isPlaying}
          on:change={handleTimeChange}
          on:play={handlePlay}
          on:pause={handlePause}
        />
      </div>

      <div class="view-options">
        <label class="option">
          <input type="checkbox" bind:checked={showHypoxia} />
          <span>Highlight Hypoxia</span>
        </label>
        <label class="option slider-option">
          <span>Opacity</span>
          <input type="range" min="0.1" max="1" step="0.1" bind:value={opacity} />
          <span class="value">{opacity.toFixed(1)}</span>
        </label>
      </div>
    </div>

    <aside class="sidebar">
      <section class="boundary-section">
        <h3>Boundary Conditions</h3>

        <div class="input-group">
          <label>O2 Supply (mmHg)</label>
          <input type="number" bind:value={boundaryConditions.oxygenSupply} min="0" max="200" />
        </div>

        <div class="input-group">
          <label>Consumption Rate (mmHg/hr)</label>
          <input type="number" bind:value={boundaryConditions.consumptionRate} min="0" max="2" step="0.1" />
        </div>

        <div class="input-group">
          <label>Diffusion Coeff. (mm^2/hr)</label>
          <input type="number" bind:value={boundaryConditions.diffusionCoeff} min="0" max="0.01" step="0.0001" />
        </div>

        <div class="input-group">
          <label>Porosity</label>
          <input type="number" bind:value={boundaryConditions.scaffoldPorosity} min="0" max="1" step="0.01" />
        </div>
      </section>

      <section class="hypoxia-section">
        <HypoxiaOverlay
          hypoxicVolume={$pinn.hypoxicVolume}
          hypoxicRegions={$pinn.hypoxicRegions}
          minOxygen={$pinn.minOxygen}
        />
      </section>

      <section class="legend-section">
        <h3>Color Scale</h3>
        <div class="color-legend">
          <div class="legend-gradient"></div>
          <div class="legend-labels">
            <span>0</span>
            <span>{($pinn.maxOxygen / 2).toFixed(0)}</span>
            <span>{$pinn.maxOxygen.toFixed(0)} mmHg</span>
          </div>
        </div>
      </section>
    </aside>
  </div>
</div>

<style>
  .pinn-panel {
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

  .run-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .run-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .run-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
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
    min-width: 0;
  }

  .viewer-container {
    flex: 1;
    border-radius: 12px;
    overflow: hidden;
    min-height: 400px;
  }

  .time-controls {
    flex-shrink: 0;
  }

  .view-options {
    display: flex;
    gap: 20px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .option {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .slider-option {
    flex: 1;
  }

  .slider-option input[type="range"] {
    flex: 1;
    margin: 0 8px;
  }

  .option .value {
    min-width: 30px;
    text-align: right;
    color: var(--text-muted);
  }

  .sidebar {
    width: 300px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }

  .boundary-section, .legend-section {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .boundary-section h3, .legend-section h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .input-group {
    margin-bottom: 12px;
  }

  .input-group label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .input-group input {
    width: 100%;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .input-group input:focus {
    outline: none;
    border-color: var(--primary);
  }

  .hypoxia-section {
    flex: 1;
    min-height: 200px;
  }

  .color-legend {
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .legend-gradient {
    height: 16px;
    border-radius: 4px;
    background: linear-gradient(to right,
      rgb(139, 0, 0) 0%,
      rgb(255, 0, 0) 6%,
      rgb(0, 128, 255) 20%,
      rgb(0, 255, 128) 50%,
      rgb(128, 255, 0) 75%,
      rgb(255, 255, 0) 100%
    );
  }

  .legend-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-muted);
  }
</style>
