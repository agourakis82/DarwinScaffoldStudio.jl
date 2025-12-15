<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    tissueGrowth,
    DEFAULT_CONFIG,
    CellType,
    CELL_COLORS,
    calculateGrowthRate,
    type SimulationConfig,
    type Cell,
    type PopulationSnapshot
  } from '$lib/stores/tissueGrowth';
  import CellVisualization3D from './CellVisualization3D.svelte';
  import GrowthTimeline from './GrowthTimeline.svelte';
  import CellPopulationChart from './CellPopulationChart.svelte';
  import DifferentiationTree from './DifferentiationTree.svelte';

  let config: SimulationConfig = { ...DEFAULT_CONFIG };
  let playInterval: ReturnType<typeof setInterval> | null = null;

  onDestroy(() => {
    if (playInterval) clearInterval(playInterval);
  });

  async function runSimulation() {
    tissueGrowth.startSimulation();

    try {
      const response = await fetch('http://localhost:8081/growth/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!response.ok) throw new Error('Simulation failed');

      const data = await response.json();

      tissueGrowth.setResults(
        data.simulation_id,
        data.cells_over_time,
        data.population_history
      );
    } catch {
      // Generate demo simulation
      generateDemoSimulation();
    }
  }

  function generateDemoSimulation() {
    const { initialCellCount, simulationHours, divisionRate, differentiationRate } = config;

    const cellsOverTime: Cell[][] = [];
    const populationHistory: PopulationSnapshot[] = [];

    let currentCells: Cell[] = [];

    // Initialize with MSCs
    for (let i = 0; i < initialCellCount; i++) {
      currentCells.push({
        id: i,
        type: CellType.MSC,
        x: Math.random() * 3,
        y: Math.random() * 3,
        z: Math.random() * 3,
        age: 0,
        health: 0.9 + Math.random() * 0.1,
        divisionReady: false,
      });
    }

    // Simulate each hour
    for (let t = 0; t <= simulationHours; t++) {
      // Copy current state
      cellsOverTime.push([...currentCells]);

      // Count by type
      const counts: Record<CellType, number> = {
        [CellType.MSC]: 0,
        [CellType.PREOSTEOBLAST]: 0,
        [CellType.OSTEOBLAST]: 0,
        [CellType.OSTEOCYTE]: 0,
        [CellType.DEAD]: 0,
      };

      for (const cell of currentCells) {
        counts[cell.type]++;
      }

      populationHistory.push({
        time: t,
        counts,
        totalCells: currentCells.length,
      });

      // Skip evolution on last step
      if (t >= simulationHours) break;

      // Evolve cells
      const newCells: Cell[] = [];
      let nextId = currentCells.length > 0 ? Math.max(...currentCells.map(c => c.id)) + 1 : 0;

      for (const cell of currentCells) {
        // Age
        cell.age++;

        // Division (simplified)
        if (Math.random() < divisionRate && cell.type !== CellType.OSTEOCYTE && currentCells.length < 10000) {
          newCells.push({
            id: nextId++,
            type: cell.type,
            x: cell.x + (Math.random() - 0.5) * 0.1,
            y: cell.y + (Math.random() - 0.5) * 0.1,
            z: cell.z + (Math.random() - 0.5) * 0.1,
            age: 0,
            health: cell.health,
            divisionReady: false,
          });
        }

        // Differentiation
        if (Math.random() < differentiationRate) {
          const nextType = getNextDifferentiation(cell.type);
          if (nextType) cell.type = nextType;
        }

        // Movement
        cell.x = Math.max(0, Math.min(3, cell.x + (Math.random() - 0.5) * 0.05));
        cell.y = Math.max(0, Math.min(3, cell.y + (Math.random() - 0.5) * 0.05));
        cell.z = Math.max(0, Math.min(3, cell.z + (Math.random() - 0.5) * 0.05));
      }

      currentCells.push(...newCells);
    }

    tissueGrowth.setResults('demo-growth', cellsOverTime, populationHistory);
  }

  function getNextDifferentiation(type: CellType): CellType | null {
    const pathway: Record<CellType, CellType | null> = {
      [CellType.MSC]: CellType.PREOSTEOBLAST,
      [CellType.PREOSTEOBLAST]: CellType.OSTEOBLAST,
      [CellType.OSTEOBLAST]: CellType.OSTEOCYTE,
      [CellType.OSTEOCYTE]: null,
      [CellType.DEAD]: null,
    };
    return pathway[type];
  }

  function handleTimeChange(e: CustomEvent<number>) {
    tissueGrowth.setTimeIndex(e.detail);
  }

  function handlePlay() {
    tissueGrowth.play();
    playInterval = setInterval(() => {
      if ($tissueGrowth.currentTimeIndex >= $tissueGrowth.maxTime) {
        handlePause();
        tissueGrowth.setTimeIndex(0);
      }
    }, 100);
  }

  function handlePause() {
    tissueGrowth.pause();
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }

  function handleSpeedChange(e: CustomEvent<number>) {
    tissueGrowth.setPlaybackSpeed(e.detail);
  }

  $: currentSnapshot = $tissueGrowth.populationHistory[$tissueGrowth.currentTimeIndex];
  $: growthRate = calculateGrowthRate($tissueGrowth.populationHistory);

  // Auto-run demo on mount
  onMount(() => {
    if ($tissueGrowth.populationHistory.length === 0) {
      generateDemoSimulation();
    }
  });
</script>

<div class="growth-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>Tissue Growth Simulation</h1>
      <p>Agent-based model of cell proliferation and osteogenic differentiation</p>
    </div>

    <button
      class="run-btn"
      on:click={runSimulation}
      disabled={$tissueGrowth.isSimulating}
    >
      {#if $tissueGrowth.isSimulating}
        <span class="spinner"></span>
        Simulating...
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
        <CellVisualization3D cells={$tissueGrowth.cells} />
      </div>

      <GrowthTimeline
        currentTime={currentSnapshot?.time ?? 0}
        maxTime={$tissueGrowth.maxTime}
        isPlaying={$tissueGrowth.isPlaying}
        playbackSpeed={$tissueGrowth.playbackSpeed}
        on:timeChange={handleTimeChange}
        on:play={handlePlay}
        on:pause={handlePause}
        on:speedChange={handleSpeedChange}
      />
    </div>

    <aside class="sidebar">
      <section class="config-section">
        <h3>Simulation Parameters</h3>

        <div class="input-group">
          <label>Initial Cells</label>
          <input type="number" bind:value={config.initialCellCount} min="100" max="5000" step="100" />
        </div>

        <div class="input-group">
          <label>Duration (hours)</label>
          <input type="number" bind:value={config.simulationHours} min="24" max="336" step="24" />
        </div>

        <div class="input-group">
          <label>Division Rate (/hr)</label>
          <input type="number" bind:value={config.divisionRate} min="0" max="0.2" step="0.01" />
        </div>

        <div class="input-group">
          <label>Differentiation Rate (/hr)</label>
          <input type="number" bind:value={config.differentiationRate} min="0" max="0.1" step="0.005" />
        </div>
      </section>

      <section class="metrics-section">
        <h3>Growth Metrics</h3>
        <div class="metric">
          <span class="metric-label">Total Cells</span>
          <span class="metric-value">{currentSnapshot?.totalCells.toLocaleString() ?? 0}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Growth Rate</span>
          <span class="metric-value">{(growthRate * 100).toFixed(2)}%/hr</span>
        </div>
      </section>

      <DifferentiationTree {currentSnapshot} />
    </aside>
  </div>

  <div class="chart-section">
    <CellPopulationChart
      history={$tissueGrowth.populationHistory}
      currentTime={currentSnapshot?.time ?? 0}
      width={800}
      height={200}
    />
  </div>
</div>

<style>
  .growth-panel {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
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
    min-height: 350px;
  }

  .sidebar {
    width: 280px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }

  .config-section, .metrics-section {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .config-section h3, .metrics-section h3 {
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

  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .metric:last-child {
    border-bottom: none;
  }

  .metric-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .chart-section {
    flex-shrink: 0;
  }
</style>
