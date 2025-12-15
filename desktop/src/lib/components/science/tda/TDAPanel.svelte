<script lang="ts">
  import { tda, scoreInterconnectivity, type PersistencePoint } from '$lib/stores/tda';
  import { scaffold } from '$lib/stores/scaffold';
  import PersistenceDiagramChart from './PersistenceDiagramChart.svelte';
  import BettiNumbersDisplay from './BettiNumbersDisplay.svelte';
  import TopologyScoreCard from './TopologyScoreCard.svelte';

  let activeTab: 'diagrams' | 'betti' | 'score' = 'diagrams';

  async function runAnalysis() {
    tda.startAnalysis();

    try {
      // Call Julia backend
      const response = await fetch('http://localhost:8081/tda/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scaffoldId: $scaffold.id }),
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();

      // Transform to our format
      const diagrams = {
        H0: data.diagrams.H0.map((p: number[]) => ({
          birth: p[0],
          death: p[1],
          persistence: p[1] - p[0],
        })),
        H1: data.diagrams.H1.map((p: number[]) => ({
          birth: p[0],
          death: p[1],
          persistence: p[1] - p[0],
        })),
        H2: data.diagrams.H2.map((p: number[]) => ({
          birth: p[0],
          death: p[1],
          persistence: p[1] - p[0],
        })),
      };

      const bettiNumbers = {
        beta0: data.betti_numbers.beta0,
        beta1: data.betti_numbers.beta1,
        beta2: data.betti_numbers.beta2,
      };

      const score = data.interconnectivity_score ?? scoreInterconnectivity(bettiNumbers);

      tda.setResults(diagrams, bettiNumbers, score);
    } catch (err) {
      // Use demo data for development
      const demoDiagrams = generateDemoData();
      const demoBetti = { beta0: 1, beta1: 42, beta2: 3 };
      const demoScore = scoreInterconnectivity(demoBetti);

      tda.setResults(demoDiagrams, demoBetti, demoScore);
    }
  }

  function generateDemoData() {
    const H0: PersistencePoint[] = [{ birth: 0, death: 1, persistence: 1 }];
    const H1: PersistencePoint[] = [];
    const H2: PersistencePoint[] = [];

    // Generate realistic persistence points
    for (let i = 0; i < 42; i++) {
      const birth = Math.random() * 0.3;
      const persistence = Math.random() * 0.5 + 0.1;
      H1.push({ birth, death: birth + persistence, persistence });
    }

    for (let i = 0; i < 3; i++) {
      const birth = Math.random() * 0.2 + 0.1;
      const persistence = Math.random() * 0.3 + 0.05;
      H2.push({ birth, death: birth + persistence, persistence });
    }

    return { H0, H1, H2 };
  }
</script>

<div class="tda-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>Topological Data Analysis</h1>
      <p>Persistent homology analysis of scaffold pore structure</p>
    </div>

    <button
      class="analyze-btn"
      on:click={runAnalysis}
      disabled={$tda.isAnalyzing}
    >
      {#if $tda.isAnalyzing}
        <span class="spinner"></span>
        Analyzing...
      {:else}
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
        Run Analysis
      {/if}
    </button>
  </header>

  <div class="tabs">
    <button
      class="tab"
      class:active={activeTab === 'diagrams'}
      on:click={() => activeTab = 'diagrams'}
    >
      Persistence Diagrams
    </button>
    <button
      class="tab"
      class:active={activeTab === 'betti'}
      on:click={() => activeTab = 'betti'}
    >
      Betti Numbers
    </button>
    <button
      class="tab"
      class:active={activeTab === 'score'}
      on:click={() => activeTab = 'score'}
    >
      Score & Recommendations
    </button>
  </div>

  <div class="panel-content">
    {#if activeTab === 'diagrams'}
      <div class="diagrams-grid">
        <PersistenceDiagramChart
          points={$tda.diagrams.H0}
          title="H0 (Components)"
          color="#3b82f6"
        />
        <PersistenceDiagramChart
          points={$tda.diagrams.H1}
          title="H1 (Tunnels)"
          color="#10b981"
        />
        <PersistenceDiagramChart
          points={$tda.diagrams.H2}
          title="H2 (Voids)"
          color="#f59e0b"
        />
      </div>

      <div class="diagram-help">
        <h4>Reading Persistence Diagrams</h4>
        <p>
          Each point represents a topological feature. The x-axis shows when a feature appears (birth),
          and the y-axis shows when it disappears (death). Points far from the diagonal represent
          significant, persistent features. Points near the diagonal are noise.
        </p>
      </div>
    {:else if activeTab === 'betti'}
      <BettiNumbersDisplay bettiNumbers={$tda.bettiNumbers} />
    {:else if activeTab === 'score'}
      <TopologyScoreCard
        score={$tda.interconnectivityScore}
        bettiNumbers={$tda.bettiNumbers}
      />
    {/if}
  </div>

  {#if $tda.lastAnalyzed}
    <footer class="panel-footer">
      Last analyzed: {$tda.lastAnalyzed.toLocaleString()}
    </footer>
  {/if}
</div>

<style>
  .tda-panel {
    background: var(--bg-primary);
    border-radius: 12px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px 24px;
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

  .analyze-btn {
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

  .analyze-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .analyze-btn:disabled {
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

  .tabs {
    display: flex;
    gap: 4px;
    padding: 12px 24px;
    background: var(--bg-secondary);
  }

  .tab {
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: transparent;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .tab:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  .tab.active {
    color: var(--primary);
    background: rgba(74, 158, 255, 0.1);
  }

  .panel-content {
    flex: 1;
    padding: 24px;
    overflow: auto;
  }

  .diagrams-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }

  .diagram-help {
    margin-top: 24px;
    padding: 16px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .diagram-help h4 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
  }

  .diagram-help p {
    font-size: 12px;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.5;
  }

  .panel-footer {
    padding: 12px 24px;
    border-top: 1px solid var(--border-color);
    font-size: 11px;
    color: var(--text-muted);
  }

  @media (max-width: 900px) {
    .diagrams-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 600px) {
    .diagrams-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
