<script lang="ts">
  import { onMount } from 'svelte';
  import { causal, type CausalGraph, type CausalNode, type CausalEdge, type CausalEffect } from '$lib/stores/causal';
  import CausalDAGViewer from './CausalDAGViewer.svelte';
  import EffectEstimateCard from './EffectEstimateCard.svelte';

  let selectedTreatment: string = '';
  let selectedOutcome: string = '';

  async function discoverGraph() {
    causal.startDiscovery();

    try {
      const response = await fetch('http://localhost:8081/causal/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: $causal.algorithm }),
      });

      if (!response.ok) throw new Error('Discovery failed');

      const data = await response.json();
      causal.setGraph(data.graph);
    } catch {
      // Generate demo graph
      generateDemoGraph();
    }
  }

  function generateDemoGraph() {
    const nodes: CausalNode[] = [
      { id: 'porosity', label: 'Porosity', type: 'treatment' },
      { id: 'pore_size', label: 'Pore Size', type: 'variable' },
      { id: 'surface_area', label: 'Surface Area', type: 'mediator' },
      { id: 'cell_attachment', label: 'Cell Attachment', type: 'mediator' },
      { id: 'material', label: 'Material', type: 'confounder' },
      { id: 'bone_formation', label: 'Bone Formation', type: 'outcome' },
    ];

    const edges: CausalEdge[] = [
      { source: 'porosity', target: 'pore_size', type: 'directed' },
      { source: 'porosity', target: 'surface_area', type: 'directed' },
      { source: 'pore_size', target: 'cell_attachment', type: 'directed' },
      { source: 'surface_area', target: 'cell_attachment', type: 'directed' },
      { source: 'cell_attachment', target: 'bone_formation', type: 'directed' },
      { source: 'material', target: 'porosity', type: 'directed' },
      { source: 'material', target: 'bone_formation', type: 'directed' },
    ];

    causal.setGraph({ nodes, edges });
    selectedTreatment = 'porosity';
    selectedOutcome = 'bone_formation';
  }

  async function estimateEffect() {
    if (!selectedTreatment || !selectedOutcome) return;

    causal.setTreatmentOutcome(selectedTreatment, selectedOutcome);
    causal.startEstimation();

    try {
      const response = await fetch('http://localhost:8081/causal/effect/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          treatment: selectedTreatment,
          outcome: selectedOutcome,
          method: 'backdoor',
        }),
      });

      if (!response.ok) throw new Error('Estimation failed');

      const data = await response.json();
      causal.setEffect(data.effect, data.adjustment_set);
    } catch {
      // Generate demo effect
      const demoEffect: CausalEffect = {
        ate: 0.342,
        standardError: 0.078,
        ciLower: 0.189,
        ciUpper: 0.495,
        method: 'Backdoor Adjustment',
        pValue: 0.0001,
      };
      causal.setEffect(demoEffect, ['material']);
    }
  }

  onMount(() => {
    if (!$causal.graph) {
      generateDemoGraph();
    }
  });
</script>

<div class="causal-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>Causal Discovery</h1>
      <p>Discover causal relationships and estimate treatment effects</p>
    </div>

    <div class="header-actions">
      <select class="algorithm-select" bind:value={$causal.algorithm}>
        <option value="pc">PC Algorithm</option>
        <option value="fci">FCI (Latent Confounders)</option>
        <option value="notears">NOTEARS</option>
        <option value="ges">GES (Score-based)</option>
      </select>

      <button
        class="discover-btn"
        on:click={discoverGraph}
        disabled={$causal.isDiscovering}
      >
        {#if $causal.isDiscovering}
          <span class="spinner"></span>
          Discovering...
        {:else}
          Discover DAG
        {/if}
      </button>
    </div>
  </header>

  <div class="panel-content">
    <div class="main-view">
      <CausalDAGViewer
        graph={$causal.graph}
        treatment={selectedTreatment}
        outcome={selectedOutcome}
        adjustmentSet={$causal.adjustmentSet}
        width={700}
        height={450}
      />

      <div class="variable-selection">
        <div class="select-group">
          <label>Treatment Variable</label>
          <select bind:value={selectedTreatment}>
            <option value="">Select treatment...</option>
            {#if $causal.graph}
              {#each $causal.graph.nodes as node}
                <option value={node.id}>{node.label || node.id}</option>
              {/each}
            {/if}
          </select>
        </div>

        <div class="arrow">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        </div>

        <div class="select-group">
          <label>Outcome Variable</label>
          <select bind:value={selectedOutcome}>
            <option value="">Select outcome...</option>
            {#if $causal.graph}
              {#each $causal.graph.nodes as node}
                <option value={node.id}>{node.label || node.id}</option>
              {/each}
            {/if}
          </select>
        </div>

        <button
          class="estimate-btn"
          on:click={estimateEffect}
          disabled={!selectedTreatment || !selectedOutcome || $causal.isEstimating}
        >
          {#if $causal.isEstimating}
            <span class="spinner"></span>
          {:else}
            Estimate Effect
          {/if}
        </button>
      </div>
    </div>

    <aside class="sidebar">
      <EffectEstimateCard
        effect={$causal.effect}
        treatment={selectedTreatment}
        outcome={selectedOutcome}
      />

      {#if $causal.adjustmentSet.length > 0}
        <div class="adjustment-set">
          <h3>Adjustment Set</h3>
          <p class="adjustment-desc">Variables controlled for in the analysis:</p>
          <div class="adjustment-vars">
            {#each $causal.adjustmentSet as variable}
              <span class="var-tag">{variable}</span>
            {/each}
          </div>
        </div>
      {/if}

      <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
          <div class="legend-item">
            <span class="legend-dot" style="background: #10b981"></span>
            <span>Treatment</span>
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: #f59e0b"></span>
            <span>Outcome</span>
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: #ef4444"></span>
            <span>Confounder</span>
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: #8b5cf6"></span>
            <span>Mediator</span>
          </div>
          <div class="legend-item">
            <span class="legend-line dashed"></span>
            <span>Latent Confounder</span>
          </div>
        </div>
      </div>
    </aside>
  </div>
</div>

<style>
  .causal-panel {
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
    gap: 12px;
  }

  .algorithm-select {
    padding: 10px 14px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .discover-btn, .estimate-btn {
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

  .discover-btn:hover:not(:disabled), .estimate-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .discover-btn:disabled, .estimate-btn:disabled {
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
    gap: 16px;
  }

  .variable-selection {
    display: flex;
    align-items: flex-end;
    gap: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border-radius: 12px;
  }

  .select-group {
    flex: 1;
  }

  .select-group label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  .select-group select {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .arrow {
    color: var(--text-muted);
    padding-bottom: 8px;
  }

  .sidebar {
    width: 300px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }

  .adjustment-set, .legend {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .adjustment-set h3, .legend h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
  }

  .adjustment-desc {
    font-size: 12px;
    color: var(--text-muted);
    margin: 0 0 12px 0;
  }

  .adjustment-vars {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .var-tag {
    padding: 4px 10px;
    background: rgba(239, 68, 68, 0.15);
    border-radius: 4px;
    font-size: 12px;
    color: var(--error);
  }

  .legend-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .legend-line {
    width: 20px;
    height: 2px;
    background: var(--warning);
  }

  .legend-line.dashed {
    background: repeating-linear-gradient(
      90deg,
      var(--warning) 0px,
      var(--warning) 4px,
      transparent 4px,
      transparent 8px
    );
  }
</style>
