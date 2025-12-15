<script lang="ts">
  import { onMount } from 'svelte';
  import { gnn, type ScaffoldGraph, type GNNPrediction, type GraphNode } from '$lib/stores/gnn';
  import NodeImportanceHeatmap3D from './NodeImportanceHeatmap3D.svelte';
  import GNNPredictionCard from './GNNPredictionCard.svelte';
  import NodeFeatureInspector from './NodeFeatureInspector.svelte';

  let selectedNodeData: GraphNode | null = null;

  $: if ($gnn.graph && $gnn.selectedNode !== null) {
    selectedNodeData = $gnn.graph.nodes.find(n => n.id === $gnn.selectedNode) || null;
  } else {
    selectedNodeData = null;
  }

  async function loadGraph() {
    gnn.startLoading();

    try {
      const response = await fetch('http://localhost:8081/gnn/graph');
      if (!response.ok) throw new Error('Failed to load graph');

      const data = await response.json();
      gnn.setGraph(data.graph);
    } catch {
      // Generate demo graph
      generateDemoGraph();
    }
  }

  function generateDemoGraph() {
    const nodes: GraphNode[] = [];
    const nNodes = 50;

    for (let i = 0; i < nNodes; i++) {
      nodes.push({
        id: i,
        x: (Math.random() - 0.5) * 15,
        y: (Math.random() - 0.5) * 15,
        z: (Math.random() - 0.5) * 15,
        features: Array.from({ length: 8 }, () => Math.random()),
        importance: Math.random(),
      });
    }

    const edges: { source: number; target: number; features: number[]; attention?: number }[] = [];
    for (let i = 0; i < nNodes; i++) {
      // Connect to 2-4 nearest neighbors
      const nNeighbors = 2 + Math.floor(Math.random() * 3);
      const distances = nodes
        .filter(n => n.id !== i)
        .map(n => ({
          id: n.id,
          dist: Math.sqrt((n.x - nodes[i].x) ** 2 + (n.y - nodes[i].y) ** 2 + (n.z - nodes[i].z) ** 2)
        }))
        .sort((a, b) => a.dist - b.dist);

      for (let j = 0; j < Math.min(nNeighbors, distances.length); j++) {
        edges.push({
          source: i,
          target: distances[j].id,
          features: Array.from({ length: 4 }, () => Math.random()),
          attention: Math.random(),
        });
      }
    }

    const graph: ScaffoldGraph = {
      nodes,
      edges,
      nNodes: nodes.length,
      nEdges: edges.length,
      meanDegree: edges.length / nodes.length,
    };

    gnn.setGraph(graph);
  }

  async function predictProperties() {
    gnn.startPrediction();

    try {
      const response = await fetch('http://localhost:8081/gnn/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: $gnn.modelType }),
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      gnn.setPredictions(data.predictions, data.node_importance);
    } catch {
      // Generate demo predictions
      const demoPredictions: GNNPrediction[] = [
        { property: 'Permeability', value: 1.23e-10, unit: 'm^2', confidence: 0.92 },
        { property: 'Elastic Modulus', value: 450, unit: 'MPa', confidence: 0.87 },
        { property: 'Yield Strength', value: 12.5, unit: 'MPa', confidence: 0.78 },
        { property: 'Fatigue Life', value: 1e6, unit: 'cycles', confidence: 0.65 },
      ];

      const demoImportance = $gnn.graph?.nodes.map(() => Math.random()) || [];
      gnn.setPredictions(demoPredictions, demoImportance);
    }
  }

  onMount(() => {
    if (!$gnn.graph) {
      generateDemoGraph();
    }
  });
</script>

<div class="gnn-panel">
  <header class="panel-header">
    <div class="header-title">
      <h1>GNN Property Prediction</h1>
      <p>Graph neural network for scaffold property prediction with node importance</p>
    </div>

    <div class="header-actions">
      <select class="model-select" bind:value={$gnn.modelType}>
        <option value="gcn">GCN (Graph Convolutional)</option>
        <option value="sage">GraphSAGE</option>
        <option value="gat">GAT (Graph Attention)</option>
        <option value="transformer">Graph Transformer</option>
      </select>

      <button class="load-btn" on:click={loadGraph} disabled={$gnn.isLoading}>
        {#if $gnn.isLoading}
          <span class="spinner"></span>
          Loading...
        {:else}
          Load Graph
        {/if}
      </button>

      <button
        class="predict-btn"
        on:click={predictProperties}
        disabled={!$gnn.graph || $gnn.isPredicting}
      >
        {#if $gnn.isPredicting}
          <span class="spinner"></span>
          Predicting...
        {:else}
          Predict Properties
        {/if}
      </button>
    </div>
  </header>

  <div class="panel-content">
    <div class="main-view">
      <NodeImportanceHeatmap3D
        graph={$gnn.graph}
        nodeImportance={$gnn.nodeImportance}
        selectedNode={$gnn.selectedNode}
        width={700}
        height={450}
      />

      {#if $gnn.graph}
        <div class="graph-stats">
          <div class="stat">
            <span class="stat-value">{$gnn.graph.nNodes}</span>
            <span class="stat-label">Nodes</span>
          </div>
          <div class="stat">
            <span class="stat-value">{$gnn.graph.nEdges}</span>
            <span class="stat-label">Edges</span>
          </div>
          <div class="stat">
            <span class="stat-value">{$gnn.graph.meanDegree.toFixed(2)}</span>
            <span class="stat-label">Mean Degree</span>
          </div>
          <div class="stat">
            <span class="stat-value">{$gnn.modelType.toUpperCase()}</span>
            <span class="stat-label">Model</span>
          </div>
        </div>
      {/if}
    </div>

    <aside class="sidebar">
      <GNNPredictionCard
        predictions={$gnn.predictions}
        isLoading={$gnn.isPredicting}
      />

      <NodeFeatureInspector node={selectedNodeData} />
    </aside>
  </div>
</div>

<style>
  .gnn-panel {
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

  .model-select {
    padding: 10px 14px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .load-btn, .predict-btn {
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

  .load-btn {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  .load-btn:hover:not(:disabled) {
    background: var(--bg-tertiary);
  }

  .predict-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }

  .load-btn:disabled, .predict-btn:disabled {
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

  .load-btn .spinner {
    border-color: var(--border-color);
    border-top-color: var(--text-primary);
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

  .graph-stats {
    display: flex;
    gap: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border-radius: 12px;
  }

  .stat {
    flex: 1;
    text-align: center;
  }

  .stat-value {
    display: block;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .sidebar {
    width: 300px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }
</style>
