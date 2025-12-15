<script lang="ts">
  import type { GraphNode } from '$lib/stores/gnn';
  import { NODE_FEATURE_NAMES, importanceToColor } from '$lib/stores/gnn';

  export let node: GraphNode | null = null;
</script>

<div class="inspector">
  <h3 class="title">Node Inspector</h3>

  {#if node}
    <div class="node-info">
      <div class="info-header">
        <div class="node-id">Node #{node.id}</div>
        <div class="importance-badge" style="background: {importanceToColor(node.importance)}">
          {(node.importance * 100).toFixed(1)}% importance
        </div>
      </div>

      <div class="coordinates">
        <div class="coord">
          <span class="coord-label">X</span>
          <span class="coord-value">{node.x.toFixed(2)}</span>
        </div>
        <div class="coord">
          <span class="coord-label">Y</span>
          <span class="coord-value">{node.y.toFixed(2)}</span>
        </div>
        <div class="coord">
          <span class="coord-label">Z</span>
          <span class="coord-value">{node.z.toFixed(2)}</span>
        </div>
      </div>

      <div class="features">
        <h4>Node Features</h4>
        <div class="feature-list">
          {#each node.features as feature, i}
            <div class="feature-row">
              <span class="feature-name">{NODE_FEATURE_NAMES[i] || `Feature ${i}`}</span>
              <div class="feature-bar-container">
                <div class="feature-bar" style="width: {Math.abs(feature) * 100}%"></div>
              </div>
              <span class="feature-value">{feature.toFixed(3)}</span>
            </div>
          {/each}
        </div>
      </div>

      {#if node.label}
        <div class="node-label">
          <span class="label-key">Label:</span>
          <span class="label-value">{node.label}</span>
        </div>
      {/if}
    </div>
  {:else}
    <div class="no-selection">
      <p>No node selected</p>
      <p class="hint">Click on a node in the 3D view to inspect its features</p>
    </div>
  {/if}
</div>

<style>
  .inspector {
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

  .node-info {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .info-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .node-id {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .importance-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    color: white;
  }

  .coordinates {
    display: flex;
    gap: 12px;
  }

  .coord {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .coord-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
  }

  .coord-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 4px;
  }

  .features h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 12px 0;
  }

  .feature-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .feature-row {
    display: grid;
    grid-template-columns: 120px 1fr 50px;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }

  .feature-name {
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .feature-bar-container {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .feature-bar {
    height: 100%;
    background: var(--primary);
    border-radius: 3px;
  }

  .feature-value {
    text-align: right;
    color: var(--text-primary);
    font-family: var(--font-mono);
  }

  .node-label {
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    display: flex;
    gap: 8px;
  }

  .label-key {
    font-size: 12px;
    color: var(--text-muted);
  }

  .label-value {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .no-selection {
    padding: 24px;
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
