<script lang="ts">
  import { HYPOXIA_THRESHOLD, type HypoxicRegion } from '$lib/stores/pinn';

  export let hypoxicVolume: number = 0;
  export let hypoxicRegions: HypoxicRegion[] = [];
  export let minOxygen: number = 0;

  $: severity = hypoxicVolume > 20 ? 'critical' : hypoxicVolume > 10 ? 'warning' : 'normal';
  $: severityColor = severity === 'critical' ? 'var(--error)' : severity === 'warning' ? 'var(--warning)' : 'var(--success)';
</script>

<div class="hypoxia-overlay">
  <div class="overlay-header">
    <h3>Hypoxia Analysis</h3>
    <span class="threshold-badge">Threshold: {HYPOXIA_THRESHOLD} mmHg</span>
  </div>

  <div class="stats-grid">
    <div class="stat-card" class:critical={severity === 'critical'} class:warning={severity === 'warning'}>
      <span class="stat-value" style="color: {severityColor}">{hypoxicVolume.toFixed(1)}%</span>
      <span class="stat-label">Hypoxic Volume</span>
    </div>

    <div class="stat-card">
      <span class="stat-value">{minOxygen.toFixed(1)}</span>
      <span class="stat-label">Min O2 (mmHg)</span>
    </div>

    <div class="stat-card">
      <span class="stat-value">{hypoxicRegions.length}</span>
      <span class="stat-label">Hypoxic Regions</span>
    </div>
  </div>

  {#if hypoxicRegions.length > 0}
    <div class="regions-list">
      <h4>Critical Regions</h4>
      <div class="regions-scroll">
        {#each hypoxicRegions.slice(0, 5) as region, i}
          <div class="region-item">
            <span class="region-index">#{i + 1}</span>
            <span class="region-pos">({region.x.toFixed(0)}, {region.y.toFixed(0)}, {region.z.toFixed(0)})</span>
            <span class="region-min">{region.minO2.toFixed(1)} mmHg</span>
            <span class="region-vol">{region.volume} voxels</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <div class="recommendations">
    <h4>Recommendations</h4>
    {#if severity === 'critical'}
      <p class="rec critical">
        Critical hypoxia detected. Consider increasing porosity or reducing scaffold thickness
        to improve oxygen diffusion.
      </p>
    {:else if severity === 'warning'}
      <p class="rec warning">
        Moderate hypoxia in some regions. Monitor cell viability in these areas. Consider
        adding vascularization channels.
      </p>
    {:else}
      <p class="rec success">
        Oxygen levels are adequate throughout the scaffold. Good conditions for cell survival
        and differentiation.
      </p>
    {/if}
  </div>
</div>

<style>
  .hypoxia-overlay {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .overlay-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .overlay-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .threshold-badge {
    font-size: 11px;
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    color: var(--text-muted);
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .stat-card {
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .stat-card.warning {
    border: 1px solid var(--warning);
  }

  .stat-card.critical {
    border: 1px solid var(--error);
  }

  .stat-value {
    display: block;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-label {
    display: block;
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .regions-list {
    margin-bottom: 16px;
  }

  .regions-list h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
  }

  .regions-scroll {
    max-height: 150px;
    overflow-y: auto;
  }

  .region-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 11px;
  }

  .region-index {
    font-weight: 600;
    color: var(--error);
  }

  .region-pos {
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }

  .region-min {
    color: var(--warning);
    font-weight: 500;
  }

  .region-vol {
    color: var(--text-muted);
    margin-left: auto;
  }

  .recommendations h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
  }

  .rec {
    font-size: 12px;
    line-height: 1.5;
    padding: 10px 12px;
    border-radius: 6px;
    margin: 0;
  }

  .rec.critical {
    background: rgba(239, 68, 68, 0.1);
    color: var(--error);
  }

  .rec.warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
  }

  .rec.success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
  }
</style>
