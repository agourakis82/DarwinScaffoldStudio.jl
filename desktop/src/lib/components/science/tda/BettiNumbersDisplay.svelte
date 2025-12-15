<script lang="ts">
  import type { BettiNumbers } from '$lib/stores/tda';

  export let bettiNumbers: BettiNumbers = { beta0: 0, beta1: 0, beta2: 0 };

  let container: HTMLDivElement;

  const labels = [
    { key: 'beta0', label: 'Beta 0', desc: 'Connected components', color: '#3b82f6' },
    { key: 'beta1', label: 'Beta 1', desc: 'Tunnels / Loops', color: '#10b981' },
    { key: 'beta2', label: 'Beta 2', desc: 'Voids / Cavities', color: '#f59e0b' },
  ];

  $: maxValue = Math.max(bettiNumbers.beta0, bettiNumbers.beta1, bettiNumbers.beta2, 1);

  function getBarWidth(value: number): string {
    return `${Math.max(5, (value / maxValue) * 100)}%`;
  }

  function getBettiValue(key: string): number {
    if (key === 'beta0') return bettiNumbers.beta0;
    if (key === 'beta1') return bettiNumbers.beta1;
    if (key === 'beta2') return bettiNumbers.beta2;
    return 0;
  }
</script>

<div class="betti-display" bind:this={container}>
  <h3 class="title">Betti Numbers</h3>

  <div class="bars">
    {#each labels as item}
      <div class="bar-row">
        <div class="bar-label">
          <span class="label-name">{item.label}</span>
          <span class="label-desc">{item.desc}</span>
        </div>
        <div class="bar-container">
          <div
            class="bar"
            style="width: {getBarWidth(getBettiValue(item.key))}; background: {item.color}"
          ></div>
          <span class="bar-value">{getBettiValue(item.key)}</span>
        </div>
      </div>
    {/each}
  </div>

  <div class="legend">
    <div class="legend-item">
      <span class="legend-dot" style="background: #3b82f6"></span>
      <span>B0: Disconnected regions (ideal: 1)</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background: #10b981"></span>
      <span>B1: Through-channels (higher = better flow)</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background: #f59e0b"></span>
      <span>B2: Enclosed voids (lower = fewer trapped cells)</span>
    </div>
  </div>
</div>

<style>
  .betti-display {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 20px 0;
  }

  .bars {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .bar-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .bar-label {
    width: 100px;
    flex-shrink: 0;
  }

  .label-name {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .label-desc {
    display: block;
    font-size: 10px;
    color: var(--text-muted);
  }

  .bar-container {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 12px;
    height: 28px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    padding: 4px;
  }

  .bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease-out;
    min-width: 4px;
  }

  .bar-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 40px;
    text-align: right;
  }

  .legend {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
</style>
