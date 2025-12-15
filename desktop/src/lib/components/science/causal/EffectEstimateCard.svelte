<script lang="ts">
  import { formatEffect, isSignificant, type CausalEffect } from '$lib/stores/causal';

  export let effect: CausalEffect | null = null;
  export let treatment: string = '';
  export let outcome: string = '';
</script>

<div class="effect-card">
  <h3 class="title">Causal Effect Estimate</h3>

  {#if effect}
    <div class="effect-display">
      <div class="effect-value" class:significant={isSignificant(effect)} class:not-significant={!isSignificant(effect)}>
        <span class="value">{effect.ate >= 0 ? '+' : ''}{effect.ate.toFixed(3)}</span>
        <span class="label">Average Treatment Effect</span>
      </div>

      <div class="effect-details">
        <div class="detail-row">
          <span class="detail-label">Treatment</span>
          <span class="detail-value treatment">{treatment}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Outcome</span>
          <span class="detail-value outcome">{outcome}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">95% CI</span>
          <span class="detail-value">[{effect.ciLower.toFixed(3)}, {effect.ciUpper.toFixed(3)}]</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Std. Error</span>
          <span class="detail-value">{effect.standardError.toFixed(4)}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">p-value</span>
          <span class="detail-value" class:significant={isSignificant(effect)}>
            {effect.pValue < 0.001 ? '< 0.001' : effect.pValue.toFixed(4)}
            {#if isSignificant(effect)}*{/if}
          </span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Method</span>
          <span class="detail-value">{effect.method}</span>
        </div>
      </div>

      <div class="interpretation">
        {#if isSignificant(effect)}
          <p class="significant-text">
            {#if effect.ate > 0}
              Increasing <strong>{treatment}</strong> is associated with a statistically significant
              <strong>increase</strong> in <strong>{outcome}</strong>.
            {:else}
              Increasing <strong>{treatment}</strong> is associated with a statistically significant
              <strong>decrease</strong> in <strong>{outcome}</strong>.
            {/if}
          </p>
        {:else}
          <p class="not-significant-text">
            No statistically significant causal effect detected at the 0.05 significance level.
          </p>
        {/if}
      </div>
    </div>
  {:else}
    <div class="no-effect">
      <p>Select treatment and outcome variables, then click "Estimate Effect" to compute the causal effect.</p>
    </div>
  {/if}
</div>

<style>
  .effect-card {
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

  .effect-display {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .effect-value {
    text-align: center;
    padding: 20px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    border-left: 4px solid var(--text-muted);
  }

  .effect-value.significant {
    border-left-color: var(--success);
  }

  .effect-value.not-significant {
    border-left-color: var(--warning);
  }

  .value {
    display: block;
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .effect-value.significant .value {
    color: var(--success);
  }

  .label {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .effect-details {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .detail-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .detail-value {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .detail-value.treatment {
    color: var(--success);
  }

  .detail-value.outcome {
    color: var(--warning);
  }

  .detail-value.significant {
    color: var(--success);
  }

  .interpretation {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .interpretation p {
    font-size: 13px;
    line-height: 1.5;
    margin: 0;
  }

  .significant-text {
    color: var(--text-secondary);
  }

  .not-significant-text {
    color: var(--text-muted);
  }

  .no-effect {
    padding: 24px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }
</style>
