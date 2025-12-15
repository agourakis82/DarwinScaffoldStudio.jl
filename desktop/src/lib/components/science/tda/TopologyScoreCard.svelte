<script lang="ts">
  import { interpretBettiNumbers, type BettiNumbers } from '$lib/stores/tda';

  export let score: number = 0;
  export let bettiNumbers: BettiNumbers = { beta0: 0, beta1: 0, beta2: 0 };

  $: interpretation = interpretBettiNumbers(bettiNumbers);
  $: scoreColor = score >= 80 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--error)';
  $: scoreLabel = score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Needs Improvement';

  // Calculate stroke dasharray for circular progress
  const circumference = 2 * Math.PI * 45;
  $: dashOffset = circumference - (score / 100) * circumference;
</script>

<div class="score-card">
  <div class="score-visual">
    <svg width="120" height="120" viewBox="0 0 120 120">
      <!-- Background circle -->
      <circle
        cx="60"
        cy="60"
        r="45"
        fill="none"
        stroke="var(--bg-tertiary)"
        stroke-width="8"
      />
      <!-- Progress circle -->
      <circle
        cx="60"
        cy="60"
        r="45"
        fill="none"
        stroke={scoreColor}
        stroke-width="8"
        stroke-linecap="round"
        stroke-dasharray={circumference}
        stroke-dashoffset={dashOffset}
        transform="rotate(-90 60 60)"
        style="transition: stroke-dashoffset 0.8s ease-out"
      />
    </svg>
    <div class="score-text">
      <span class="score-value" style="color: {scoreColor}">{Math.round(score)}</span>
      <span class="score-label">{scoreLabel}</span>
    </div>
  </div>

  <div class="score-details">
    <h3 class="title">Interconnectivity Score</h3>

    <div class="interpretation">
      <p>{interpretation}</p>
    </div>

    <div class="recommendations">
      <h4>Recommendations</h4>
      <ul>
        {#if bettiNumbers.beta0 > 1}
          <li class="warning">
            <span class="dot"></span>
            Consider increasing unit cell overlap to connect {bettiNumbers.beta0} isolated regions
          </li>
        {:else}
          <li class="success">
            <span class="dot"></span>
            Scaffold is fully connected - excellent for cell migration
          </li>
        {/if}

        {#if bettiNumbers.beta1 < 20}
          <li class="warning">
            <span class="dot"></span>
            Low tunnel count may limit nutrient transport - consider gyroid or diamond TPMS
          </li>
        {:else}
          <li class="success">
            <span class="dot"></span>
            Good channel network for nutrient and waste exchange
          </li>
        {/if}

        {#if bettiNumbers.beta2 > 5}
          <li class="warning">
            <span class="dot"></span>
            {bettiNumbers.beta2} enclosed voids detected - may trap cells and limit growth
          </li>
        {:else}
          <li class="success">
            <span class="dot"></span>
            Minimal enclosed voids - open structure supports cell colonization
          </li>
        {/if}
      </ul>
    </div>
  </div>
</div>

<style>
  .score-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    display: flex;
    gap: 24px;
  }

  .score-visual {
    position: relative;
    width: 120px;
    height: 120px;
    flex-shrink: 0;
  }

  .score-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
  }

  .score-value {
    display: block;
    font-size: 28px;
    font-weight: 700;
  }

  .score-label {
    display: block;
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .score-details {
    flex: 1;
    min-width: 0;
  }

  .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 12px 0;
  }

  .interpretation {
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
  }

  .interpretation p {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.5;
  }

  .recommendations h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 10px 0;
  }

  .recommendations ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .recommendations li {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .recommendations .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
  }

  .recommendations .success .dot {
    background: var(--success);
  }

  .recommendations .warning .dot {
    background: var(--warning);
  }
</style>
