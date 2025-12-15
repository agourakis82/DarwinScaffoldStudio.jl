<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let value: number = 0;
  export let min: number = 0;
  export let max: number = 100;
  export let step: number = 1;
  export let timePoints: number[] = [];
  export let isPlaying: boolean = false;
  export let label: string = 'Time';
  export let unit: string = 'h';

  const dispatch = createEventDispatcher<{
    change: number;
    play: void;
    pause: void;
  }>();

  function handleInput(e: Event) {
    const target = e.target as HTMLInputElement;
    dispatch('change', parseInt(target.value));
  }

  function togglePlay() {
    if (isPlaying) {
      dispatch('pause');
    } else {
      dispatch('play');
    }
  }

  function stepBackward() {
    if (value > min) {
      dispatch('change', value - step);
    }
  }

  function stepForward() {
    if (value < max) {
      dispatch('change', value + step);
    }
  }

  $: currentTime = timePoints[value] ?? value;
</script>

<div class="time-slider">
  <div class="slider-header">
    <span class="slider-label">{label}</span>
    <span class="slider-value">{currentTime.toFixed(1)} {unit}</span>
  </div>

  <div class="slider-controls">
    <button class="control-btn" on:click={stepBackward} disabled={value <= min}>
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="19 20 9 12 19 4 19 20"></polygon>
        <line x1="5" y1="19" x2="5" y2="5"></line>
      </svg>
    </button>

    <button class="control-btn play-btn" on:click={togglePlay}>
      {#if isPlaying}
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="4" width="4" height="16"></rect>
          <rect x="14" y="4" width="4" height="16"></rect>
        </svg>
      {:else}
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      {/if}
    </button>

    <button class="control-btn" on:click={stepForward} disabled={value >= max}>
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 4 15 12 5 20 5 4"></polygon>
        <line x1="19" y1="5" x2="19" y2="19"></line>
      </svg>
    </button>

    <div class="slider-track-wrapper">
      <input
        type="range"
        class="slider-input"
        {min}
        {max}
        {step}
        {value}
        on:input={handleInput}
      />
      <div class="slider-progress" style="width: {((value - min) / (max - min)) * 100}%"></div>
    </div>

    <span class="time-range">{timePoints[max]?.toFixed(1) ?? max} {unit}</span>
  </div>
</div>

<style>
  .time-slider {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 14px 16px;
  }

  .slider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .slider-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .slider-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--primary);
  }

  .slider-controls {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .control-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    background: var(--bg-tertiary);
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .control-btn:hover:not(:disabled) {
    background: var(--bg-elevated);
    color: var(--text-primary);
  }

  .control-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .play-btn {
    width: 36px;
    height: 36px;
    background: var(--primary);
    color: white;
  }

  .play-btn:hover {
    background: var(--primary-hover) !important;
    color: white !important;
  }

  .slider-track-wrapper {
    flex: 1;
    position: relative;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .slider-input {
    position: absolute;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
    z-index: 2;
    margin: 0;
  }

  .slider-progress {
    position: absolute;
    height: 100%;
    background: var(--primary);
    border-radius: 3px;
    pointer-events: none;
    transition: width 0.1s ease-out;
  }

  .time-range {
    font-size: 11px;
    color: var(--text-muted);
    min-width: 50px;
    text-align: right;
  }
</style>
