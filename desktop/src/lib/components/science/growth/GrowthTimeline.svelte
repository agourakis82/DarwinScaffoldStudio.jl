<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let currentTime: number = 0;
  export let maxTime: number = 168;
  export let isPlaying: boolean = false;
  export let playbackSpeed: number = 1;

  const dispatch = createEventDispatcher<{
    timeChange: number;
    play: void;
    pause: void;
    speedChange: number;
  }>();

  function handleSliderInput(e: Event) {
    const target = e.target as HTMLInputElement;
    dispatch('timeChange', parseInt(target.value));
  }

  function togglePlay() {
    if (isPlaying) {
      dispatch('pause');
    } else {
      dispatch('play');
    }
  }

  function setSpeed(speed: number) {
    dispatch('speedChange', speed);
  }

  function formatTime(hours: number): string {
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    const remaining = hours % 24;
    return remaining > 0 ? `${days}d ${remaining}h` : `${days}d`;
  }
</script>

<div class="growth-timeline">
  <div class="timeline-header">
    <span class="time-display">{formatTime(currentTime)}</span>
    <span class="time-total">/ {formatTime(maxTime)}</span>
  </div>

  <div class="timeline-controls">
    <button class="control-btn" on:click={() => dispatch('timeChange', 0)} title="Reset">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="19 20 9 12 19 4 19 20"></polygon>
        <line x1="5" y1="19" x2="5" y2="5"></line>
      </svg>
    </button>

    <button class="control-btn play-btn" on:click={togglePlay}>
      {#if isPlaying}
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="4" width="4" height="16"></rect>
          <rect x="14" y="4" width="4" height="16"></rect>
        </svg>
      {:else}
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      {/if}
    </button>

    <div class="slider-wrapper">
      <input
        type="range"
        class="timeline-slider"
        min="0"
        max={maxTime}
        value={currentTime}
        on:input={handleSliderInput}
      />
      <div class="slider-progress" style="width: {(currentTime / maxTime) * 100}%"></div>
    </div>
  </div>

  <div class="speed-controls">
    <span class="speed-label">Speed:</span>
    {#each [0.5, 1, 2, 4] as speed}
      <button
        class="speed-btn"
        class:active={playbackSpeed === speed}
        on:click={() => setSpeed(speed)}
      >
        {speed}x
      </button>
    {/each}
  </div>
</div>

<style>
  .growth-timeline {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
  }

  .timeline-header {
    display: flex;
    align-items: baseline;
    gap: 4px;
    margin-bottom: 12px;
  }

  .time-display {
    font-size: 24px;
    font-weight: 700;
    color: var(--primary);
  }

  .time-total {
    font-size: 14px;
    color: var(--text-muted);
  }

  .timeline-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .control-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: var(--bg-tertiary);
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .control-btn:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
  }

  .play-btn {
    width: 44px;
    height: 44px;
    background: var(--primary);
    color: white;
  }

  .play-btn:hover {
    background: var(--primary-hover) !important;
    color: white !important;
  }

  .slider-wrapper {
    flex: 1;
    position: relative;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .timeline-slider {
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
    border-radius: 4px;
    pointer-events: none;
    transition: width 0.05s linear;
  }

  .speed-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .speed-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .speed-btn {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    background: var(--bg-tertiary);
    border: none;
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .speed-btn:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
  }

  .speed-btn.active {
    background: var(--primary);
    color: white;
  }
</style>
