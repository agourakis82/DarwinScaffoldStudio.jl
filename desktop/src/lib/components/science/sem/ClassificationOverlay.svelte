<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { DetectedCell } from '$lib/stores/sem';
  import { CELL_TYPE_INFO } from '$lib/stores/sem';

  export let imageUrl: string;
  export let imageSize: { width: number; height: number };
  export let cells: DetectedCell[] = [];
  export let selectedCell: number | null = null;
  export let showOverlay: boolean = true;
  export let overlayOpacity: number = 0.7;

  const dispatch = createEventDispatcher<{ select: number | null }>();

  let containerWidth = 600;
  let containerHeight = 450;

  $: scale = Math.min(containerWidth / imageSize.width, containerHeight / imageSize.height);
  $: displayWidth = imageSize.width * scale;
  $: displayHeight = imageSize.height * scale;
  $: offsetX = (containerWidth - displayWidth) / 2;
  $: offsetY = (containerHeight - displayHeight) / 2;

  function handleCellClick(cellId: number) {
    dispatch('select', selectedCell === cellId ? null : cellId);
  }

  function scaleBox(box: { x: number; y: number; width: number; height: number }) {
    return {
      x: box.x * scale + offsetX,
      y: box.y * scale + offsetY,
      width: box.width * scale,
      height: box.height * scale,
    };
  }
</script>

<div class="overlay-container" bind:clientWidth={containerWidth} bind:clientHeight={containerHeight}>
  <img src={imageUrl} alt="SEM Image" class="base-image" style="width: {displayWidth}px; height: {displayHeight}px; left: {offsetX}px; top: {offsetY}px" />

  {#if showOverlay && cells.length > 0}
    <svg class="overlay-svg" viewBox="0 0 {containerWidth} {containerHeight}">
      {#each cells as cell}
        {@const box = scaleBox(cell.boundingBox)}
        {@const info = CELL_TYPE_INFO[cell.classification.predictedType]}
        <g
          class="cell-marker"
          class:selected={selectedCell === cell.id}
          style="--cell-color: {info.color}; opacity: {overlayOpacity}"
          on:click={() => handleCellClick(cell.id)}
          role="button"
          tabindex="0"
          on:keypress={(e) => e.key === 'Enter' && handleCellClick(cell.id)}
        >
          <rect
            x={box.x}
            y={box.y}
            width={box.width}
            height={box.height}
            fill={info.color}
            fill-opacity="0.2"
            stroke={info.color}
            stroke-width={selectedCell === cell.id ? 3 : 2}
          />
          <circle
            cx={cell.centroid.x * scale + offsetX}
            cy={cell.centroid.y * scale + offsetY}
            r="4"
            fill={info.color}
          />
          <text
            x={box.x + 4}
            y={box.y - 4}
            fill={info.color}
            font-size="11"
            font-weight="600"
          >
            {info.label} ({(cell.classification.confidence * 100).toFixed(0)}%)
          </text>
        </g>
      {/each}
    </svg>
  {/if}
</div>

<style>
  .overlay-container {
    position: relative;
    width: 100%;
    height: 450px;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
  }

  .base-image {
    position: absolute;
    object-fit: contain;
  }

  .overlay-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  .cell-marker {
    cursor: pointer;
    transition: opacity var(--transition-fast);
  }

  .cell-marker:hover rect {
    fill-opacity: 0.4;
  }

  .cell-marker.selected rect {
    stroke-dasharray: 4,2;
    animation: dash 0.5s linear infinite;
  }

  @keyframes dash {
    to {
      stroke-dashoffset: -6;
    }
  }
</style>
