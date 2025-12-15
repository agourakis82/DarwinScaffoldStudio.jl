<script lang="ts">
  export let width: string = '100%';
  export let height: string = '20px';
  export let rounded: boolean = false;
  export let circle: boolean = false;
</script>

<div
  class="skeleton"
  class:rounded
  class:circle
  style="width: {width}; height: {height};"
></div>

<style>
  .skeleton {
    background: linear-gradient(
      90deg,
      var(--bg-tertiary) 0%,
      var(--bg-secondary) 50%,
      var(--bg-tertiary) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
  }

  .skeleton.rounded {
    border-radius: 8px;
  }

  .skeleton.circle {
    border-radius: 50%;
  }

  @keyframes shimmer {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }
</style>
