<script lang="ts">
  import { tda, interpretBettiNumbers } from '$lib/stores/tda';
  import TDAPanel from '$lib/components/science/tda/TDAPanel.svelte';
</script>

<div class="tda-page">
  <TDAPanel />
</div>

<style>
  .tda-page {
    height: 100%;
  }
</style>
