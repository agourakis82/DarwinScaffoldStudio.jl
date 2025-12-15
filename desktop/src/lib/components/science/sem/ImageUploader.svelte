<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let imageUrl: string | null = null;
  export let isLoading: boolean = false;

  const dispatch = createEventDispatcher<{ upload: { url: string; width: number; height: number } }>();

  let dragOver = false;
  let fileInput: HTMLInputElement;

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;

    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragOver = true;
  }

  function handleDragLeave() {
    dragOver = false;
  }

  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      processFile(file);
    }
  }

  function processFile(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      const img = new Image();
      img.onload = () => {
        dispatch('upload', { url, width: img.width, height: img.height });
      };
      img.src = url;
    };
    reader.readAsDataURL(file);
  }

  function triggerUpload() {
    fileInput.click();
  }
</script>

<div
  class="uploader"
  class:drag-over={dragOver}
  class:has-image={imageUrl !== null}
  on:drop={handleDrop}
  on:dragover={handleDragOver}
  on:dragleave={handleDragLeave}
  role="button"
  tabindex="0"
  on:click={triggerUpload}
  on:keypress={(e) => e.key === 'Enter' && triggerUpload()}
>
  <input
    type="file"
    accept="image/*"
    bind:this={fileInput}
    on:change={handleFileSelect}
    class="hidden"
  />

  {#if isLoading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Processing image...</span>
    </div>
  {:else if imageUrl}
    <img src={imageUrl} alt="SEM Image" class="preview" />
    <div class="overlay">
      <span>Click or drop to replace</span>
    </div>
  {:else}
    <div class="placeholder">
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
      </svg>
      <span class="label">Drop SEM image here</span>
      <span class="hint">or click to browse</span>
      <span class="formats">Supports: PNG, JPG, TIFF</span>
    </div>
  {/if}
</div>

<style>
  .uploader {
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
    background: var(--bg-secondary);
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    cursor: pointer;
    transition: all var(--transition-fast);
    overflow: hidden;
  }

  .uploader:hover {
    border-color: var(--primary);
    background: rgba(74, 158, 255, 0.05);
  }

  .uploader.drag-over {
    border-color: var(--primary);
    background: rgba(74, 158, 255, 0.1);
    transform: scale(1.01);
  }

  .uploader.has-image {
    border-style: solid;
    border-color: var(--border-color);
  }

  .hidden {
    display: none;
  }

  .loading {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-muted);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .preview {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #000;
  }

  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.6);
    opacity: 0;
    transition: opacity var(--transition-fast);
  }

  .uploader.has-image:hover .overlay {
    opacity: 1;
  }

  .overlay span {
    color: white;
    font-size: 14px;
    font-weight: 500;
  }

  .placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    gap: 8px;
  }

  .placeholder svg {
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .label {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .hint {
    font-size: 13px;
  }

  .formats {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
  }
</style>
