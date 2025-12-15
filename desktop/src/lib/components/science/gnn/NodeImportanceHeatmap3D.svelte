<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import type { ScaffoldGraph, GraphNode } from '$lib/stores/gnn';
  import { importanceToColor } from '$lib/stores/gnn';

  export let graph: ScaffoldGraph | null = null;
  export let nodeImportance: number[] = [];
  export let selectedNode: number | null = null;
  export let width: number = 600;
  export let height: number = 400;

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let nodeMeshes: THREE.InstancedMesh;
  let edgeLines: THREE.LineSegments;
  let animationId: number;

  const nodeRadius = 0.5;
  const dummy = new THREE.Object3D();

  $: if (scene && graph) {
    updateVisualization();
  }

  $: if (nodeMeshes && nodeImportance.length > 0) {
    updateNodeColors();
  }

  onMount(() => {
    initScene();
  });

  onDestroy(() => {
    if (animationId) cancelAnimationFrame(animationId);
    if (renderer) renderer.dispose();
    if (controls) controls.dispose();
  });

  function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x161b22);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(20, 15, 20);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20, 0x3d4450, 0x3d4450);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    if (graph) updateVisualization();

    animate();
  }

  function updateVisualization() {
    if (!scene || !graph) return;

    // Clear previous meshes
    if (nodeMeshes) scene.remove(nodeMeshes);
    if (edgeLines) scene.remove(edgeLines);

    // Create instanced mesh for nodes
    const geometry = new THREE.SphereGeometry(nodeRadius, 16, 16);
    const material = new THREE.MeshPhongMaterial({ vertexColors: true });

    nodeMeshes = new THREE.InstancedMesh(geometry, material, graph.nodes.length);

    // Position nodes and set colors
    const colors = new Float32Array(graph.nodes.length * 3);
    graph.nodes.forEach((node, i) => {
      dummy.position.set(node.x, node.y, node.z);
      dummy.updateMatrix();
      nodeMeshes.setMatrixAt(i, dummy.matrix);

      // Color based on importance
      const importance = nodeImportance[i] ?? node.importance;
      const color = new THREE.Color(importanceToColor(importance));
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    nodeMeshes.instanceMatrix.needsUpdate = true;
    scene.add(nodeMeshes);

    // Create edges
    const edgeGeometry = new THREE.BufferGeometry();
    const positions: number[] = [];

    graph.edges.forEach(edge => {
      const source = graph.nodes.find(n => n.id === edge.source);
      const target = graph.nodes.find(n => n.id === edge.target);
      if (source && target) {
        positions.push(source.x, source.y, source.z);
        positions.push(target.x, target.y, target.z);
      }
    });

    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x6b7280, opacity: 0.5, transparent: true });
    edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    scene.add(edgeLines);

    // Center camera on graph
    if (graph.nodes.length > 0) {
      const center = new THREE.Vector3();
      graph.nodes.forEach(n => center.add(new THREE.Vector3(n.x, n.y, n.z)));
      center.divideScalar(graph.nodes.length);
      controls.target.copy(center);
    }
  }

  function updateNodeColors() {
    if (!nodeMeshes || !graph) return;

    const colorAttribute = new Float32Array(graph.nodes.length * 3);
    graph.nodes.forEach((node, i) => {
      const importance = nodeImportance[i] ?? node.importance;
      const color = new THREE.Color(importanceToColor(importance));
      colorAttribute[i * 3] = color.r;
      colorAttribute[i * 3 + 1] = color.g;
      colorAttribute[i * 3 + 2] = color.b;
    });

    nodeMeshes.instanceColor = new THREE.InstancedBufferAttribute(colorAttribute, 3);
    nodeMeshes.instanceColor.needsUpdate = true;
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
</script>

<div class="heatmap-container" bind:this={container}>
  {#if !graph}
    <div class="no-data">
      <p>No graph data available</p>
      <p class="hint">Load a scaffold to visualize the GNN graph</p>
    </div>
  {/if}
</div>

<div class="color-bar">
  <span class="label">Low</span>
  <div class="gradient"></div>
  <span class="label">High</span>
  <span class="title">Node Importance</span>
</div>

<style>
  .heatmap-container {
    position: relative;
    background: var(--bg-secondary);
    border-radius: 12px;
    overflow: hidden;
  }

  .no-data {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
  }

  .no-data p {
    margin: 0;
  }

  .no-data .hint {
    font-size: 12px;
    margin-top: 4px;
  }

  .color-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-radius: 8px;
    margin-top: 12px;
  }

  .gradient {
    flex: 1;
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right,
      rgb(0, 0, 255) 0%,
      rgb(255, 255, 128) 50%,
      rgb(255, 0, 0) 100%
    );
  }

  .label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-left: auto;
  }
</style>
