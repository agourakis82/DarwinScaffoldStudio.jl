<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { CellType, CELL_COLORS, type Cell } from '$lib/stores/tissueGrowth';

  export let cells: Cell[] = [];
  export let scaffoldSize: number = 3; // mm

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let cellMeshes: Map<CellType, THREE.InstancedMesh> = new Map();
  let animationId: number;

  const MAX_CELLS_PER_TYPE = 5000;

  onMount(() => {
    initScene();
    animate();
  });

  onDestroy(() => {
    if (animationId) cancelAnimationFrame(animationId);
    if (renderer) renderer.dispose();
    if (controls) controls.dispose();
  });

  $: if (scene && cells) {
    updateCells();
  }

  function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    // Camera
    camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(5, 5, 5);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    // Scaffold bounding box
    const scaffoldGeometry = new THREE.BoxGeometry(scaffoldSize, scaffoldSize, scaffoldSize);
    const scaffoldEdges = new THREE.EdgesGeometry(scaffoldGeometry);
    const scaffoldLine = new THREE.LineSegments(
      scaffoldEdges,
      new THREE.LineBasicMaterial({ color: 0x4a9eff, opacity: 0.3, transparent: true })
    );
    scaffoldLine.position.set(scaffoldSize/2, scaffoldSize/2, scaffoldSize/2);
    scene.add(scaffoldLine);

    // Grid helper
    const gridHelper = new THREE.GridHelper(scaffoldSize, 10, 0x333333, 0x222222);
    scene.add(gridHelper);

    // Create instanced meshes for each cell type
    const sphereGeometry = new THREE.SphereGeometry(0.03, 8, 8);

    for (const cellType of Object.values(CellType)) {
      if (cellType === CellType.DEAD) continue;

      const color = new THREE.Color(CELL_COLORS[cellType]);
      const material = new THREE.MeshLambertMaterial({ color });
      const instancedMesh = new THREE.InstancedMesh(sphereGeometry, material, MAX_CELLS_PER_TYPE);
      instancedMesh.count = 0;
      scene.add(instancedMesh);
      cellMeshes.set(cellType, instancedMesh);
    }

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);

    // Generate demo cells if none provided
    if (cells.length === 0) {
      generateDemoCells();
    } else {
      updateCells();
    }
  }

  function generateDemoCells() {
    const demoCells: Cell[] = [];
    const types = [CellType.MSC, CellType.PREOSTEOBLAST, CellType.OSTEOBLAST, CellType.OSTEOCYTE];
    const counts = [300, 200, 150, 50];

    let id = 0;
    types.forEach((type, typeIdx) => {
      for (let i = 0; i < counts[typeIdx]; i++) {
        demoCells.push({
          id: id++,
          type,
          x: Math.random() * scaffoldSize,
          y: Math.random() * scaffoldSize,
          z: Math.random() * scaffoldSize,
          age: Math.random() * 168,
          health: 0.8 + Math.random() * 0.2,
          divisionReady: Math.random() > 0.8,
        });
      }
    });

    cells = demoCells;
    updateCells();
  }

  function updateCells() {
    if (!scene || cells.length === 0) return;

    const dummy = new THREE.Object3D();
    const cellsByType = new Map<CellType, Cell[]>();

    // Group cells by type
    for (const cell of cells) {
      if (cell.type === CellType.DEAD) continue;
      if (!cellsByType.has(cell.type)) {
        cellsByType.set(cell.type, []);
      }
      cellsByType.get(cell.type)!.push(cell);
    }

    // Update instanced meshes
    for (const [type, mesh] of cellMeshes) {
      const typeCells = cellsByType.get(type) || [];
      mesh.count = Math.min(typeCells.length, MAX_CELLS_PER_TYPE);

      for (let i = 0; i < mesh.count; i++) {
        const cell = typeCells[i];
        dummy.position.set(cell.x, cell.y, cell.z);

        // Scale based on health
        const scale = 0.8 + cell.health * 0.4;
        dummy.scale.setScalar(scale);

        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
</script>

<div class="cell-viz-3d" bind:this={container}></div>

<style>
  .cell-viz-3d {
    width: 100%;
    height: 100%;
    min-height: 400px;
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg-primary);
  }
</style>
