<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { oxygenToColor, HYPOXIA_THRESHOLD, type PINNDimensions } from '$lib/stores/pinn';

  export let concentrationData: Float32Array | null = null;
  export let dimensions: PINNDimensions = { nx: 32, ny: 32, nz: 32, nt: 1 };
  export let timeIndex: number = 0;
  export let showHypoxia: boolean = true;
  export let opacity: number = 0.6;

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let volumeMesh: THREE.Points;
  let hypoxiaMesh: THREE.Points;
  let animationId: number;

  onMount(() => {
    initScene();
    animate();
  });

  onDestroy(() => {
    if (animationId) cancelAnimationFrame(animationId);
    if (renderer) renderer.dispose();
    if (controls) controls.dispose();
  });

  $: if (scene && concentrationData) {
    updateVisualization();
  }

  $: if (hypoxiaMesh) {
    hypoxiaMesh.visible = showHypoxia;
  }

  function initScene() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    // Camera
    camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(2, 2, 2);

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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Bounding box helper
    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const boxEdges = new THREE.EdgesGeometry(boxGeometry);
    const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x4a9eff, opacity: 0.3, transparent: true }));
    boxLine.position.set(0.5, 0.5, 0.5);
    scene.add(boxLine);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(0.3);
    scene.add(axesHelper);

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);

    // Generate demo data if none provided
    if (!concentrationData) {
      generateDemoData();
    } else {
      updateVisualization();
    }
  }

  function generateDemoData() {
    const { nx, ny, nz } = dimensions;
    const data = new Float32Array(nx * ny * nz);

    // Generate realistic concentration gradient (higher at edges, lower in center)
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const cx = (x - nx/2) / (nx/2);
          const cy = (y - ny/2) / (ny/2);
          const cz = (z - nz/2) / (nz/2);

          // Distance from center (0-1)
          const dist = Math.sqrt(cx*cx + cy*cy + cz*cz) / Math.sqrt(3);

          // Concentration: high at edges (160 mmHg), low at center (potentially hypoxic)
          const base = 160 * dist;
          const noise = (Math.random() - 0.5) * 20;
          const value = Math.max(0, Math.min(160, base + noise));

          data[x + y * nx + z * nx * ny] = value;
        }
      }
    }

    concentrationData = data;
    updateVisualization();
  }

  function updateVisualization() {
    if (!concentrationData || !scene) return;

    // Remove existing meshes
    if (volumeMesh) scene.remove(volumeMesh);
    if (hypoxiaMesh) scene.remove(hypoxiaMesh);

    const { nx, ny, nz } = dimensions;
    const positions: number[] = [];
    const colors: number[] = [];
    const hypoxiaPositions: number[] = [];

    // Sample points for visualization
    const step = Math.max(1, Math.floor(Math.max(nx, ny, nz) / 32));

    for (let z = 0; z < nz; z += step) {
      for (let y = 0; y < ny; y += step) {
        for (let x = 0; x < nx; x += step) {
          const idx = x + y * nx + z * nx * ny;
          const value = concentrationData[idx];

          // Normalized position (0-1)
          const px = x / nx;
          const py = y / ny;
          const pz = z / nz;

          // Add to main visualization
          positions.push(px, py, pz);

          // Convert value to color
          const colorStr = oxygenToColor(value, 0, 160);
          const rgb = parseColor(colorStr);
          colors.push(rgb.r, rgb.g, rgb.b);

          // Track hypoxic regions
          if (value < HYPOXIA_THRESHOLD) {
            hypoxiaPositions.push(px, py, pz);
          }
        }
      }
    }

    // Create main point cloud
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      transparent: true,
      opacity: opacity,
      sizeAttenuation: true,
    });

    volumeMesh = new THREE.Points(geometry, material);
    scene.add(volumeMesh);

    // Create hypoxia highlight
    if (hypoxiaPositions.length > 0) {
      const hypoxiaGeometry = new THREE.BufferGeometry();
      hypoxiaGeometry.setAttribute('position', new THREE.Float32BufferAttribute(hypoxiaPositions, 3));

      const hypoxiaMaterial = new THREE.PointsMaterial({
        size: 0.05,
        color: 0xff0000,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: true,
      });

      hypoxiaMesh = new THREE.Points(hypoxiaGeometry, hypoxiaMaterial);
      hypoxiaMesh.visible = showHypoxia;
      scene.add(hypoxiaMesh);
    }
  }

  function parseColor(colorStr: string): { r: number; g: number; b: number } {
    const match = colorStr.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match) {
      return {
        r: parseInt(match[1]) / 255,
        g: parseInt(match[2]) / 255,
        b: parseInt(match[3]) / 255,
      };
    }
    return { r: 0.5, g: 0.5, b: 0.5 };
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
</script>

<div class="heatmap-3d" bind:this={container}></div>

<style>
  .heatmap-3d {
    width: 100%;
    height: 100%;
    min-height: 400px;
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg-primary);
  }
</style>
