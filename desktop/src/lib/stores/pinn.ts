import { writable, derived } from 'svelte/store';

export interface PINNDimensions {
  nx: number;
  ny: number;
  nz: number;
  nt: number;
}

export interface HypoxicRegion {
  x: number;
  y: number;
  z: number;
  volume: number; // voxel count
  minO2: number;
}

export interface PINNState {
  simulationId: string | null;
  concentrationField: Float32Array | null;
  dimensions: PINNDimensions;
  currentTimeIndex: number;
  timePoints: number[]; // in hours
  hypoxicVolume: number; // percentage
  hypoxicRegions: HypoxicRegion[];
  minOxygen: number;
  maxOxygen: number;
  isComputing: boolean;
  error: string | null;
}

// Boundary conditions
export interface PINNBoundaryConditions {
  oxygenSupply: number; // mmHg at surface
  consumptionRate: number; // mmHg/hr per cell
  diffusionCoeff: number; // mm^2/hr
  scaffoldPorosity: number;
}

export const DEFAULT_BOUNDARY_CONDITIONS: PINNBoundaryConditions = {
  oxygenSupply: 160, // Atmospheric partial pressure
  consumptionRate: 0.5,
  diffusionCoeff: 2.5e-3,
  scaffoldPorosity: 0.85,
};

export const HYPOXIA_THRESHOLD = 10; // mmHg - below this is hypoxic

const initialState: PINNState = {
  simulationId: null,
  concentrationField: null,
  dimensions: { nx: 0, ny: 0, nz: 0, nt: 0 },
  currentTimeIndex: 0,
  timePoints: [],
  hypoxicVolume: 0,
  hypoxicRegions: [],
  minOxygen: 0,
  maxOxygen: 160,
  isComputing: false,
  error: null,
};

function createPINNStore() {
  const { subscribe, set, update } = writable<PINNState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    startSimulation: () => update(state => ({
      ...state,
      isComputing: true,
      error: null,
    })),

    setResults: (
      simulationId: string,
      field: Float32Array,
      dimensions: PINNDimensions,
      timePoints: number[],
      hypoxicVolume: number,
      hypoxicRegions: HypoxicRegion[]
    ) => {
      // Find min/max oxygen values
      let minO2 = Infinity;
      let maxO2 = -Infinity;
      for (let i = 0; i < field.length; i++) {
        if (field[i] < minO2) minO2 = field[i];
        if (field[i] > maxO2) maxO2 = field[i];
      }

      return update(state => ({
        ...state,
        simulationId,
        concentrationField: field,
        dimensions,
        timePoints,
        currentTimeIndex: 0,
        hypoxicVolume,
        hypoxicRegions,
        minOxygen: minO2,
        maxOxygen: maxO2,
        isComputing: false,
      }));
    },

    setTimeIndex: (index: number) => update(state => ({
      ...state,
      currentTimeIndex: Math.max(0, Math.min(index, state.dimensions.nt - 1)),
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isComputing: false,
      error,
    })),
  };
}

export const pinn = createPINNStore();

// Derived stores
export const isComputing = derived(pinn, $pinn => $pinn.isComputing);
export const currentTime = derived(pinn, $pinn =>
  $pinn.timePoints[$pinn.currentTimeIndex] ?? 0
);
export const hypoxicVolume = derived(pinn, $pinn => $pinn.hypoxicVolume);

// Helper to get concentration at specific time slice
export function getTimeSlice(
  field: Float32Array | null,
  dims: PINNDimensions,
  timeIndex: number
): Float32Array | null {
  if (!field || timeIndex < 0 || timeIndex >= dims.nt) return null;

  const sliceSize = dims.nx * dims.ny * dims.nz;
  const start = timeIndex * sliceSize;

  return field.slice(start, start + sliceSize);
}

// Helper to convert 3D index to 1D
export function index3D(x: number, y: number, z: number, dims: PINNDimensions): number {
  return x + y * dims.nx + z * dims.nx * dims.ny;
}

// Helper to get value at position
export function getConcentration(
  field: Float32Array | null,
  dims: PINNDimensions,
  x: number,
  y: number,
  z: number,
  t: number
): number {
  if (!field) return 0;

  const sliceSize = dims.nx * dims.ny * dims.nz;
  const idx = t * sliceSize + index3D(x, y, z, dims);

  return field[idx] ?? 0;
}

// Color mapping for oxygen concentration
export function oxygenToColor(value: number, min: number = 0, max: number = 160): string {
  const normalized = (value - min) / (max - min);

  if (value < HYPOXIA_THRESHOLD) {
    // Hypoxic - red to dark red
    const intensity = value / HYPOXIA_THRESHOLD;
    return `rgb(${Math.round(139 + intensity * 116)}, 0, 0)`;
  }

  // Normal gradient: blue (low) -> green (medium) -> yellow (high)
  if (normalized < 0.5) {
    const t = normalized * 2;
    return `rgb(0, ${Math.round(t * 255)}, ${Math.round((1 - t) * 255)})`;
  } else {
    const t = (normalized - 0.5) * 2;
    return `rgb(${Math.round(t * 255)}, 255, 0)`;
  }
}
