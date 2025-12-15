import { writable, derived } from 'svelte/store';

export enum CellType {
  MSC = 'MSC',                   // Mesenchymal Stem Cell
  PREOSTEOBLAST = 'PREOSTEOBLAST',
  OSTEOBLAST = 'OSTEOBLAST',
  OSTEOCYTE = 'OSTEOCYTE',
  DEAD = 'DEAD',
}

export interface Cell {
  id: number;
  type: CellType;
  x: number;
  y: number;
  z: number;
  age: number;       // hours since creation
  health: number;    // 0-1
  divisionReady: boolean;
}

export interface PopulationSnapshot {
  time: number; // hours
  counts: Record<CellType, number>;
  totalCells: number;
}

export interface SimulationConfig {
  initialCellCount: number;
  simulationHours: number;
  divisionRate: number;    // divisions per hour
  differentiationRate: number;
  deathRate: number;
  scaffoldVolume: number;  // mm^3
}

export interface TissueGrowthState {
  simulationId: string | null;
  cells: Cell[];
  populationHistory: PopulationSnapshot[];
  currentTimeIndex: number;
  maxTime: number;
  isPlaying: boolean;
  playbackSpeed: number; // multiplier
  isSimulating: boolean;
  error: string | null;
}

export const DEFAULT_CONFIG: SimulationConfig = {
  initialCellCount: 1000,
  simulationHours: 168, // 1 week
  divisionRate: 0.04,   // ~once per day
  differentiationRate: 0.02,
  deathRate: 0.005,
  scaffoldVolume: 27,   // 3x3x3 mm
};

// Cell type colors for visualization
export const CELL_COLORS: Record<CellType, string> = {
  [CellType.MSC]: '#3b82f6',          // Blue
  [CellType.PREOSTEOBLAST]: '#8b5cf6', // Purple
  [CellType.OSTEOBLAST]: '#10b981',    // Green
  [CellType.OSTEOCYTE]: '#f59e0b',     // Amber
  [CellType.DEAD]: '#6b7280',          // Gray
};

const initialState: TissueGrowthState = {
  simulationId: null,
  cells: [],
  populationHistory: [],
  currentTimeIndex: 0,
  maxTime: 0,
  isPlaying: false,
  playbackSpeed: 1,
  isSimulating: false,
  error: null,
};

function createTissueGrowthStore() {
  const { subscribe, set, update } = writable<TissueGrowthState>(initialState);

  let playbackInterval: ReturnType<typeof setInterval> | null = null;

  const store = {
    subscribe,
    reset: () => {
      if (playbackInterval) clearInterval(playbackInterval);
      set(initialState);
    },

    startSimulation: () => update(state => ({
      ...state,
      isSimulating: true,
      error: null,
    })),

    setResults: (
      simulationId: string,
      cellsOverTime: Cell[][],
      populationHistory: PopulationSnapshot[]
    ) => update(state => ({
      ...state,
      simulationId,
      cells: cellsOverTime[0] || [],
      populationHistory,
      currentTimeIndex: 0,
      maxTime: populationHistory.length - 1,
      isSimulating: false,
    })),

    setCellsAtTime: (cells: Cell[]) => update(state => ({
      ...state,
      cells,
    })),

    setTimeIndex: (index: number) => update(state => ({
      ...state,
      currentTimeIndex: Math.max(0, Math.min(index, state.maxTime)),
    })),

    play: () => {
      update(state => ({ ...state, isPlaying: true }));

      playbackInterval = setInterval(() => {
        update(state => {
          const nextIndex = state.currentTimeIndex + 1;
          if (nextIndex > state.maxTime) {
            store.pause();
            return { ...state, currentTimeIndex: 0 };
          }
          return { ...state, currentTimeIndex: nextIndex };
        });
      }, 100); // 10 fps base, modified by playbackSpeed
    },

    pause: () => {
      if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
      }
      update(state => ({ ...state, isPlaying: false }));
    },

    setPlaybackSpeed: (speed: number) => update(state => ({
      ...state,
      playbackSpeed: Math.max(0.25, Math.min(4, speed)),
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isSimulating: false,
      error,
    })),
  };

  return store;
}

export const tissueGrowth = createTissueGrowthStore();

// Derived stores
export const isPlaying = derived(tissueGrowth, $tg => $tg.isPlaying);
export const isSimulating = derived(tissueGrowth, $tg => $tg.isSimulating);
export const currentPopulation = derived(tissueGrowth, $tg =>
  $tg.populationHistory[$tg.currentTimeIndex]
);

// Helper to get differentiation pathway
export function getDifferentiationPath(fromType: CellType): CellType | null {
  const pathway: Record<CellType, CellType | null> = {
    [CellType.MSC]: CellType.PREOSTEOBLAST,
    [CellType.PREOSTEOBLAST]: CellType.OSTEOBLAST,
    [CellType.OSTEOBLAST]: CellType.OSTEOCYTE,
    [CellType.OSTEOCYTE]: null,
    [CellType.DEAD]: null,
  };
  return pathway[fromType];
}

// Helper to calculate growth metrics
export function calculateGrowthRate(history: PopulationSnapshot[]): number {
  if (history.length < 2) return 0;

  const first = history[0].totalCells;
  const last = history[history.length - 1].totalCells;
  const hours = history[history.length - 1].time - history[0].time;

  if (hours === 0 || first === 0) return 0;

  // Exponential growth rate
  return Math.log(last / first) / hours;
}

// Helper to calculate differentiation progress
export function differentiationProgress(snapshot: PopulationSnapshot): number {
  const { counts, totalCells } = snapshot;
  if (totalCells === 0) return 0;

  // Weight by differentiation stage
  const weights = {
    [CellType.MSC]: 0,
    [CellType.PREOSTEOBLAST]: 0.33,
    [CellType.OSTEOBLAST]: 0.66,
    [CellType.OSTEOCYTE]: 1,
    [CellType.DEAD]: 0,
  };

  let weighted = 0;
  let living = 0;

  for (const [type, count] of Object.entries(counts)) {
    if (type !== CellType.DEAD) {
      weighted += (weights[type as CellType] || 0) * count;
      living += count;
    }
  }

  return living > 0 ? weighted / living : 0;
}
