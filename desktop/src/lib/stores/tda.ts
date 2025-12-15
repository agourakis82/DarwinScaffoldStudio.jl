import { writable, derived } from 'svelte/store';

export interface PersistencePoint {
  birth: number;
  death: number;
  persistence: number;
}

export interface PersistenceDiagrams {
  H0: PersistencePoint[]; // Connected components
  H1: PersistencePoint[]; // Loops/tunnels
  H2: PersistencePoint[]; // Voids/cavities
}

export interface BettiNumbers {
  beta0: number; // Connected components count
  beta1: number; // Tunnels count
  beta2: number; // Voids count
}

export interface TDAState {
  diagrams: PersistenceDiagrams;
  bettiNumbers: BettiNumbers;
  interconnectivityScore: number;
  isAnalyzing: boolean;
  error: string | null;
  lastAnalyzed: Date | null;
}

const initialState: TDAState = {
  diagrams: { H0: [], H1: [], H2: [] },
  bettiNumbers: { beta0: 0, beta1: 0, beta2: 0 },
  interconnectivityScore: 0,
  isAnalyzing: false,
  error: null,
  lastAnalyzed: null,
};

function createTDAStore() {
  const { subscribe, set, update } = writable<TDAState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    startAnalysis: () => update(state => ({
      ...state,
      isAnalyzing: true,
      error: null,
    })),

    setResults: (diagrams: PersistenceDiagrams, bettiNumbers: BettiNumbers, score: number) => update(state => ({
      ...state,
      diagrams,
      bettiNumbers,
      interconnectivityScore: score,
      isAnalyzing: false,
      lastAnalyzed: new Date(),
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isAnalyzing: false,
      error,
    })),
  };
}

export const tda = createTDAStore();

// Derived stores for convenience
export const isAnalyzing = derived(tda, $tda => $tda.isAnalyzing);
export const diagrams = derived(tda, $tda => $tda.diagrams);
export const bettiNumbers = derived(tda, $tda => $tda.bettiNumbers);
export const interconnectivityScore = derived(tda, $tda => $tda.interconnectivityScore);

// Helper to interpret Betti numbers
export function interpretBettiNumbers(betti: BettiNumbers): string {
  const parts: string[] = [];

  if (betti.beta0 === 1) {
    parts.push('Scaffold is fully connected');
  } else if (betti.beta0 > 1) {
    parts.push(`${betti.beta0} disconnected regions detected`);
  }

  if (betti.beta1 > 0) {
    parts.push(`${betti.beta1} through-channels for nutrient transport`);
  }

  if (betti.beta2 > 0) {
    parts.push(`${betti.beta2} enclosed voids may trap cells`);
  }

  return parts.join('. ') || 'No significant topological features detected.';
}

// Helper to score interconnectivity (0-100)
export function scoreInterconnectivity(betti: BettiNumbers, targetPorosity: number = 0.85): number {
  // Good interconnectivity: beta0 = 1 (connected), high beta1 (tunnels), low beta2 (no trapped voids)
  let score = 100;

  // Penalize disconnected components
  if (betti.beta0 > 1) {
    score -= (betti.beta0 - 1) * 15;
  }

  // Reward tunnels (up to a point)
  const optimalTunnels = 50;
  if (betti.beta1 < optimalTunnels) {
    score -= (optimalTunnels - betti.beta1) * 0.5;
  }

  // Penalize trapped voids
  score -= betti.beta2 * 5;

  return Math.max(0, Math.min(100, score));
}
