import { writable, derived } from 'svelte/store';

export interface CausalNode {
  id: string;
  label: string;
  type: 'treatment' | 'outcome' | 'confounder' | 'mediator' | 'instrument' | 'variable';
  x?: number;
  y?: number;
}

export interface CausalEdge {
  source: string;
  target: string;
  type: 'directed' | 'bidirected';
  strength?: number;
}

export interface CausalGraph {
  nodes: CausalNode[];
  edges: CausalEdge[];
}

export interface CausalEffect {
  ate: number;
  standardError: number;
  ciLower: number;
  ciUpper: number;
  method: string;
  pValue: number;
}

export interface CausalState {
  graph: CausalGraph | null;
  treatment: string | null;
  outcome: string | null;
  effect: CausalEffect | null;
  adjustmentSet: string[];
  isDiscovering: boolean;
  isEstimating: boolean;
  algorithm: 'pc' | 'fci' | 'notears' | 'ges';
  error: string | null;
}

const initialState: CausalState = {
  graph: null,
  treatment: null,
  outcome: null,
  effect: null,
  adjustmentSet: [],
  isDiscovering: false,
  isEstimating: false,
  algorithm: 'pc',
  error: null,
};

function createCausalStore() {
  const { subscribe, set, update } = writable<CausalState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    setAlgorithm: (algo: CausalState['algorithm']) => update(state => ({
      ...state,
      algorithm: algo,
    })),

    startDiscovery: () => update(state => ({
      ...state,
      isDiscovering: true,
      error: null,
    })),

    setGraph: (graph: CausalGraph) => update(state => ({
      ...state,
      graph,
      isDiscovering: false,
    })),

    setTreatmentOutcome: (treatment: string, outcome: string) => update(state => ({
      ...state,
      treatment,
      outcome,
    })),

    startEstimation: () => update(state => ({
      ...state,
      isEstimating: true,
      error: null,
    })),

    setEffect: (effect: CausalEffect, adjustmentSet: string[]) => update(state => ({
      ...state,
      effect,
      adjustmentSet,
      isEstimating: false,
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isDiscovering: false,
      isEstimating: false,
      error,
    })),

    addNode: (node: CausalNode) => update(state => {
      if (!state.graph) return state;
      return {
        ...state,
        graph: {
          ...state.graph,
          nodes: [...state.graph.nodes, node],
        },
      };
    }),

    addEdge: (edge: CausalEdge) => update(state => {
      if (!state.graph) return state;
      return {
        ...state,
        graph: {
          ...state.graph,
          edges: [...state.graph.edges, edge],
        },
      };
    }),

    removeEdge: (source: string, target: string) => update(state => {
      if (!state.graph) return state;
      return {
        ...state,
        graph: {
          ...state.graph,
          edges: state.graph.edges.filter(e => !(e.source === source && e.target === target)),
        },
      };
    }),
  };
}

export const causal = createCausalStore();

// Derived stores
export const graph = derived(causal, $causal => $causal.graph);
export const effect = derived(causal, $causal => $causal.effect);
export const isDiscovering = derived(causal, $causal => $causal.isDiscovering);

// Helper to check if effect is significant
export function isSignificant(effect: CausalEffect, alpha: number = 0.05): boolean {
  return effect.pValue < alpha;
}

// Helper to format effect size
export function formatEffect(effect: CausalEffect): string {
  const sign = effect.ate >= 0 ? '+' : '';
  return `${sign}${effect.ate.toFixed(3)} (95% CI: ${effect.ciLower.toFixed(3)}, ${effect.ciUpper.toFixed(3)})`;
}
