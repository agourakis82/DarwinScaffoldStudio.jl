import { writable, derived } from 'svelte/store';

export interface GraphNode {
  id: number;
  x: number;
  y: number;
  z: number;
  features: number[];  // 8-dim feature vector
  importance: number;  // 0-1 normalized
  label?: string;
}

export interface GraphEdge {
  source: number;
  target: number;
  features: number[];  // 4-dim feature vector
  attention?: number;  // For GAT models
}

export interface ScaffoldGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  nNodes: number;
  nEdges: number;
  meanDegree: number;
}

export interface GNNPrediction {
  property: string;
  value: number;
  unit: string;
  confidence?: number;
}

export interface GNNState {
  graph: ScaffoldGraph | null;
  predictions: GNNPrediction[];
  nodeImportance: number[];
  selectedNode: number | null;
  modelType: 'gcn' | 'sage' | 'gat' | 'transformer';
  isLoading: boolean;
  isPredicting: boolean;
  error: string | null;
}

const initialState: GNNState = {
  graph: null,
  predictions: [],
  nodeImportance: [],
  selectedNode: null,
  modelType: 'gat',
  isLoading: false,
  isPredicting: false,
  error: null,
};

function createGNNStore() {
  const { subscribe, set, update } = writable<GNNState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    setModelType: (type: GNNState['modelType']) => update(state => ({
      ...state,
      modelType: type,
    })),

    startLoading: () => update(state => ({
      ...state,
      isLoading: true,
      error: null,
    })),

    setGraph: (graph: ScaffoldGraph) => update(state => ({
      ...state,
      graph,
      isLoading: false,
    })),

    startPrediction: () => update(state => ({
      ...state,
      isPredicting: true,
      error: null,
    })),

    setPredictions: (predictions: GNNPrediction[], nodeImportance: number[]) => update(state => ({
      ...state,
      predictions,
      nodeImportance,
      isPredicting: false,
    })),

    selectNode: (nodeId: number | null) => update(state => ({
      ...state,
      selectedNode: nodeId,
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isLoading: false,
      isPredicting: false,
      error,
    })),
  };
}

export const gnn = createGNNStore();

// Derived stores
export const graph = derived(gnn, $gnn => $gnn.graph);
export const predictions = derived(gnn, $gnn => $gnn.predictions);
export const nodeImportance = derived(gnn, $gnn => $gnn.nodeImportance);

// Helper to get importance color
export function importanceToColor(importance: number): string {
  // Blue (low) -> Yellow (medium) -> Red (high)
  if (importance < 0.5) {
    const t = importance * 2;
    return `rgb(${Math.round(t * 255)}, ${Math.round(t * 255)}, ${Math.round(255 - t * 127)})`;
  } else {
    const t = (importance - 0.5) * 2;
    return `rgb(255, ${Math.round(255 - t * 255)}, ${Math.round(128 - t * 128)})`;
  }
}

// Feature names for display
export const NODE_FEATURE_NAMES = [
  'Local Porosity',
  'X Position',
  'Y Position',
  'Z Position',
  'Surface Area',
  'Curvature',
  'Boundary Distance',
  'Cluster Size',
];

export const EDGE_FEATURE_NAMES = [
  'Distance',
  'Path Porosity',
  'Throat Width',
  'Tortuosity',
];
