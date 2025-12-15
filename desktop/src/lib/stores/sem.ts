import { writable, derived } from 'svelte/store';

export type CellType = 'osteoblast' | 'chondrocyte' | 'fibroblast' | 'endothelial' | 'macrophage' | 'stem_cell';

export interface CellClassification {
  predictedType: CellType;
  confidence: number;
  allProbabilities: Record<CellType, number>;
}

export interface MorphologyFeatures {
  area: number;
  perimeter: number;
  aspectRatio: number;
  circularity: number;
  textureVariance: number;
  neighborCount: number;
  perimeterRoughness: number;
}

export interface DetectedCell {
  id: number;
  boundingBox: { x: number; y: number; width: number; height: number };
  centroid: { x: number; y: number };
  classification: CellClassification;
  morphology: MorphologyFeatures;
}

export interface SEMState {
  imageUrl: string | null;
  imageSize: { width: number; height: number } | null;
  detectedCells: DetectedCell[];
  selectedCell: number | null;
  isClassifying: boolean;
  showOverlay: boolean;
  overlayOpacity: number;
  error: string | null;
}

// Cell type display info
export const CELL_TYPE_INFO: Record<CellType, { label: string; color: string; description: string }> = {
  osteoblast: { label: 'Osteoblast', color: '#f59e0b', description: 'Bone-forming cells' },
  chondrocyte: { label: 'Chondrocyte', color: '#3b82f6', description: 'Cartilage cells' },
  fibroblast: { label: 'Fibroblast', color: '#10b981', description: 'Connective tissue' },
  endothelial: { label: 'Endothelial', color: '#ec4899', description: 'Blood vessel lining' },
  macrophage: { label: 'Macrophage', color: '#8b5cf6', description: 'Immune cells' },
  stem_cell: { label: 'Stem Cell', color: '#06b6d4', description: 'Undifferentiated' },
};

const initialState: SEMState = {
  imageUrl: null,
  imageSize: null,
  detectedCells: [],
  selectedCell: null,
  isClassifying: false,
  showOverlay: true,
  overlayOpacity: 0.7,
  error: null,
};

function createSEMStore() {
  const { subscribe, set, update } = writable<SEMState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    setImage: (url: string, width: number, height: number) => update(state => ({
      ...state,
      imageUrl: url,
      imageSize: { width, height },
      detectedCells: [],
      selectedCell: null,
    })),

    startClassification: () => update(state => ({
      ...state,
      isClassifying: true,
      error: null,
    })),

    setDetectedCells: (cells: DetectedCell[]) => update(state => ({
      ...state,
      detectedCells: cells,
      isClassifying: false,
    })),

    selectCell: (cellId: number | null) => update(state => ({
      ...state,
      selectedCell: cellId,
    })),

    toggleOverlay: () => update(state => ({
      ...state,
      showOverlay: !state.showOverlay,
    })),

    setOverlayOpacity: (opacity: number) => update(state => ({
      ...state,
      overlayOpacity: Math.max(0, Math.min(1, opacity)),
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isClassifying: false,
      error,
    })),

    clearImage: () => update(state => ({
      ...state,
      imageUrl: null,
      imageSize: null,
      detectedCells: [],
      selectedCell: null,
    })),
  };
}

export const sem = createSEMStore();

// Derived stores
export const detectedCells = derived(sem, $sem => $sem.detectedCells);
export const selectedCell = derived(sem, $sem =>
  $sem.selectedCell !== null ? $sem.detectedCells.find(c => c.id === $sem.selectedCell) : null
);

// Cell type statistics
export const cellTypeStats = derived(sem, $sem => {
  const stats: Record<CellType, number> = {
    osteoblast: 0,
    chondrocyte: 0,
    fibroblast: 0,
    endothelial: 0,
    macrophage: 0,
    stem_cell: 0,
  };

  for (const cell of $sem.detectedCells) {
    stats[cell.classification.predictedType]++;
  }

  return stats;
});

// Helper to get confidence level
export function getConfidenceLevel(confidence: number): 'high' | 'medium' | 'low' {
  if (confidence >= 0.8) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}
