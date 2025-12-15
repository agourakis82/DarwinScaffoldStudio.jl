import { writable, derived } from 'svelte/store';

export interface SensorReading {
  timestamp: Date;
  pH: number;
  O2: number;      // mmHg or %
  glucose: number; // mM
  temperature: number; // Celsius
  cellDensity?: number;
  mechanicalStrain?: number;
}

export interface Anomaly {
  sensor: string;
  value: number;
  threshold: { min: number; max: number };
  severity: 'warning' | 'critical';
  timestamp: Date;
}

export interface Prediction {
  timestamp: Date;
  hoursAhead: number;
  pH: number;
  O2: number;
  glucose: number;
  confidence: number;
}

export interface DigitalTwinState {
  twinId: string | null;
  current: SensorReading | null;
  history: SensorReading[];
  predictions: Prediction[];
  anomalies: Anomaly[];
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

// Optimal ranges for tissue culture
export const SENSOR_RANGES = {
  pH: { min: 7.2, max: 7.4, unit: '', optimal: 7.35 },
  O2: { min: 15, max: 21, unit: '%', optimal: 20 },
  glucose: { min: 4, max: 8, unit: 'mM', optimal: 5.5 },
  temperature: { min: 36.5, max: 37.5, unit: 'C', optimal: 37 },
};

const initialState: DigitalTwinState = {
  twinId: null,
  current: null,
  history: [],
  predictions: [],
  anomalies: [],
  isConnected: false,
  isLoading: false,
  error: null,
};

function createDigitalTwinStore() {
  const { subscribe, set, update } = writable<DigitalTwinState>(initialState);

  return {
    subscribe,
    reset: () => set(initialState),

    connect: (twinId: string) => update(state => ({
      ...state,
      twinId,
      isLoading: true,
      error: null,
    })),

    setConnected: (connected: boolean) => update(state => ({
      ...state,
      isConnected: connected,
      isLoading: false,
    })),

    updateSensors: (reading: SensorReading) => update(state => {
      const anomalies = detectAnomalies(reading);
      return {
        ...state,
        current: reading,
        history: [...state.history.slice(-99), reading], // Keep last 100
        anomalies: [...state.anomalies.filter(a =>
          Date.now() - a.timestamp.getTime() < 3600000 // Keep last hour
        ), ...anomalies],
      };
    }),

    setPredictions: (predictions: Prediction[]) => update(state => ({
      ...state,
      predictions,
    })),

    setError: (error: string) => update(state => ({
      ...state,
      isLoading: false,
      error,
    })),

    clearAnomalies: () => update(state => ({
      ...state,
      anomalies: [],
    })),
  };
}

function detectAnomalies(reading: SensorReading): Anomaly[] {
  const anomalies: Anomaly[] = [];
  const now = new Date();

  const checkSensor = (name: string, value: number, range: typeof SENSOR_RANGES.pH) => {
    if (value < range.min || value > range.max) {
      const deviation = value < range.min ? range.min - value : value - range.max;
      const maxDeviation = (range.max - range.min) / 2;
      const severity = deviation > maxDeviation ? 'critical' : 'warning';

      anomalies.push({
        sensor: name,
        value,
        threshold: { min: range.min, max: range.max },
        severity,
        timestamp: now,
      });
    }
  };

  checkSensor('pH', reading.pH, SENSOR_RANGES.pH);
  checkSensor('O2', reading.O2, SENSOR_RANGES.O2);
  checkSensor('glucose', reading.glucose, SENSOR_RANGES.glucose);
  checkSensor('temperature', reading.temperature, SENSOR_RANGES.temperature);

  return anomalies;
}

export const digitalTwin = createDigitalTwinStore();

// Derived stores
export const currentSensors = derived(digitalTwin, $twin => $twin.current);
export const sensorHistory = derived(digitalTwin, $twin => $twin.history);
export const activeAnomalies = derived(digitalTwin, $twin => $twin.anomalies);
export const predictions = derived(digitalTwin, $twin => $twin.predictions);

// Helper to calculate sensor health (0-100)
export function sensorHealth(value: number, range: { min: number; max: number; optimal: number }): number {
  if (value < range.min || value > range.max) return 0;

  const deviation = Math.abs(value - range.optimal);
  const maxDeviation = Math.max(range.optimal - range.min, range.max - range.optimal);

  return Math.round((1 - deviation / maxDeviation) * 100);
}
