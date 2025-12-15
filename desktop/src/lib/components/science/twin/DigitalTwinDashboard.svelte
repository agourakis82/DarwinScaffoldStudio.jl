<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { digitalTwin, SENSOR_RANGES, type SensorReading, type Prediction } from '$lib/stores/digitalTwin';
  import SensorGauge from './SensorGauge.svelte';
  import SensorTimeSeries from './SensorTimeSeries.svelte';
  import AnomalyIndicator from './AnomalyIndicator.svelte';
  import PredictionChart from './PredictionChart.svelte';

  let selectedSensor: 'pH' | 'O2' | 'glucose' | 'temperature' = 'pH';
  let updateInterval: ReturnType<typeof setInterval>;

  onMount(() => {
    // Try to connect to live twin
    connectToTwin();

    // Poll for updates
    updateInterval = setInterval(fetchSensorData, 5000);
  });

  onDestroy(() => {
    if (updateInterval) clearInterval(updateInterval);
  });

  async function connectToTwin() {
    digitalTwin.connect('twin-001');

    try {
      const response = await fetch('http://localhost:8081/twin/twin-001/state');
      if (response.ok) {
        digitalTwin.setConnected(true);
        await fetchSensorData();
        await fetchPredictions();
      }
    } catch {
      // Use demo data
      simulateSensorData();
      digitalTwin.setConnected(false);
    }
  }

  async function fetchSensorData() {
    try {
      const response = await fetch('http://localhost:8081/twin/twin-001/state');
      if (!response.ok) throw new Error('Failed to fetch');

      const data = await response.json();
      digitalTwin.updateSensors({
        timestamp: new Date(),
        pH: data.pH,
        O2: data.O2,
        glucose: data.glucose,
        temperature: data.temperature,
      });
    } catch {
      // Continue with simulation
      simulateSensorData();
    }
  }

  async function fetchPredictions() {
    try {
      const predictions: Prediction[] = [];

      for (const hours of [6, 12, 24]) {
        const response = await fetch(`http://localhost:8081/twin/twin-001/predict/${hours}`);
        if (response.ok) {
          const data = await response.json();
          predictions.push({
            timestamp: new Date(data.timestamp),
            hoursAhead: hours,
            pH: data.pH,
            O2: data.O2,
            glucose: data.glucose,
            confidence: data.confidence,
          });
        }
      }

      if (predictions.length > 0) {
        digitalTwin.setPredictions(predictions);
      }
    } catch {
      // Use simulated predictions
      simulatePredictions();
    }
  }

  function simulateSensorData() {
    const baseValues = {
      pH: 7.35 + (Math.random() - 0.5) * 0.1,
      O2: 18 + (Math.random() - 0.5) * 4,
      glucose: 5.5 + (Math.random() - 0.5) * 1.5,
      temperature: 37 + (Math.random() - 0.5) * 0.4,
    };

    digitalTwin.updateSensors({
      timestamp: new Date(),
      ...baseValues,
    });
  }

  function simulatePredictions() {
    const current = $digitalTwin.current;
    if (!current) return;

    const predictions: Prediction[] = [6, 12, 24].map(hours => {
      const drift = (hours / 24) * 0.1;
      return {
        timestamp: new Date(Date.now() + hours * 3600000),
        hoursAhead: hours,
        pH: current.pH + (Math.random() - 0.5) * drift,
        O2: Math.max(10, current.O2 - Math.random() * drift * 20),
        glucose: Math.max(3, current.glucose - Math.random() * drift * 5),
        confidence: Math.max(0.5, 0.95 - hours * 0.015),
      };
    });

    digitalTwin.setPredictions(predictions);
  }

  // Initial simulation
  $: if ($digitalTwin.history.length === 0) {
    simulateSensorData();
    simulatePredictions();
  }
</script>

<div class="twin-dashboard">
  <header class="dashboard-header">
    <div class="header-title">
      <h1>Digital Twin Dashboard</h1>
      <p>Real-time biosensor monitoring and predictive analytics</p>
    </div>

    <div class="connection-status" class:connected={$digitalTwin.isConnected}>
      <span class="status-dot"></span>
      <span>{$digitalTwin.isConnected ? 'Live' : 'Simulation'}</span>
    </div>
  </header>

  <div class="dashboard-grid">
    <!-- Sensor Gauges -->
    <section class="gauges-section">
      <h2 class="section-title">Current Readings</h2>
      <div class="gauges-grid">
        <div class="gauge-wrapper" on:click={() => selectedSensor = 'pH'} class:selected={selectedSensor === 'pH'}>
          <SensorGauge value={$digitalTwin.current?.pH ?? 7.35} sensor="pH" />
        </div>
        <div class="gauge-wrapper" on:click={() => selectedSensor = 'O2'} class:selected={selectedSensor === 'O2'}>
          <SensorGauge value={$digitalTwin.current?.O2 ?? 20} sensor="O2" />
        </div>
        <div class="gauge-wrapper" on:click={() => selectedSensor = 'glucose'} class:selected={selectedSensor === 'glucose'}>
          <SensorGauge value={$digitalTwin.current?.glucose ?? 5.5} sensor="glucose" />
        </div>
        <div class="gauge-wrapper" on:click={() => selectedSensor = 'temperature'} class:selected={selectedSensor === 'temperature'}>
          <SensorGauge value={$digitalTwin.current?.temperature ?? 37} sensor="temperature" />
        </div>
      </div>
    </section>

    <!-- Anomaly Indicator -->
    <section class="anomaly-section">
      <h2 class="section-title">System Status</h2>
      <AnomalyIndicator anomalies={$digitalTwin.anomalies} />
    </section>

    <!-- Time Series Chart -->
    <section class="chart-section">
      <h2 class="section-title">
        {selectedSensor.toUpperCase()} History
        <span class="section-subtitle">Last {$digitalTwin.history.length} readings</span>
      </h2>
      <SensorTimeSeries
        history={$digitalTwin.history}
        sensor={selectedSensor}
        width={600}
        height={200}
      />
    </section>

    <!-- Predictions -->
    <section class="predictions-section">
      <h2 class="section-title">Predictive Forecast</h2>
      <div class="predictions-grid">
        <PredictionChart predictions={$digitalTwin.predictions} sensor="pH" />
        <PredictionChart predictions={$digitalTwin.predictions} sensor="O2" />
        <PredictionChart predictions={$digitalTwin.predictions} sensor="glucose" />
      </div>
    </section>
  </div>
</div>

<style>
  .twin-dashboard {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
  }

  .header-title h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
  }

  .header-title p {
    font-size: 13px;
    color: var(--text-muted);
    margin: 0;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 16px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .connection-status.connected {
    color: var(--success);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
  }

  .connection-status.connected .status-dot {
    background: var(--success);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 300px;
    grid-template-rows: auto auto auto;
    gap: 20px;
    flex: 1;
    overflow: auto;
  }

  .section-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 12px 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .section-subtitle {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-muted);
  }

  .gauges-section {
    grid-column: 1;
    grid-row: 1;
  }

  .gauges-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .gauge-wrapper {
    cursor: pointer;
    border-radius: 12px;
    border: 2px solid transparent;
    transition: all var(--transition-fast);
  }

  .gauge-wrapper:hover {
    border-color: var(--border-color);
  }

  .gauge-wrapper.selected {
    border-color: var(--primary);
  }

  .anomaly-section {
    grid-column: 2;
    grid-row: 1 / 3;
  }

  .chart-section {
    grid-column: 1;
    grid-row: 2;
  }

  .predictions-section {
    grid-column: 1 / 3;
    grid-row: 3;
  }

  .predictions-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }

  @media (max-width: 1000px) {
    .dashboard-grid {
      grid-template-columns: 1fr;
    }

    .gauges-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .anomaly-section {
      grid-column: 1;
      grid-row: auto;
    }

    .predictions-section {
      grid-column: 1;
    }

    .predictions-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
