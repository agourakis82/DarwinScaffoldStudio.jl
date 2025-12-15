<script lang="ts">
  import { SENSOR_RANGES, sensorHealth } from '$lib/stores/digitalTwin';

  export let value: number = 0;
  export let sensor: 'pH' | 'O2' | 'glucose' | 'temperature' = 'pH';
  export let size: number = 120;

  $: range = SENSOR_RANGES[sensor];
  $: health = sensorHealth(value, range);
  $: healthColor = health >= 80 ? 'var(--success)' : health >= 50 ? 'var(--warning)' : 'var(--error)';

  // SVG arc calculation
  const startAngle = -135;
  const endAngle = 135;
  const totalAngle = endAngle - startAngle;

  $: valueAngle = startAngle + (normalizedValue * totalAngle);
  $: normalizedValue = Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));

  function polarToCartesian(centerX: number, centerY: number, radius: number, angleInDegrees: number) {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + (radius * Math.cos(angleInRadians)),
      y: centerY + (radius * Math.sin(angleInRadians))
    };
  }

  function describeArc(x: number, y: number, radius: number, startAngle: number, endAngle: number) {
    const start = polarToCartesian(x, y, radius, endAngle);
    const end = polarToCartesian(x, y, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
    return [
      "M", start.x, start.y,
      "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y
    ].join(" ");
  }

  $: center = size / 2;
  $: radius = size * 0.38;
  $: backgroundArc = describeArc(center, center, radius, startAngle, endAngle);
  $: valueArc = describeArc(center, center, radius, startAngle, valueAngle);

  const sensorLabels = {
    pH: { name: 'pH', icon: 'droplet' },
    O2: { name: 'Oxygen', icon: 'wind' },
    glucose: { name: 'Glucose', icon: 'activity' },
    temperature: { name: 'Temp', icon: 'thermometer' },
  };
</script>

<div class="gauge" style="width: {size}px; height: {size}px;">
  <svg width={size} height={size} viewBox="0 0 {size} {size}">
    <!-- Background arc -->
    <path
      d={backgroundArc}
      fill="none"
      stroke="var(--bg-tertiary)"
      stroke-width="10"
      stroke-linecap="round"
    />

    <!-- Value arc -->
    <path
      d={valueArc}
      fill="none"
      stroke={healthColor}
      stroke-width="10"
      stroke-linecap="round"
      style="transition: all 0.5s ease-out"
    />

    <!-- Min/Max labels -->
    <text
      x={polarToCartesian(center, center, radius + 15, startAngle).x}
      y={polarToCartesian(center, center, radius + 15, startAngle).y}
      text-anchor="middle"
      fill="var(--text-muted)"
      font-size="9"
    >
      {range.min}
    </text>
    <text
      x={polarToCartesian(center, center, radius + 15, endAngle).x}
      y={polarToCartesian(center, center, radius + 15, endAngle).y}
      text-anchor="middle"
      fill="var(--text-muted)"
      font-size="9"
    >
      {range.max}
    </text>
  </svg>

  <div class="gauge-content">
    <span class="gauge-value" style="color: {healthColor}">
      {value.toFixed(sensor === 'pH' ? 2 : 1)}
    </span>
    <span class="gauge-unit">{range.unit}</span>
    <span class="gauge-label">{sensorLabels[sensor].name}</span>
  </div>

  {#if health < 50}
    <div class="alert-badge" style="background: {healthColor}">!</div>
  {/if}
</div>

<style>
  .gauge {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-secondary);
    border-radius: 12px;
  }

  .gauge-content {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin-top: 10px;
  }

  .gauge-value {
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
  }

  .gauge-unit {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .gauge-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
  }

  .alert-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: white;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
