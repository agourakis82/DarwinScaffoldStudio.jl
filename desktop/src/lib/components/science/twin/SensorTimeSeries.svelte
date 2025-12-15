<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import type { SensorReading } from '$lib/stores/digitalTwin';
  import { SENSOR_RANGES } from '$lib/stores/digitalTwin';

  export let history: SensorReading[] = [];
  export let sensor: 'pH' | 'O2' | 'glucose' | 'temperature' = 'pH';
  export let width: number = 400;
  export let height: number = 200;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  const margin = { top: 20, right: 20, bottom: 30, left: 45 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  $: range = SENSOR_RANGES[sensor];
  $: data = history.map(r => ({ time: r.timestamp, value: r[sensor] }));
  $: if (svg && data) updateChart();

  onMount(() => {
    createChart();
  });

  onDestroy(() => {
    if (svg) svg.remove();
  });

  function createChart() {
    svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add gradient for area fill
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', `gradient-${sensor}`)
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', 'var(--primary)')
      .attr('stop-opacity', 0.3);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', 'var(--primary)')
      .attr('stop-opacity', 0);

    // Add optimal range band
    g.append('rect')
      .attr('class', 'optimal-band')
      .attr('fill', 'var(--success)')
      .attr('fill-opacity', 0.1);

    // Add area path
    g.append('path')
      .attr('class', 'area')
      .attr('fill', `url(#gradient-${sensor})`);

    // Add line path
    g.append('path')
      .attr('class', 'line')
      .attr('fill', 'none')
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2);

    // Add axes
    g.append('g').attr('class', 'x-axis');
    g.append('g').attr('class', 'y-axis');

    // Add Y axis label
    svg.append('text')
      .attr('class', 'y-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px')
      .text(`${sensor} ${range.unit ? `(${range.unit})` : ''}`);

    // Add current value indicator
    g.append('circle')
      .attr('class', 'current-dot')
      .attr('r', 5)
      .attr('fill', 'var(--primary)')
      .attr('stroke', 'var(--bg-primary)')
      .attr('stroke-width', 2);

    updateChart();
  }

  function updateChart() {
    if (!svg || data.length === 0) return;

    const g = svg.select('g');

    // Create scales
    const xExtent = d3.extent(data, d => d.time) as [Date, Date];
    const xScale = d3.scaleTime()
      .domain(xExtent)
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([range.min - (range.max - range.min) * 0.1, range.max + (range.max - range.min) * 0.1])
      .range([innerHeight, 0]);

    // Update optimal band
    g.select('.optimal-band')
      .attr('x', 0)
      .attr('y', yScale(range.max))
      .attr('width', innerWidth)
      .attr('height', yScale(range.min) - yScale(range.max));

    // Update area
    const area = d3.area<{ time: Date; value: number }>()
      .x(d => xScale(d.time))
      .y0(innerHeight)
      .y1(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.select('.area')
      .datum(data)
      .transition()
      .duration(300)
      .attr('d', area);

    // Update line
    const line = d3.line<{ time: Date; value: number }>()
      .x(d => xScale(d.time))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.select('.line')
      .datum(data)
      .transition()
      .duration(300)
      .attr('d', line);

    // Update current value dot
    const lastPoint = data[data.length - 1];
    if (lastPoint) {
      g.select('.current-dot')
        .transition()
        .duration(300)
        .attr('cx', xScale(lastPoint.time))
        .attr('cy', yScale(lastPoint.value));
    }

    // Update axes
    g.select('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat('%H:%M') as any) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '9px');

    g.select('.y-axis')
      .call(d3.axisLeft(yScale).ticks(5) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '9px');

    // Style axis lines
    g.selectAll('.domain, .tick line')
      .attr('stroke', 'var(--border-color)');
  }
</script>

<div class="time-series" bind:this={container}></div>

<style>
  .time-series {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 8px;
  }

  :global(.time-series .domain) {
    stroke: var(--border-color) !important;
  }

  :global(.time-series .tick line) {
    stroke: var(--border-color) !important;
  }
</style>
