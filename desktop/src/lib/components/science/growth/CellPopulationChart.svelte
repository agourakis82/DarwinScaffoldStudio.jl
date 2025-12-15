<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import { CellType, CELL_COLORS, type PopulationSnapshot } from '$lib/stores/tissueGrowth';

  export let history: PopulationSnapshot[] = [];
  export let currentTime: number = 0;
  export let width: number = 500;
  export let height: number = 250;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  const margin = { top: 20, right: 100, bottom: 40, left: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const cellTypes = [CellType.MSC, CellType.PREOSTEOBLAST, CellType.OSTEOBLAST, CellType.OSTEOCYTE];

  $: if (svg && history.length > 0) {
    updateChart();
  }

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

    // Add stacked area group
    g.append('g').attr('class', 'stacked-areas');

    // Add current time indicator
    g.append('line')
      .attr('class', 'time-indicator')
      .attr('stroke', 'var(--text-primary)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4');

    // Add axes
    g.append('g').attr('class', 'x-axis');
    g.append('g').attr('class', 'y-axis');

    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - margin.right + 10}, ${margin.top})`);

    cellTypes.forEach((type, i) => {
      const item = legend.append('g')
        .attr('transform', `translate(0, ${i * 22})`);

      item.append('rect')
        .attr('width', 14)
        .attr('height', 14)
        .attr('rx', 3)
        .attr('fill', CELL_COLORS[type]);

      item.append('text')
        .attr('x', 20)
        .attr('y', 11)
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', '10px')
        .text(type);
    });

    // Add Y axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Cell Count');

    // Add X axis label
    svg.append('text')
      .attr('x', margin.left + innerWidth / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Time (hours)');

    updateChart();
  }

  function updateChart() {
    if (!svg || history.length === 0) return;

    const g = svg.select('g');

    // Prepare data for stacking
    const stackedData = d3.stack<PopulationSnapshot>()
      .keys(cellTypes)
      .value((d, key) => d.counts[key] || 0)(history);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(history, d => d.time) || 168])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(history, d => d.totalCells) || 1000])
      .range([innerHeight, 0]);

    // Create area generator
    const area = d3.area<d3.SeriesPoint<PopulationSnapshot>>()
      .x(d => xScale(d.data.time))
      .y0(d => yScale(d[0]))
      .y1(d => yScale(d[1]))
      .curve(d3.curveMonotoneX);

    // Update areas
    const areas = g.select('.stacked-areas')
      .selectAll('path')
      .data(stackedData);

    areas.enter()
      .append('path')
      .merge(areas as any)
      .transition()
      .duration(300)
      .attr('d', area)
      .attr('fill', (d, i) => CELL_COLORS[cellTypes[i]])
      .attr('opacity', 0.8);

    areas.exit().remove();

    // Update time indicator
    g.select('.time-indicator')
      .attr('x1', xScale(currentTime))
      .attr('x2', xScale(currentTime))
      .attr('y1', 0)
      .attr('y2', innerHeight);

    // Update axes
    g.select('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(6) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px');

    g.select('.y-axis')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0s')) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px');

    // Style axis lines
    g.selectAll('.domain, .tick line')
      .attr('stroke', 'var(--border-color)');
  }
</script>

<div class="population-chart" bind:this={container}></div>

<style>
  .population-chart {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 12px;
  }

  :global(.population-chart .domain) {
    stroke: var(--border-color) !important;
  }

  :global(.population-chart .tick line) {
    stroke: var(--border-color) !important;
  }
</style>
