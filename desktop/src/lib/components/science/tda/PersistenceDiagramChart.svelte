<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import type { PersistencePoint } from '$lib/stores/tda';

  export let points: PersistencePoint[] = [];
  export let title: string = 'Persistence Diagram';
  export let color: string = '#4a9eff';
  export let width: number = 300;
  export let height: number = 300;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  const margin = { top: 30, right: 20, bottom: 40, left: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  $: if (svg && points) {
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

    // Add group for chart content
    const g = svg.append('g')
      .attr('class', 'chart-content')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add diagonal line (death = birth)
    g.append('line')
      .attr('class', 'diagonal')
      .attr('stroke', 'var(--border-color)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4');

    // Add axes groups
    g.append('g').attr('class', 'x-axis');
    g.append('g').attr('class', 'y-axis');

    // Add title
    svg.append('text')
      .attr('class', 'chart-title')
      .attr('x', width / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .text(title);

    // Add axis labels
    svg.append('text')
      .attr('class', 'x-label')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Birth');

    svg.append('text')
      .attr('class', 'y-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '11px')
      .text('Death');

    // Add points group
    g.append('g').attr('class', 'points');

    updateChart();
  }

  function updateChart() {
    if (!svg || !points.length) return;

    const g = svg.select('.chart-content');

    // Calculate domain
    const allValues = points.flatMap(p => [p.birth, p.death]);
    const maxVal = Math.max(...allValues, 1);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, maxVal])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, maxVal])
      .range([innerHeight, 0]);

    // Update diagonal
    g.select('.diagonal')
      .attr('x1', 0)
      .attr('y1', innerHeight)
      .attr('x2', innerWidth)
      .attr('y2', 0);

    // Update axes
    g.select('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px');

    g.select('.y-axis')
      .call(d3.axisLeft(yScale).ticks(5) as any)
      .selectAll('text')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', '10px');

    // Style axis lines
    g.selectAll('.domain, .tick line')
      .attr('stroke', 'var(--border-color)');

    // Update points with animation
    const pointsGroup = g.select('.points');

    const circles = pointsGroup.selectAll('circle')
      .data(points, (d: any) => `${d.birth}-${d.death}`);

    // Remove old points
    circles.exit()
      .transition()
      .duration(300)
      .attr('r', 0)
      .remove();

    // Add new points
    circles.enter()
      .append('circle')
      .attr('cx', d => xScale(d.birth))
      .attr('cy', d => yScale(d.death))
      .attr('r', 0)
      .attr('fill', color)
      .attr('fill-opacity', 0.7)
      .attr('stroke', color)
      .attr('stroke-width', 1)
      .on('mouseenter', function(event, d: any) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('r', 8);

        // Show tooltip
        const tooltip = d3.select(container).select('.tooltip');
        tooltip
          .style('opacity', 1)
          .style('left', `${event.offsetX + 10}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`Birth: ${d.birth.toFixed(3)}<br/>Death: ${d.death.toFixed(3)}<br/>Persistence: ${d.persistence.toFixed(3)}`);
      })
      .on('mouseleave', function() {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('r', 5);

        d3.select(container).select('.tooltip')
          .style('opacity', 0);
      })
      .transition()
      .duration(500)
      .attr('r', 5);

    // Update existing points
    circles
      .transition()
      .duration(500)
      .attr('cx', d => xScale(d.birth))
      .attr('cy', d => yScale(d.death));
  }
</script>

<div class="persistence-diagram" bind:this={container}>
  <div class="tooltip"></div>
</div>

<style>
  .persistence-diagram {
    position: relative;
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 8px;
  }

  .tooltip {
    position: absolute;
    background: var(--bg-elevated);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px;
    color: var(--text-primary);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 10;
    white-space: nowrap;
  }

  :global(.persistence-diagram .domain) {
    stroke: var(--border-color) !important;
  }

  :global(.persistence-diagram .tick line) {
    stroke: var(--border-color) !important;
  }
</style>
