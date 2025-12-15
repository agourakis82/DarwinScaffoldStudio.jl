<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import type { CausalGraph, CausalNode, CausalEdge } from '$lib/stores/causal';

  export let graph: CausalGraph | null = null;
  export let treatment: string | null = null;
  export let outcome: string | null = null;
  export let adjustmentSet: string[] = [];
  export let width: number = 600;
  export let height: number = 400;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  let simulation: d3.Simulation<CausalNode, CausalEdge>;

  const nodeRadius = 25;
  const nodeColors = {
    treatment: '#10b981',
    outcome: '#f59e0b',
    confounder: '#ef4444',
    mediator: '#8b5cf6',
    instrument: '#3b82f6',
    variable: '#6b7280',
  };

  $: if (svg && graph) {
    updateGraph();
  }

  onMount(() => {
    createSVG();
  });

  onDestroy(() => {
    if (simulation) simulation.stop();
  });

  function createSVG() {
    svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Arrow marker for directed edges
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', 'var(--text-secondary)');

    // Bidirected edge marker
    svg.select('defs').append('marker')
      .attr('id', 'bidirected')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('circle')
      .attr('cx', 5)
      .attr('cy', 0)
      .attr('r', 3)
      .attr('fill', 'var(--warning)');

    // Edge group
    svg.append('g').attr('class', 'edges');

    // Node group
    svg.append('g').attr('class', 'nodes');

    if (graph) updateGraph();
  }

  function updateGraph() {
    if (!svg || !graph) return;

    // Determine node types based on treatment/outcome selection
    const nodes = graph.nodes.map(n => ({
      ...n,
      type: n.id === treatment ? 'treatment' as const :
            n.id === outcome ? 'outcome' as const :
            adjustmentSet.includes(n.id) ? 'confounder' as const :
            n.type,
    }));

    const edges = graph.edges.map(e => ({ ...e }));

    // Create force simulation
    if (simulation) simulation.stop();

    simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(edges as any).id((d: any) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(nodeRadius + 10));

    // Update edges
    const edgeSelection = svg.select('.edges')
      .selectAll('path')
      .data(edges, (d: any) => `${d.source}-${d.target}`);

    edgeSelection.exit().remove();

    const edgeEnter = edgeSelection.enter()
      .append('path')
      .attr('fill', 'none')
      .attr('stroke', d => d.type === 'bidirected' ? 'var(--warning)' : 'var(--text-secondary)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', d => d.type === 'bidirected' ? '5,5' : 'none')
      .attr('marker-end', d => d.type === 'directed' ? 'url(#arrowhead)' : 'url(#bidirected)');

    const allEdges = edgeEnter.merge(edgeSelection as any);

    // Update nodes
    const nodeSelection = svg.select('.nodes')
      .selectAll('g.node')
      .data(nodes, (d: any) => d.id);

    nodeSelection.exit().remove();

    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragEnded) as any);

    nodeEnter.append('circle')
      .attr('r', nodeRadius)
      .attr('stroke', 'var(--bg-primary)')
      .attr('stroke-width', 3);

    nodeEnter.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '11px')
      .attr('font-weight', '600');

    const allNodes = nodeEnter.merge(nodeSelection as any);

    allNodes.select('circle')
      .attr('fill', (d: any) => nodeColors[d.type] || nodeColors.variable);

    allNodes.select('text')
      .text((d: any) => d.label || d.id);

    // Simulation tick
    simulation.on('tick', () => {
      allEdges.attr('d', (d: any) => {
        const sx = d.source.x;
        const sy = d.source.y;
        const tx = d.target.x;
        const ty = d.target.y;

        if (d.type === 'bidirected') {
          // Curved path for bidirected
          const midX = (sx + tx) / 2;
          const midY = (sy + ty) / 2 - 30;
          return `M${sx},${sy} Q${midX},${midY} ${tx},${ty}`;
        }
        return `M${sx},${sy} L${tx},${ty}`;
      });

      allNodes.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragStarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }
</script>

<div class="dag-viewer" bind:this={container}></div>

<style>
  .dag-viewer {
    background: var(--bg-secondary);
    border-radius: 12px;
    overflow: hidden;
  }

  :global(.dag-viewer .node) {
    cursor: grab;
  }

  :global(.dag-viewer .node:active) {
    cursor: grabbing;
  }
</style>
