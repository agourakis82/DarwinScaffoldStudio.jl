<p align="center">
  <h1 align="center">Darwin Scaffold Studio</h1>
  <p align="center">
    <strong>Computational Platform for Tissue Engineering Scaffold Analysis</strong>
  </p>
</p>

<p align="center">
  <a href="https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml">
    <img src="https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://github.com/agourakis82/darwin-scaffold-studio/releases/latest">
    <img src="https://img.shields.io/github/v/release/agourakis82/darwin-scaffold-studio" alt="Latest Release">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://www.julia-lang.org/">
    <img src="https://img.shields.io/badge/Julia-1.10+-9558B2.svg?logo=julia" alt="Julia 1.10+">
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

Darwin Scaffold Studio is an open-source computational platform for analyzing and optimizing tissue engineering scaffolds from MicroCT and SEM imaging data. It integrates biomedical ontologies, validated metrics computation, and AI-assisted design optimization.

### Key Capabilities

- **Image Analysis**: Load and process MicroCT/SEM data with automated segmentation
- **Metrics Computation**: Porosity, pore size, interconnectivity, tortuosity, mechanical properties
- **Ontology Integration**: 1200+ biomedical terms from OBO Foundry (UBERON, CL, CHEBI)
- **Design Optimization**: Target-driven scaffold optimization with fabrication recommendations
- **FAIR Data Export**: Schema.org compatible JSON-LD with full provenance tracking

---

## Features

### Metrics

| Metric | Method | Validation Status |
|--------|--------|-------------------|
| Porosity | Voxel counting | <1% error (synthetic) |
| Surface Area | Marching cubes | <1% error (synthetic) |
| Pore Size | Connected components + Otsu | 14% APE (real SEM data) |
| Interconnectivity | Connected components | Theoretical validation |
| Tortuosity | Dijkstra shortest path | Theoretical validation |
| Mechanical Properties | Gibson-Ashby model | Literature-based |

### Ontology Integration

```julia
# Lookup optimal parameters for bone tissue
bone = OntologyManager.lookup_tissue("bone")
# Returns: UBERON:0002481, optimal porosity 85-95%, pore size 100-300μm

# Get cell requirements
osteoblast = OntologyManager.lookup_cell("osteoblast")
# Returns: CL:0000062, size 20-30μm, markers, growth factors

# Material properties
ha = OntologyManager.lookup_material("hydroxyapatite")
# Returns: CHEBI:52254, E=80-120 GPa, biocompatibility: excellent
```

### TPMS Scaffold Generation

Analytical Triply Periodic Minimal Surfaces for validation:
- **Gyroid**: High surface area, interconnected pores
- **Diamond (Schwarz D)**: Isotropic mechanical properties
- **Schwarz P**: Simple cubic symmetry
- **Neovius**: Complex multi-scale porosity

---

## Installation

### Requirements

- Julia 1.10 or higher
- 8GB RAM minimum (16GB recommended for large datasets)

### Option 1: Julia Package

```julia
using Pkg
Pkg.add(url="https://github.com/agourakis82/darwin-scaffold-studio")
```

### Option 2: Development Setup

```bash
git clone https://github.com/agourakis82/darwin-scaffold-studio.git
cd darwin-scaffold-studio
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Option 3: Docker

```bash
docker build -t darwin-scaffold-studio .
docker run -it -v $(pwd)/data:/app/user_data darwin-scaffold-studio
```

---

## Quick Start

```julia
# Load the module
include("src/DarwinScaffoldStudio.jl")
using .DarwinScaffoldStudio

# 1. Load MicroCT data
img = load_microct("scaffold.raw", (512, 512, 512))

# 2. Preprocess and segment
processed = preprocess_image(img; denoise=true, normalize=true)
binary = segment_scaffold(processed, "otsu")

# 3. Compute metrics
metrics = compute_metrics(binary, 10.0)  # 10 μm voxel size

println("Porosity: $(round(metrics.porosity * 100, digits=1))%")
println("Pore size: $(round(metrics.mean_pore_size_um, digits=1)) μm")
println("Interconnectivity: $(round(metrics.interconnectivity * 100, digits=1))%")

# 4. Validate against literature
bone = OntologyManager.lookup_tissue("bone")
if bone.optimal_porosity[1] <= metrics.porosity <= bone.optimal_porosity[2]
    println("✓ Porosity optimal for bone tissue engineering")
end

# 5. Export mesh for 3D printing
vertices, faces = create_mesh_simple(binary, 10.0)
export_stl("scaffold.stl", vertices, faces)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Tutorial](docs/guides/tutorial.md) | Complete end-to-end workflow |
| [API Reference](docs/reference/api.md) | Full function documentation |
| [Materials Reference](docs/reference/EXTENDED_MATERIALS_REFERENCE.md) | Biomaterial properties database |

### Architecture

```
DarwinScaffoldStudio/
├── Core/           # Types, Config, Utils
├── MicroCT/        # Image loading, segmentation, metrics
├── Optimization/   # Scaffold optimization algorithms
├── Visualization/  # Mesh generation, export
├── Science/        # Topology, percolation, ML
├── Ontology/       # OBO Foundry integration
└── Agents/         # AI-assisted analysis (optional)
```

---

## Validation

### Synthetic Ground Truth (TPMS Scaffolds)

Validation against analytical TPMS surfaces with known geometry:

| Metric | Mean Error | Threshold | Status |
|--------|-----------|-----------|--------|
| Porosity | <1% | <1% | PASS |
| Surface Area | <1% | <1% | PASS |

### Real Experimental Data (PoreScript Dataset)

Validation against manual measurements from SEM images (DOI: 10.5281/zenodo.5562953):

| Metric | Darwin | Ground Truth | APE |
|--------|--------|--------------|-----|
| Pore Size | 149.4 um | 174.0 um | 14.1% |

**Limitations:**
- Systematic underestimation of ~15% on pore size
- Validated on 3 SEM images only
- 2D analysis (SEM), not 3D (microCT)

```bash
# Run validation
julia --project=. scripts/validate_honest.jl
```

See [docs/validation/](docs/validation/) for detailed validation reports.

---

## Citation

If you use Darwin Scaffold Studio in your research, please cite:

```bibtex
@software{darwin_scaffold_studio,
  author = {Agourakis, Demetrios},
  title = {Darwin Scaffold Studio: Computational Platform for Tissue Engineering},
  year = {2024},
  url = {https://github.com/agourakis82/darwin-scaffold-studio},
  version = {0.2.1}
}
```

See [CITATION.cff](CITATION.cff) for full citation information.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/development/CONTRIBUTING.md) for guidelines.

### Development

```bash
# Run tests
julia --project=. test/runtests.jl

# Quick tests (CI)
julia --project=. test/test_quick.jl
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- PUC-SP Biomaterials and Regenerative Medicine Program
- OBO Foundry for standardized biomedical ontologies
- Julia community for scientific computing ecosystem

---

<p align="center">
  <sub>Built with Julia for reproducible tissue engineering research</sub>
</p>
