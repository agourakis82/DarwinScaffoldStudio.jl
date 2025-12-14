# DarwinScaffoldStudio.jl

Julia package for analyzing tissue engineering scaffolds from MicroCT and SEM imaging data.

[![CI](https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("DarwinScaffoldStudio")
```

Or clone and install locally:

```bash
git clone https://github.com/agourakis82/darwin-scaffold-studio.git
cd darwin-scaffold-studio
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

```julia
using DarwinScaffoldStudio

# Load MicroCT data
img = load_microct("scaffold.raw", (512, 512, 512))

# Preprocess and segment
processed = preprocess_image(img)
binary = segment_scaffold(processed, "otsu")

# Compute scaffold metrics (10 μm voxel size)
metrics = compute_metrics(binary, 10.0)

println("Porosity: $(round(metrics.porosity * 100, digits=1))%")
println("Pore size: $(round(metrics.mean_pore_size_um, digits=1)) μm")

# Export mesh for 3D printing
vertices, faces = create_mesh_simple(binary, 10.0)
export_stl("scaffold.stl", vertices, faces)
```

## Features

**Image Processing**
- Load MicroCT/SEM data (RAW, TIFF, NIfTI)
- Denoising, normalization, artifact removal
- Segmentation via Otsu, adaptive thresholding, watershed

**Scaffold Metrics**
- Porosity (voxel counting)
- Surface area (marching cubes)
- Pore size distribution (connected components)
- Interconnectivity (graph analysis)
- Tortuosity (Dijkstra paths)

**Additional Tools**
- TPMS surface generation (Gyroid, Schwarz D/P, Neovius)
- Topological analysis (Betti numbers, persistent homology)
- STL/OBJ mesh export

## Documentation

- [MANUAL.md](MANUAL.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started

## Citation

```bibtex
@software{darwin_scaffold_studio,
  author = {Agourakis, Demetrios Chiuratto},
  title = {DarwinScaffoldStudio.jl},
  year = {2025},
  url = {https://github.com/agourakis82/darwin-scaffold-studio}
}
```

## License

MIT
