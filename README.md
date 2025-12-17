# DarwinScaffoldStudio.jl

Julia package for analyzing tissue engineering scaffolds from MicroCT and SEM imaging data.

[![CI](https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/agourakis82/darwin-scaffold-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Desktop App

Premium desktop application with 3D visualization, AI-powered analysis, and scientific panels.

![Darwin Scaffold Studio Desktop App](docs/screenshot.png)

| Platform | Download |
|----------|----------|
| Linux (AppImage) | [darwin-scaffold-studio_1.3.0_amd64.AppImage](https://github.com/agourakis82/DarwinScaffoldStudio.jl/releases/download/desktop-v1.3.0/darwin-scaffold-studio_1.3.0_amd64.AppImage) |
| Linux (Debian/Ubuntu) | [darwin-scaffold-studio_1.3.0_amd64.deb](https://github.com/agourakis82/DarwinScaffoldStudio.jl/releases/download/desktop-v1.3.0/darwin-scaffold-studio_1.3.0_amd64.deb) |
| Windows (Installer) | [Darwin.Scaffold.Studio_1.3.0_x64-setup.exe](https://github.com/agourakis82/DarwinScaffoldStudio.jl/releases/download/desktop-v1.3.0/Darwin.Scaffold.Studio_1.3.0_x64-setup.exe) |
| Windows (MSI) | [Darwin.Scaffold.Studio_1.3.0_x64_en-US.msi](https://github.com/agourakis82/DarwinScaffoldStudio.jl/releases/download/desktop-v1.3.0/Darwin.Scaffold.Studio_1.3.0_x64_en-US.msi) |
| macOS (Apple Silicon) | [Darwin.Scaffold.Studio_1.3.0_aarch64.dmg](https://github.com/agourakis82/DarwinScaffoldStudio.jl/releases/download/desktop-v1.3.0/Darwin.Scaffold.Studio_1.3.0_aarch64.dmg) |

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
