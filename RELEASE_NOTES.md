## Darwin Scaffold Studio v0.2.1

### Major Features

#### OBO Foundry Ontology Integration
- 12 ontology libraries with **1200+ biomedical terms**
- Standardized IDs: UBERON (anatomy), CL (cells), CHEBI (chemicals), GO (processes), DOID (diseases)
- 3-tier lookup system: hardcoded -> SQLite cache -> EBI OLS API
- FAIR JSON-LD export with Schema.org vocabulary

#### Scientific Validation
- 16 TPMS synthetic scaffolds (Gyroid, Diamond, Schwarz P, Neovius)
- Analytical ground truth at 50%, 70%, 85%, 90% porosity
- **Validation benchmark: less than 1% error** on porosity, surface area, pore size

#### Reproducibility
- Comprehensive test suite (11 test files, 100+ assertions)
- Docker multi-stage build for reproducible environment
- GitHub Actions CI/CD pipeline
- Quick tests for fast CI (~0.3s)

### Documentation
- docs/tutorial.md - Complete end-to-end workflow
- docs/api.md - Full API reference

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/agourakis82/darwin-scaffold-studio")
```

Or with Docker:
```bash
docker build -t darwin-scaffold-studio .
docker run -it darwin-scaffold-studio
```

### Requirements
- Julia 1.10+
- See Project.toml for dependencies

---
*Tissue Engineering Scaffold Analysis Platform*
