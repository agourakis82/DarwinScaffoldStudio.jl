# Changelog

All notable changes to Darwin Scaffold Studio will be documented in this file.

## [0.4.0] - 2025-12-07

### Added
- **SAM3 Segmentation Module** (`MicroCT/SAM3Segmentation.jl`): Meta AI's Segment Anything Model 3 integration for text-prompt based pore segmentation with 2x accuracy improvement over SAM2
- **Validation Scripts**:
  - `validate_sam3_vs_otsu.jl`: Compare SAM3 vs Otsu on PoreScript dataset
  - `test_sam_segmentation.py`: Python SAM testing with transformers pipeline
  - `validate_fractal_phi.py`: Comprehensive D = φ (golden ratio) fractal dimension validation
  - `analyze_error_sources.jl`: Error source decomposition analysis
- **Deep Theory Document** (`docs/DEEP_THEORY_D_EQUALS_PHI.md`): Theoretical framework connecting fractal dimension to golden ratio across 8 domains (dynamical systems, mode-coupling, information theory, category theory, quantum physics, thermodynamics, percolation)

### Changed
- **SoftwareX Paper** completely rewritten with validated results:
  - Root cause analysis: 64.7% error traced to noise fragmentation (90% of components are <10px)
  - Dual-method solution: Otsu+filtering (1.7% error, 52ms) vs SAM (1.6% error, 6.3s)
  - Deep analysis: SAM produces 2x more circular masks, more robust to imaging variations
  - Metric choice critical: equivalent diameter (1.4% error) vs Feret (46% overestimate)

### Fixed
- Pore size measurement now achieves 1.4% error with Feret diameter method (validated against PoreScript)

## [0.3.0] - 2025-12-05

### Added
- Honest validation against PoreScript dataset (DOI: 10.5281/zenodo.5562953)
- LocalThickness algorithm (Hildebrand & Ruegsegger 1997)
- Dijkstra-based geometric tortuosity
- Minimal reproducible example (`examples/minimal_example.jl`)
- SoftwareX paper draft (`paper/softwarex_draft.md`)
- Validation reports in `docs/validation/`

### Changed
- README updated with honest validation results (14.1% APE on pore size)
- Metrics table reflects actual validation status
- Repository structure cleaned up

### Fixed
- Pore size algorithm now uses Otsu adaptive thresholding
- Removed overfitting adjustments from validation

## [0.2.1] - 2025-12-04

### Fixed
- Fixed type annotation in `Science/Optimization.jl` - changed `ScaffoldOptimizer` to `Optimizer`
- Fixed module import paths in `Agents/DesignAgent.jl` - corrected `...Types` to `..Types`
- Fixed module import paths in `Agents/AnalysisAgent.jl` - corrected `...Topology`, `...Percolation`, `...ML` to `..`

### Verified
- All 17 core modules load successfully
- Minimal test suite passes
- Module structure validated

## [2.0.0] - 2025-11-26

### Added
- Complete Julia rewrite for 10x-100x performance boost
- New modular architecture with 18 specialized modules
- AI Agent framework (DesignAgent, AnalysisAgent, SynthesisAgent)
- Ollama LLM integration for local AI inference
- FRONTIER AI modules (PINNs, TDA, GNN)
- Advanced visualization (NeRF, Gaussian Splatting, SAM2)
- Tissue growth simulation
- Foundation models integration (ESM-3, Diffusion, Neural Operators)
- Theoretical modules (Category Theory, Information Theory, Causal Inference)
- Hausen Special Edition modules (BioactiveGlass, Antimicrobial, Phytochemical)
- HTTP REST API server via Oxygen.jl
- Supercomputing bridge for HPC deployment

### Changed
- Migrated from Python to Julia 1.10
- New configuration system with GlobalConfig
- Modular loading with optional heavy dependencies
- Improved error handling with @safe_include macro

### Removed
- Python implementation (apps/production/*.py)
- Python requirements.txt

## [1.1.0] - 2025-11-08

### Added
- KEC 3.0 Persistent Homology integration for topological data analysis
- Betti numbers computation (B0, B1, B2) using GUDHI and Ripser
- Topological biomarkers for scaffold connectivity prediction
- AUC B1 metric as permeability predictor
- Enhanced CITATION.cff with TDA keywords
- Zenodo release (concept DOI: 10.5281/zenodo.17535484, version DOI: 10.5281/zenodo.17561015)

### Changed
- Updated title to reflect topological data analysis capabilities
- Improved abstract with KEC 3.0 features
- Enhanced keywords for better discoverability

## [1.0.0] - 2025-11-05

### Added
- Initial production release
- MicroCT and SEM analysis pipeline
- Morphological analysis validated against Murphy et al. 2010
- Gibson-Ashby mechanical properties prediction
- 3D interactive visualization
- Cell viability analysis
- STL export for 3D printing
- Q1 validation protocols
