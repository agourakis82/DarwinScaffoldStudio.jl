# Darwin Scaffold Studio - Development Context

## Project Overview
Julia-based platform for tissue engineering scaffold analysis with AI agents.

## Quick Commands
```bash
# Test modules load
julia --project=. test/test_minimal.jl

# Build Demetrios compiler (submodule)
git submodule update --init --recursive
./scripts/build_demetrios.sh

# Start Julia REPL with project
julia --project=.

# Load the module
include("src/DarwinScaffoldStudio.jl")
using .DarwinScaffoldStudio
```

## Architecture

### Module Structure
```
src/DarwinScaffoldStudio/
  Core/           # Config, Types, Utils, ErrorHandling
  MicroCT/        # Image loading, preprocessing, segmentation, metrics
  Optimization/   # Parametric design, scaffold optimization
  Visualization/  # Mesh3D, Export, NeRF, GaussianSplatting
  Science/        # Topology, Percolation, ML, PINNs, TDA, GNN
  Agents/         # Core, DesignAgent, AnalysisAgent, SynthesisAgent
  LLM/            # OllamaClient
  Advanced/       # Quantum, Blockchain, DigitalTwin, etc.
  Theory/         # CategoryTheory, InformationTheory, CausalInference
  Foundation/     # ESM-3, Diffusion, NeuralOperators
  Hausen/         # BioactiveGlass, Antimicrobial, Phytochemical
  Pipeline/       # End-to-end workflow
  Simulation/     # Tissue growth
  Vision/         # SEM cell identification
```

### Module Loading Order
1. Core modules (always loaded)
2. MicroCT, Optimization, Visualization (always loaded)
3. Science basics: Topology, Percolation, ML, Optimization
4. LLM + Agents
5. FRONTIER AI (optional): PINNs, TDA, GNN - controlled by `enable_frontier_ai`
6. Advanced modules (optional): controlled by `enable_advanced_modules`

### Key Types
- `ScaffoldMetrics`: porosity, pore_size, interconnectivity, tortuosity
- `ScaffoldParameters`: target values for optimization
- `OptimizationResults`: optimized volume + metrics + improvement
- `GlobalConfig`: system-wide settings

## Common Development Tasks

### Adding a New Module
1. Create file in appropriate directory: `src/DarwinScaffoldStudio/Category/NewModule.jl`
2. Use template:
```julia
module NewModule
using ..Types
using ..Config
export main_function

function main_function(args)
    # implementation
end

end # module
```
3. Add include to `src/DarwinScaffoldStudio.jl`
4. Add exports if needed
5. Run `julia --project=. test_minimal.jl` to verify

### Common Import Patterns
```julia
# From sibling module (same directory level)
using ..OtherModule

# Import specific items
using ..Types: ScaffoldMetrics, ScaffoldParameters

# From parent's sibling
using ..Config: get_config
```

### Fixing Import Errors
- `UndefVarError: Module not defined` -> Check include order in main file
- `...Module` -> Usually should be `..Module` (2 dots, not 3)
- `optimizer::ScaffoldOptimizer` -> Should be `optimizer::Optimizer` (type, not module)

## Code Style
- Functions: `snake_case`
- Types/Modules: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Always add docstrings to exported functions
- Use type annotations for public API

## Git Commits
- Do NOT include LLM co-author lines in commits
- Do NOT include "Generated with Claude Code" or similar footers
- Keep commit messages clean and human-authored

## Demetrios Compiler Bridge
- Set `DEMETRIOS_HOME` to the Demetrios repo root or `DEMETRIOS_STDLIB` to its `stdlib/` to help `CompilerBridge` locate `dc` and stdlib.

## Testing
```bash
# Quick structure test
julia --project=. test/test_minimal.jl

# Full test suite (slow, loads all deps)
julia --project=. test/runtests.jl

# Test specific module loading
julia --project=. -e 'include("src/DarwinScaffoldStudio/Core/Types.jl")'
```

## Literature References
- Murphy et al. 2010: Pore size 100-200um optimal for bone
- Karageorgiou 2005: Porosity 90-95%, interconnectivity >= 90%
- Gibson-Ashby: E_scaffold = (1-porosity)^2 * E_solid

## Slash Commands Available
- `/dev` - Development tasks
- `/new-module` - Create new module
- `/fix-imports` - Fix import issues
- `/add-feature` - Implement new feature
- `/debug` - Debug issues
- `/refactor` - Refactor code
