# Contributing to Darwin Scaffold Studio

Thank you for your interest in contributing to Darwin Scaffold Studio!

## Quick Start

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Run tests
6. Submit a pull request

## Development Setup

### Prerequisites

- Julia 1.10+
- Git

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/darwin-scaffold-studio.git
cd darwin-scaffold-studio

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run quick tests to verify setup
julia --project=. test/test_quick.jl
```

## Project Structure

```
darwin-scaffold-studio/
├── src/
│   └── DarwinScaffoldStudio/
│       ├── Core/           # Types, Config, Utils
│       ├── MicroCT/         # Image processing
│       ├── Optimization/    # Scaffold optimization
│       ├── Visualization/   # Mesh and export
│       ├── Science/         # Topology, ML
│       └── Ontology/        # OBO Foundry integration
├── test/                    # Test files
├── scripts/                 # Utility scripts
├── data/                    # Validation data
└── docs/                    # Documentation
```

## Testing

### Run Tests

```bash
# Quick tests (CI, ~0.3s)
julia --project=. test/test_quick.jl

# Full test suite
julia --project=. test/runtests.jl

# Specific test file
julia --project=. -e 'include("test/test_microct.jl")'

# Validation benchmark
julia --project=. scripts/run_validation_benchmark.jl
```

## Code Style

### Julia Style Guide

- **Functions**: `snake_case`
- **Types/Modules**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Line length**: 92 characters (soft limit)
- **Indentation**: 4 spaces

### Docstrings

All exported functions must have docstrings:

```julia
"""
    compute_porosity(scaffold::Array{Bool,3}) -> Float64

Compute the porosity of a binary scaffold volume.

# Arguments
- `scaffold`: Binary 3D array where `true` = material, `false` = pore

# Returns
- Porosity value between 0.0 and 1.0

# Example
```julia
scaffold = rand(Bool, 100, 100, 100)
porosity = compute_porosity(scaffold)
```
"""
function compute_porosity(scaffold::Array{Bool,3})
    return 1.0 - sum(scaffold) / length(scaffold)
end
```

### Type Annotations

- Required for public API functions
- Optional for internal functions
- Use abstract types when possible (`AbstractArray` vs `Array`)

## Branching Strategy

### Branch Naming

```
feature/add-sem-analysis
bugfix/fix-segmentation-threshold
docs/update-api-reference
refactor/improve-mesh-generation
```

### Commit Messages

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(ontology): add disease library with DOID terms
fix(metrics): correct surface area calculation for non-cubic voxels
docs(api): add examples for optimization module
test(tpms): add Neovius surface validation tests
```

## Pull Request Process

### Before Submitting

- [ ] Tests pass (`julia --project=. test/test_quick.jl`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)

### PR Template

```markdown
## Description
[Describe your changes]

## Motivation
[Why is this change necessary?]

## Changes
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Quick tests pass
- [ ] Added tests for new functionality
- [ ] Tested with real data (if applicable)

## Checklist
- [ ] Code follows style guide
- [ ] Docstrings added for new functions
- [ ] No breaking changes (or documented if necessary)
```

## Adding New Modules

### 1. Create the module file

```julia
# src/DarwinScaffoldStudio/Category/NewModule.jl
"""
    NewModule

Description of the module.
"""
module NewModule

using ..Types
using ..Config

export main_function

"""
    main_function(args)

Description of the function.
"""
function main_function(args)
    # implementation
end

end # module
```

### 2. Add to main file

Edit `src/DarwinScaffoldStudio.jl`:

```julia
include("DarwinScaffoldStudio/Category/NewModule.jl")
using .NewModule
export main_function
```

### 3. Add tests

Create `test/test_newmodule.jl` and add to `test/runtests.jl`.

## Areas for Contribution

### High Priority

- SEM image analysis improvements
- Additional TPMS surface types
- Performance optimization for large datasets
- Documentation examples

### Medium Priority

- New ontology libraries (more OBO terms)
- Visualization enhancements
- Export format additions (OBJ, PLY with colors)

### Good First Issues

- Improve error messages
- Add docstrings to undocumented functions
- Fix typos in documentation
- Add test coverage

## Getting Help

- **Issues**: https://github.com/agourakis82/darwin-scaffold-studio/issues
- **Email**: demetrios@agourakis.med.br

## Code of Conduct

Please read and follow our [Code of Conduct](../../CODE_OF_CONDUCT.md).

---

Thank you for contributing to Darwin Scaffold Studio!
