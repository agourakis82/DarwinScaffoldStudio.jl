# Repository Guidelines

## Project Structure & Module Organization
- `src/DarwinScaffoldStudio/` holds Julia code (Core, MicroCT, Science, Agents, Optimization, Visualization, Ontology).
- `test/` contains tests (`test_*.jl`), with `test/runtests.jl` as entrypoint.
- `desktop/` hosts the SvelteKit + Tauri app; `darwin-server/` hosts the Rust server.
- `examples/`, `docs/`, `data/`, and `scripts/` store docs, data, and utilities.

## Build, Test, and Development Commands
- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs Julia dependencies.
- `julia --project=. test/test_minimal.jl` runs module-load checks.
- `julia --project=. test/runtests.jl` runs full test suite; use `-e 'include("test/test_microct.jl")'` for one file.
- `cd darwin-server && cargo build --release` builds server; `cargo run --release` runs it.
- `cd desktop && npm install && npm run dev` starts UI; `npm run tauri:dev` runs the desktop app.
- `git submodule update --init --recursive` fetches `demetrios/` before `./scripts/build_demetrios.sh` (Demetrios compiler setup).
- `./scripts/setup_llm.sh` optionally configures local LLM support.

## Architecture & Module Loading Order
- Core loads first, followed by MicroCT/Optimization/Visualization, then Science basics, then LLM + Agents.
- Optional FRONTIER AI modules (PINNs, TDA, GNN) use `enable_frontier_ai`; Advanced modules use `enable_advanced_modules`.

## Coding Style & Naming Conventions
- Julia: 4-space indentation, 92-character soft line limit, `snake_case` functions, `PascalCase` types/modules, `SCREAMING_SNAKE_CASE` constants.
- Add docstrings for exported functions; use type annotations on public APIs; prefer abstract types (`AbstractArray`).
- Frontend: keep Svelte/TS changes consistent with patterns; use `npm run check` in `desktop/` for `svelte-check`.

## Import & Module Tips
- Sibling import: `using ..OtherModule`; selective: `using ..Types: ScaffoldMetrics`.
- Import errors: check include order in `src/DarwinScaffoldStudio.jl`; use `..Module` (two dots) for siblings.

## Testing Guidelines
- Tests live in `test/` and use Julia's `Test` stdlib.
- Name test files `test_<area>.jl` and register new files in `test/runtests.jl`.
- Run quick tests before PRs; use `include(...)` runs while iterating.

## Commit & Pull Request Guidelines
- Use conventional commits: `type(scope): description` (scope optional). Examples: `feat(ontology): add disease library`, `fix(metrics): correct voxel scaling`.
- Recent history uses `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `release`.
- PRs should include description/motivation, testing notes, and docs/tests updates; update `CHANGELOG.md` for major changes.

## Agent-Specific Notes
- Do not add LLM co-author lines or "Generated with ..." footers in commits.
- When adding a new module, include it in `src/DarwinScaffoldStudio.jl` and add matching tests.
- Set `DEMETRIOS_HOME` (repo root) or `DEMETRIOS_STDLIB` so the compiler bridge finds `dc` and stdlib.

## Available Slash Commands
- `/dev`, `/new-module`, `/fix-imports`, `/add-feature`, `/debug`, `/refactor`.
