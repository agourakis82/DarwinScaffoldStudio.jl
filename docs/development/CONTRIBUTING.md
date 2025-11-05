# Contributing to Darwin Core

Thank you for your interest in contributing to Darwin Core!

---

## 🎯 Quick Start

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Run tests
6. Submit a pull request

---

## 📋 Development Setup

### Prerequisites

- Python 3.9+
- Docker (optional)
- kubectl (optional, for K8s testing)

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/darwin-scaffold-studio.git
cd darwin-scaffold-studio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test
pytest tests/unit/test_graph_rag.py

# Integration tests
pytest tests/integration/

# Cluster tests (requires K8s)
pytest tests/cluster/
```

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/

# Type check
mypy app/

# All checks (pre-commit)
pre-commit run --all-files
```

---

## 📝 Code Style

### Python Style Guide

- **PEP 8** compliant
- **Line length:** 100 characters
- **Formatter:** Black
- **Linter:** Flake8
- **Type hints:** Required for public APIs

### Example

```python
"""Module docstring."""

from typing import List, Optional

def process_data(
    data: List[str],
    options: Optional[dict] = None
) -> dict:
    """
    Process data with optional configuration.
    
    Args:
        data: Input data list
        options: Optional configuration
    
    Returns:
        Processed results
    """
    results = {}
    # ... implementation ...
    return results
```

---

## 🌿 Branching Strategy

### Branch Naming

```
feature/add-new-rag-variant
bugfix/fix-graphrag-memory-leak
docs/update-cluster-setup
refactor/improve-embedding-manager
```

### Commit Messages

**Format:** Conventional Commits

```
<type>(<scope>): <description>

[optional body]

[optional footer]
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
feat(graphrag): add Leiden community detection
fix(selfrag): handle empty retrieval results
docs(cluster): update K8s setup guide
refactor(cache): improve unified cache performance
test(integration): add Multi-AI Hub tests
```

---

## 🔀 Pull Request Process

### Before Submitting

- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black`)
- [ ] Linting passes (`flake8`)
- [ ] Type checks pass (`mypy`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### PR Template

```markdown
## Description
[Describe your changes]

## Motivation
[Why is this change necessary?]

## Changes
- [ ] Added X
- [ ] Fixed Y
- [ ] Updated Z

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Tested on cluster (if applicable)

## Checklist
- [ ] Code follows style guide
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Automated checks (CI)
2. Code review (maintainer)
3. Testing (integration tests)
4. Approval
5. Merge to main

---

## 📦 Versioning

### Semantic Versioning

```
MAJOR.MINOR.PATCH

MAJOR: Breaking changes
MINOR: New features (backward compatible)
PATCH: Bug fixes (backward compatible)
```

### Examples

```
feat(api): new endpoint → MINOR bump (2.0.0 → 2.1.0)
fix(rag): memory leak → PATCH bump (2.0.0 → 2.0.1)
refactor(api): breaking change → MAJOR bump (2.0.0 → 3.0.0)
```

### Updating Version

```bash
# Update pyproject.toml
version = "2.1.0"

# Update CHANGELOG.md
## [2.1.0] - 2025-11-10
### Added
- New GraphRAG feature

# Tag and release
git tag v2.1.0
git push origin v2.1.0
```

---

## 🎯 Areas for Contribution

### High Priority

- RAG++ improvements (new variants, performance)
- Multi-AI enhancements (new models, routing)
- Documentation (examples, tutorials)
- Testing (unit, integration, cluster)

### Medium Priority

- Monitoring dashboards (Grafana)
- Performance optimizations
- Plugin examples
- Deployment guides

### Low Priority

- Code cleanup
- Typo fixes
- Minor refactoring

---

## 📞 Getting Help

- **Issues**: https://github.com/agourakis82/darwin-scaffold-studio/issues
- **Discussions**: https://github.com/agourakis82/darwin-scaffold-studio/discussions
- **Email**: agourakis@agourakis.med.br

---

## 🙏 Thank You!

Contributions are welcome and appreciated!

**"Ciência rigorosa. Resultados honestos. Impacto real."**

