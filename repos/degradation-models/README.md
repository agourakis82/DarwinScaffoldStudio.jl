# DegradationModels

Micro-repository for polymer degradation modeling, focusing on PLDLA (Poly-L/DL-lactic acid).

## Overview

This package provides multiple models for predicting polymer molecular weight degradation over time:

| Model | Accuracy | Interpretability | Use Case |
|-------|----------|------------------|----------|
| **NeuralModel** | 95.5% | Low | Production, highest accuracy needed |
| **HybridPINN** | ~85% | Medium | Research, need physics insights |
| **ConservativeModel** | ~82% | High | Quick estimates, understanding |
| **BronstedModel** | ~75% | High | Teaching, mechanism exploration |
| **ThermodynamicModel** | ~50% | Very High | First principles validation |

## Quick Start

```julia
using DegradationModels

# Train the neural model (best accuracy)
model = train_neural(epochs=3000)

# Predict Mn at 30 days
Mn = predict(model, "Kaique_PLDLA", 51.3, 30.0)
# → 25.7 kg/mol (experimental: 25.4)

# Validate against all datasets
results = validate(model)
# → Dict("Kaique_PLDLA" => 2.2%, ...)
```

## Models

### 1. NeuralModel (Recommended)

Neural network with material embeddings. Best accuracy for predictions.

```julia
model = train(NeuralModel, epochs=3000)
Mn = predict(model, "Kaique_PLDLA", 51.3, 30.0)
```

**Architecture:**
- Input: 6 features + 8 material embedding
- Hidden: 64 neurons × 2 (GELU + residual)
- Output: fraction remaining

### 2. BronstedModel

Based on Brønsted-Lowry acid catalysis theory.

```julia
model = train(BronstedModel)
Mn = predict(model, "Kaique_PLDLA", 51.3, 30.0)
```

**Chemistry:**
- COOH dissociation → local pH drop
- Vogel-Fulcher-Tammann temperature dependence
- Autocatalytic acceleration

### 3. ThermodynamicModel

First-principles approach using Eyring theory.

```julia
model = train(ThermodynamicModel)
```

**Theory:**
- k = (kB·T/h)·exp(-ΔG‡/RT)
- ΔG‡ = 103 kJ/mol at 37°C
- Fick's law for diffusion

### 4. ConservativeModel

Empirical model with saturating autocatalysis.

```julia
model = train(ConservativeModel)
```

**Features:**
- Material-specific parameters
- Saturating autocatalysis: tanh(α·extent)
- Crystallinity protection

### 5. HybridPINN

Physics-Informed Neural Network combining physics encoder with neural correction.

```julia
model = train(HybridPINN, epochs=1000)
```

## Experimental Data

Built-in datasets from PhD thesis (Kaique) and literature:

```julia
# Available datasets
keys(EXPERIMENTAL_DATA)
# → ["Kaique_PLDLA", "Kaique_TEC1", "Kaique_TEC2", "InVivo_Subcutaneous"]

# Access data
data = EXPERIMENTAL_DATA["Kaique_PLDLA"]
# → (Mn0=51.3, times=[0,30,60,90], Mn=[51.3,25.4,18.3,7.9], ...)
```

## API Reference

### Training

```julia
train(ModelType; kwargs...)
train_neural(; epochs=3000)
train_bronsted()
train_thermodynamic()
```

### Prediction

```julia
predict(model, material, Mn0, t; T=310.15, pH=7.4, TEC=0.0)
```

**Arguments:**
- `material`: Material name or ID (1-4)
- `Mn0`: Initial molecular weight (kg/mol)
- `t`: Time (days)
- `T`: Temperature (K), default 310.15 (37°C)
- `pH`: pH of medium, default 7.4
- `TEC`: Plasticizer content (%), default 0.0

### Validation

```julia
validate(model)  # Returns Dict with MAPE per dataset
compare_models([model1, model2, ...])  # Side-by-side comparison
```

## Physical Parameters

The thermodynamic model uses these first-principles constants:

| Parameter | Value | Description |
|-----------|-------|-------------|
| ΔH‡ | 78 kJ/mol | Activation enthalpy |
| ΔS‡ | -80 J/(mol·K) | Activation entropy |
| ΔG‡(37°C) | 103 kJ/mol | Free energy barrier |
| pKa(lactic) | 3.86 | Lactic acid dissociation |
| Tg,∞ | 57°C | Glass transition (high Mn) |
| Mc | 9 kg/mol | Entanglement threshold |

## Citation

If you use this package, please cite:

```bibtex
@software{degradation_models,
  title = {DegradationModels.jl: Physics-Informed Neural Networks for PLDLA Degradation},
  author = {Darwin Scaffold Studio},
  year = {2025},
  url = {https://github.com/darwin-scaffold-studio/degradation-models}
}
```

## License

MIT License - See LICENSE file for details.
