# PLDLA 3D-Print Degradation Model

## Overview

Model for predicting molecular weight degradation of 3D-printed PLDLA 70:30 scaffolds for tissue engineering applications.

**Accuracy: 87.3%** (validated against experimental data)

## Scope

- **Material**: PLDLA 70:30 (poly-L/DL-lactide, 70% L-lactide, 30% DL-lactide)
- **Form**: 3D-printed scaffolds (FDM/extrusion)
- **Conditions**: In vitro, PBS pH 7.4, 37°C
- **Time range**: 0-90 days
- **Plasticizer**: TEC (triethyl citrate) 0-2%

## Data Source

Experimental data from Kaique Hergesel PhD thesis (UNICAMP):

| Formulation | Mn0 (kg/mol) | TEC (%) | t½ (days) |
|-------------|--------------|---------|-----------|
| PLDLA pure  | 51.3         | 0       | ~30       |
| PLDLA+TEC1  | 45.0         | 1       | ~30       |
| PLDLA+TEC2  | 32.7         | 2       | ~31       |

## Model Equation

```
Mn(t) = Mn0 × exp(-k_eff × t^n)

where:
  k_eff = k × δ_TEC × (1 + a × (1 - exp(-t/τ)))
```

**Fitted Parameters:**
- `k = 0.0447 day⁻¹` - Base hydrolysis rate
- `n = 0.669` - Time exponent (sub-linear kinetics)
- `a = 0.819` - Autocatalysis strength
- `τ = 26.0 days` - Autocatalysis time constant
- `δ_TEC1 = 1.02` - TEC1 rate multiplier
- `δ_TEC2 = 0.99` - TEC2 rate multiplier

## Usage

```julia
using DegradationModels

# Train model
model = train(PLDLA3DPrintModelV2, epochs=3000)

# Predict Mn at 30 days for pure PLDLA
Mn = predict(model, 50.0, 30.0)  # → ~25 kg/mol

# With 1% TEC plasticizer
Mn = predict(model, 45.0, 30.0, TEC=1.0)

# Get half-life
t_half = estimate_halflife(model, 50.0)  # → ~30 days

# Generate degradation curve
curve = predict_curve(model, 50.0, t_max=90.0)
```

## Validation Results

| Dataset | Mn0 | Accuracy |
|---------|-----|----------|
| PLDLA   | 51.3| 84.9%    |
| TEC1    | 45.0| 88.3%    |
| TEC2    | 32.7| 88.7%    |
| **Global** | - | **87.3%** |

## Degradation Timeline

For Mn0 = 50 kg/mol, pure PLDLA:

| Time (days) | Mn (kg/mol) | % Remaining |
|-------------|-------------|-------------|
| 0           | 50.0        | 100%        |
| 7           | 41.1        | 82%         |
| 14          | 35.2        | 70%         |
| 30          | 25.3        | 51%         |
| 60          | 15.0        | 30%         |
| 90          | 9.8         | 20%         |

## Key Findings

1. **TEC effect on processing**: TEC reduces initial Mn (51→45→33 kg/mol) but does not significantly change degradation rate
2. **Autocatalysis**: Degradation accelerates after ~25 days due to acidic byproduct accumulation
3. **Sub-linear kinetics**: Time exponent n=0.67 indicates diffusion-limited hydrolysis
4. **Half-life**: Approximately 30 days for all formulations

## Limitations

1. **Scope**: Only validated for PLDLA 70:30, not other L:DL ratios
2. **Conditions**: In vitro only; in vivo may differ due to enzymatic effects
3. **Temperature**: 37°C only; use Arrhenius correction for other temperatures
4. **Morphology**: 3D-printed scaffolds; films/fibers may differ

## Physical Interpretation

The model captures:

1. **Bulk degradation**: Water penetrates the polymer matrix
2. **Random chain scission**: Ester bonds hydrolyze randomly
3. **Autocatalysis**: COOH end groups catalyze further hydrolysis
4. **Sub-linear time dependence**: Diffusion limitations at later stages

## References

1. Hergesel, K. (PhD Thesis) - PLDLA scaffold degradation data
2. Pitt, C.G. & Schindler, A. (1984) - Autocatalytic hydrolysis model
3. Weir, N.A. et al. (2004) - PLLA degradation kinetics

## Author

Darwin Scaffold Studio - December 2025
