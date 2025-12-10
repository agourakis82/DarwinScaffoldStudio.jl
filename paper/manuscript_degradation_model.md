# A Multi-Physics Model for Biphasic Hydrolytic Degradation of Semi-Crystalline Polymeric Scaffolds

**Authors:** [Author Names]

**Affiliations:** [Institutions]

**Corresponding Author:** [Email]

---

## Abstract

We present a comprehensive multi-physics model for predicting the hydrolytic degradation kinetics of polymeric scaffolds used in tissue engineering. The model integrates crystallinity-dependent degradation, water uptake dynamics, autocatalytic effects, and temperature dependence into a unified framework. A key innovation is the biphasic degradation model for semi-crystalline polymers (PLLA, PCL), which captures the preferential degradation of amorphous regions followed by crystalline phase breakdown. Cross-validation against six independent datasets from literature (PLDLA, PLLA, PDLLA, PLGA, PCL) demonstrates robust predictive capability with mean NRMSE of 13.2% ± 7.1% and LOOCV of 15.5% ± 7.5%. Morris sensitivity analysis identifies crystallinity (μ* = 0.681) and base degradation rate (μ* = 0.442) as the most influential parameters. The model provides a rational framework for scaffold design optimization in tissue engineering applications.

**Keywords:** Biodegradable polymers, Scaffold degradation, PLLA, Crystallinity, Multi-physics modeling, Tissue engineering

---

## 1. Introduction

### 1.1 Background

Biodegradable polymeric scaffolds are fundamental components in tissue engineering, providing temporary mechanical support and guidance for tissue regeneration [1]. The success of scaffold-based therapies depends critically on matching the degradation rate with tissue formation kinetics [2]. However, predicting degradation behavior remains challenging due to the complex interplay of multiple physical and chemical processes.

Poly(lactic acid) (PLA) derivatives—including PLLA (poly-L-lactic acid), PDLLA (poly-DL-lactic acid), and PLDLA (poly-L/DL-lactic acid)—are among the most widely used materials due to their biocompatibility, tunable degradation rates, and FDA approval for clinical applications [3]. Similarly, PLGA (poly-lactic-co-glycolic acid) and PCL (polycaprolactone) offer complementary degradation profiles for various applications.

### 1.2 Current Limitations

Existing degradation models typically fall into two categories:

1. **Empirical models**: Fit experimental data but lack predictive capability for new conditions [4]
2. **Simplified mechanistic models**: Capture single mechanisms but ignore multi-physics interactions [5]

A critical gap exists in modeling semi-crystalline polymers like PLLA, where degradation proceeds through distinct phases:
- **Phase 1**: Rapid degradation of amorphous regions
- **Phase 2**: Slower degradation of crystalline domains

This biphasic behavior, documented by Tsuji & Ikada [6] and Weir et al. [7], is not captured by standard first-order kinetic models.

### 1.3 Objectives

This work presents a unified multi-physics model that:

1. Integrates polymer-specific parameters (k₀, Eₐ, crystallinity effects)
2. Captures biphasic degradation in semi-crystalline polymers
3. Accounts for water uptake dynamics and autocatalysis
4. Validates against multiple independent datasets
5. Provides sensitivity analysis for parameter calibration guidance

---

## 2. Mathematical Model

### 2.1 Governing Equations

The molecular weight evolution is governed by:

$$\frac{dM_n}{dt} = -k_{eff}(t) \cdot M_n \cdot [1 + \alpha_{eff} \cdot \xi(t)]$$

where:
- $M_n$ is the number-average molecular weight
- $k_{eff}(t)$ is the effective degradation rate
- $\alpha_{eff}$ is the effective autocatalysis factor
- $\xi(t) = 1 - M_n(t)/M_{n0}$ is the degradation extent

### 2.2 Temperature Dependence (Arrhenius)

$$k_{temp} = k_0 \exp\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]$$

where $T_{ref}$ = 310.15 K (37°C) and $R$ = 8.314 J/(mol·K).

### 2.3 Water Uptake Dynamics

Water absorption follows a sigmoidal profile modulated by crystallinity:

$$f_{water}(t) = \left[1 - \exp\left(-\frac{0.693 \cdot t}{t_{1/2}}\right)\right] \cdot (1 - 0.4 X_c)$$

where $t_{1/2} = 7/(1 + 50w)$ is the half-saturation time and $w$ is the water uptake rate.

### 2.4 Biphasic Model for Semi-Crystalline Polymers

For polymers with initial crystallinity $X_{c,0} > 0.3$ (PLLA, PCL):

**Phase 1** (amorphous degradation): When $\phi_{am} > 0.15$:
$$k_{eff} = k_{amorphous} \cdot \phi_{am} + k_{crystalline} \cdot X_c$$

where $k_{amorphous} = 2k_{temp}$ and $k_{crystalline} = 0.15k_{temp}$

**Phase 2** (crystalline degradation): When $\phi_{am} \leq 0.15$:
$$k_{eff} = 0.4 \cdot k_{temp} \cdot (1 + \xi)$$

The apparent crystallinity increases during Phase 1 as amorphous regions degrade:
$$X_c(t) = X_{c,0} + 0.15 \cdot \min(\xi/0.5, 1)$$

### 2.5 Standard Model for Amorphous Polymers

For amorphous polymers (PDLLA, PLGA, PLDLA with low $X_c$):

$$k_{eff} = k_{temp} \cdot (1 - X_c)^{1+\gamma} \cdot f_{water} \cdot f_{Tg}$$

where $\gamma$ is the crystallinity effect exponent and $f_{Tg}$ accounts for chain mobility above $T_g$.

---

## 3. Materials and Methods

### 3.1 Polymer Parameters

| Polymer | k₀ (/day) | Eₐ (kJ/mol) | α | Xc typical | Tg (°C) |
|---------|-----------|-------------|-----|------------|---------|
| PLDLA   | 0.0175    | 80.0        | 0.066 | 0.10     | 50      |
| PLLA    | 0.0075    | 82.0        | 0.045 | 0.55     | 65      |
| PDLLA   | 0.022     | 78.0        | 0.080 | 0.00     | 45      |
| PLGA    | 0.030     | 75.0        | 0.120 | 0.00     | 48      |
| PCL     | 0.0015    | 90.0        | 0.010 | 0.50     | -60     |

### 3.2 Validation Datasets

Six independent datasets from literature were used for validation:

1. **PLDLA Kaique** (2025): Mn₀ = 51.3 kg/mol, Xc = 8%, PBS 37°C [This work]
2. **PLLA Tsuji** (2000): Mn₀ = 180 kg/mol, Xc = 55%, PBS 37°C [6]
3. **PDLLA Li** (1990): Mn₀ = 100 kg/mol, Xc = 0%, PBS 37°C [8]
4. **PLGA Grizzi** (1995): Mn₀ = 70 kg/mol, Xc = 0%, PBS 37°C [9]
5. **PCL Sun** (2006): Mn₀ = 80 kg/mol, Xc = 50%, PBS 37°C [10]
6. **PLLA Odelius** (2011): Mn₀ = 120 kg/mol, Xc = 45%, PBS 37°C [11]

### 3.3 Statistical Analysis

- **NRMSE**: Normalized root mean square error
- **LOOCV**: Leave-one-out cross-validation
- **Morris Sensitivity**: Elementary effects method with 30 trajectories

---

## 4. Results

### 4.1 Cross-Validation Performance

| Dataset | Polymer | NRMSE (%) | Status |
|---------|---------|-----------|--------|
| PLDLA Kaique | PLDLA | 11.1 | ✓ Pass |
| PLLA Tsuji | PLLA | 6.5 | ✓ Pass |
| PDLLA Li | PDLLA | 13.5 | ✓ Pass |
| PLGA Grizzi | PLGA | 24.3 | ~ Acceptable |
| PCL Sun | PCL | 18.0 | ✓ Pass |
| PLLA Odelius | PLLA | 5.6 | ✓ Pass |

**Overall Statistics:**
- Mean NRMSE: 13.2% ± 7.1%
- LOOCV: 15.5% ± 7.5%
- Pass rate: 5/6 datasets (83%)

### 4.2 Improvement over Standard Model

The biphasic model shows 34% improvement over single-phase models:

| Polymer | Standard Model | Biphasic Model | Improvement |
|---------|----------------|----------------|-------------|
| PLLA (Tsuji) | 18.9% | 6.5% | 66% |
| PLLA (Odelius) | 19.3% | 5.6% | 71% |
| PCL | 43.2% | 18.0% | 58% |

### 4.3 Morris Sensitivity Analysis

| Parameter | μ* | σ | Interpretation |
|-----------|-----|---|----------------|
| Crystallinity (Xc) | 0.681 | 0.737 | Most important, non-linear |
| Base rate (k₀) | 0.442 | 0.466 | Important, non-linear |
| Autocatalysis (α) | 0.009 | 0.009 | Minor importance |
| Initial Mn | 0.001 | 0.003 | Negligible |

---

## 5. Discussion

### 5.1 Physical Interpretation of Biphasic Model

The biphasic degradation mechanism reflects the microstructural reality of semi-crystalline polymers:

1. **Phase 1**: Water preferentially penetrates amorphous regions, initiating hydrolysis. Crystalline lamellae act as barriers, limiting diffusion.

2. **Crystallinity increase**: As amorphous chains degrade, the apparent crystallinity increases—a phenomenon documented by DSC studies [6].

3. **Phase 2**: After ~70% amorphous degradation, crystalline regions become accessible. The accumulated carboxylic acid end groups enhance autocatalysis.

### 5.2 Calibration Priorities

Morris sensitivity analysis provides clear guidance for experimental calibration:

1. **High priority**: Crystallinity (DSC measurement essential)
2. **High priority**: k₀ (requires GPC time series)
3. **Low priority**: Ea, porosity (literature values sufficient)

### 5.3 Limitations

1. **PLGA performance**: 24.3% error suggests LA:GA ratio effects need explicit modeling
2. **In vivo conditions**: Current validation limited to PBS 37°C
3. **Morphology evolution**: Pore coalescence model requires μCT validation

---

## 6. Conclusions

We developed a multi-physics model for polymeric scaffold degradation that:

1. **Captures biphasic behavior** in semi-crystalline polymers (PLLA, PCL) with 66-71% error reduction
2. **Validates across five polymer types** with 83% pass rate (NRMSE < 20%)
3. **Identifies critical parameters** through Morris sensitivity analysis
4. **Provides design guidance** for scaffold material selection

The model enables rational scaffold design by predicting degradation kinetics from measurable material properties.

---

## References

[1] Langer R, Vacanti JP. Science 1993;260:920-926.

[2] Hutmacher DW. Biomaterials 2000;21:2529-2543.

[3] Middleton JC, Tipton AJ. Biomaterials 2000;21:2335-2346.

[4] Wang Y, et al. Acta Biomater 2008;4:1244-1251.

[5] Han X, Pan J. Biomaterials 2009;30:423-430.

[6] Tsuji H, Ikada Y. Polymer 2000;41:3621-3630.

[7] Weir NA, et al. Proc Inst Mech Eng H 2004;218:307-319.

[8] Li SM, et al. J Biomed Mater Res 1990;24:595-607.

[9] Grizzi I, et al. Biomaterials 1995;16:305-311.

[10] Sun H, et al. Acta Biomater 2006;2:519-529.

[11] Odelius K, et al. Polymer 2011;52:2698-2707.

---

## Supplementary Information

### S1. Model Implementation

The model is implemented in Julia as the `UnifiedScaffoldTissueModel` module, available at:
[Repository URL]

### S2. Figure Data

All figure data files (CSV format) are provided in the supplementary materials:
- fig1_validation_data.csv
- fig2_crystallinity_effect.csv
- fig3_biphasic_model.csv
- fig4_morris_sensitivity.csv
- fig5_tissue_integration.csv
- fig6_polymer_comparison.csv
