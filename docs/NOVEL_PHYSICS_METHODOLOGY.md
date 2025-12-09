# Methodology: Systematic Search for Novel Physics in Porous Media

## Overview

This document explains the rigorous methodology used in `scripts/search_novel_physics.jl` to search for genuinely novel discoveries that could merit Nature/Science-level publication.

---

## The Standard for "Novel"

### What Counts as Novel?
1. **Universal scaling law** not previously reported
2. **Unexpected correlation** between seemingly unrelated properties
3. **Theoretical prediction** validated for the first time
4. **Fundamental bound** on physical properties
5. **Hidden symmetry** or organizing principle

### What Does NOT Count?
- Confirming known percolation theory
- Incremental improvements to existing models
- Correlations explainable by trivial coupling (e.g., both depend on porosity)
- Results requiring excessive hand-waving to explain

---

## Hypothesis 1: Percolation Threshold Scaling

### Theory
Near the percolation threshold p_c ≈ 0.3116 (3D), tortuosity should diverge:

```
τ ~ |p - p_c|^(-μ)
```

The exponent μ is predicted by percolation theory to be **universal** (independent of microscopic details).

### Literature Value
μ ≈ 1.3 for 3D percolation (from scaling relations)

### Test
1. Generate random percolation structures at p = 0.32, 0.33, ..., 0.75
2. Compute tortuosity via geodesic BFS through pore space
3. Fit power law: log(τ) vs log(|p - p_c|)
4. Compare fitted μ to literature value

### Novelty Assessment
- **If μ matches literature (within 20%)**: Validates implementation, NOT novel
- **If μ significantly differs**: Either error in code or potential discovery
  - Requires extensive validation on known systems first

### Result Classification
- ✓ **CONFIRMED KNOWN**: Validates percolation theory
- ✗ **NO DISCOVERY**: Standard behavior observed

---

## Hypothesis 2: Topology-Transport Universality

### Theory
Persistent homology characterizes pore space topology via Betti numbers:
- β₀ = number of connected components
- β₁ = number of tunnels (1-cycles)
- β₂ = number of voids (2-cycles)

Euler characteristic: χ = β₀ - β₁ + β₂

### Hypothesis
Transport properties (tortuosity τ, permeability κ) may follow a universal function of χ:

```
τ = f(χ, p)  or  κ = g(χ, p)
```

If true, this would be a topological transport law analogous to topological insulators in condensed matter.

### Test
1. Compute Betti numbers using cubical homology
2. Compute transport properties
3. Test correlations: cor(χ, τ), cor(χ, κ)
4. Partial correlation controlling for porosity p

### Critical Limitation
**Proper Betti number computation requires computational topology libraries**:
- Eirene.jl (Julia)
- GUDHI (Python)
- Dionysus (Python/C++)

Current implementation uses placeholder β₁ = β₂ = 0 (only counts components β₀).

### Novelty Assessment
- **Strong correlation (|cor| > 0.7)** after controlling for porosity → POTENTIAL DISCOVERY
- Requires rigorous homology implementation to validate

### Literature Search Required
- Topological data analysis in porous media
- Betti number correlations with permeability
- Prior work by Herring et al., Vogel et al. on pore network topology

---

## Hypothesis 3: Information-Theoretic Bounds

### Theory
Shannon's channel capacity theorem sets fundamental limits on information transmission. Analogously, there may be a **maximum transport efficiency** for given structural entropy.

Shannon entropy of pore size distribution:
```
H(X) = -Σ p(x) log₂ p(x)
```

### Hypothesis
1. **Maximum entropy → maximum transport** (at fixed porosity)
2. **Fundamental bound**: κ ≤ κ_max(p, H)
3. **Rate-distortion trade-off**: Structural complexity vs performance

### Test
1. Compute pore size distribution entropy H
2. Compute transport efficiency (1/τ) and permeability κ
3. Plot κ vs H at fixed p
4. Look for upper envelope or plateau

### Novelty Assessment
- **Clear fundamental bound**: Would be major discovery
- **Strong H-κ correlation**: Interesting but likely confounded by porosity
- **No pattern**: Information theory may not apply to this system

### Known Literature
- Kozeny-Carman relates permeability to porosity and tortuosity
- No known information-theoretic bounds on porous media transport
- **This could be genuinely novel if found**

---

## Hypothesis 4: Fractal Dimension D = φ Universality

### Background
Our previous work validated D = φ (golden ratio) for **salt-leached scaffolds** specifically.

### Hypothesis
Does D = φ emerge **universally** in:
1. Random percolation structures?
2. Natural porous media (soil, bone)?
3. Other stochastic fabrication methods?

### Test
1. Compute 3D fractal dimension via box counting on pore boundaries
2. Test on random percolation at various porosities
3. Statistical test: Is D significantly equal to φ?

### Expected Result
- **Random percolation**: D ≈ 2.5 (known from literature)
- **Salt-leached**: D ≈ 1.62 ≈ φ (our validated result)
- **Natural bone**: Unknown - POTENTIAL DISCOVERY if D ≈ φ

### Novelty Assessment
- **D ≈ φ in random percolation**: Would contradict literature (likely implementation error)
- **D ≈ 2.5 in random percolation**: Confirms known result
- **D ≈ φ ONLY in salt-leached**: Confirms our existing finding (fabrication-specific)

### Conclusion
The D = φ result is **specific to salt-leaching**, not universal. This is actually GOOD - it makes it more interesting and explainable via packing statistics.

---

## Hypothesis 5: Graph Spectral Properties

### Theory
Pore network as graph: nodes = pore voxels, edges = connectivity

Graph Laplacian: L = D - A
- D = degree matrix
- A = adjacency matrix

Eigenvalues λ of L encode network structure:
- λ₁ = 0 (always)
- λ₂ = Fiedler value (connectivity measure)
- Spectral gap = λ₂ - λ₁ = λ₂

### Hypothesis
**Spectral gap predicts transport properties**:
- Larger gap → better connectivity → lower tortuosity
- Graph resistance ∝ effective transport resistance

### Test
1. Build pore network graph (sampled for computational tractability)
2. Compute Laplacian eigenvalues
3. Test: cor(λ₂, τ) < 0? (expect negative correlation)

### Known Literature
- Graph theory extensively used in pore network modeling
- Electrical resistance network analogy (Kirchhoff's laws)
- **Check if spectral properties already used in petroleum engineering**

### Novelty Assessment
- **Strong correlation (|cor| > 0.7)**: May be rediscovery
  - Requires literature search: "spectral graph theory porous media"
- **Novel if**: No prior use of Laplacian eigenvalues as transport predictor
- **Practical value**: Fast screening tool without full CFD simulation

### Computational Challenge
Graph construction is O(n²) for n nodes. For 50³ = 125k voxels, must sample ~1000 nodes.

---

## Statistical Rigor

### Sample Size
- Minimum 10 samples per condition
- Statistical power analysis: For cor test with r=0.7, need n≥15 for 80% power at α=0.05

### Multiple Comparisons
Testing 5 hypotheses → Bonferroni correction: α = 0.05/5 = 0.01

### Effect Size
- |r| > 0.7: Strong correlation (potentially interesting)
- |r| = 0.4-0.7: Moderate (investigate confounds)
- |r| < 0.4: Weak (probably not useful)

### Reproducibility
- Random seed: 42 (fixed)
- Report all parameters
- Code openly available

---

## Critical Assessment Framework

For each "potential discovery":

### 1. Sanity Checks
- [ ] Does it violate known physics?
- [ ] Is it numerically stable?
- [ ] Does it replicate across runs?

### 2. Confound Analysis
- [ ] Could it be explained by trivial coupling?
- [ ] Does partial correlation remain strong?
- [ ] Is it just measuring porosity indirectly?

### 3. Literature Search
- [ ] Google Scholar: "topology porous media permeability"
- [ ] Check review papers in petroleum engineering
- [ ] ArXiv: Recent preprints in porous media physics

### 4. Physical Intuition
- [ ] Can we explain WHY it should be true?
- [ ] Does it make testable predictions?
- [ ] What are the boundary conditions?

### 5. Validation
- [ ] Test on real data (Zenodo soil tomography)
- [ ] Compare across length scales
- [ ] Check consistency with CFD simulations

---

## Null Results Are Valuable

**If no novel physics is found, report it honestly.**

This is valuable because:
1. Confirms existing theory works
2. Validates implementation
3. Guides future research priorities
4. Demonstrates scientific integrity

### Publication Strategy for Null Results
- PLOS ONE accepts rigorously conducted negative results
- "We systematically tested 5 hypotheses for novel scaling laws in porous media..."
- "Our results confirm existing percolation theory..."

---

## Recommended Next Steps

### If Potential Discovery Found

1. **Increase sample size** to n ≥ 50 per condition
2. **Test on real data**:
   - Zenodo soil tomography datasets
   - Natural bone μCT scans
   - Published benchmark datasets
3. **Literature deep dive**:
   - Hire expert to review (if available)
   - Check petroleum engineering journals
   - Contact domain experts
4. **Mechanistic model**:
   - Why does this relationship exist?
   - Derive from first principles if possible
5. **Independent validation**:
   - Share code with collaborators
   - Request external replication

### If No Discovery

1. **Focus on D = φ result** (already validated, genuinely interesting)
2. **Write rigorous thesis** on salt-leaching physics
3. **Connect to theory**:
   - Fibonacci universality class (Phys. Rev. E 2024)
   - Renormalization group fixed points
   - Thermodynamic non-equilibrium steady states
4. **Practical impact**:
   - Design rules for tissue engineering
   - Optimal scaffold fabrication parameters

---

## Honest Conclusion

The systematic search is designed to **avoid fooling ourselves**. 

Most likely outcome: **Validation of known physics + confirmation that D = φ is specific to salt-leaching**

This is actually the BEST outcome because:
- It's defendable
- It's explainable
- It's practically useful
- It connects to deep theory
- It's not overhyped

**A solid, well-understood result is worth more than 10 questionable "discoveries".**

---

## References

### Percolation Theory
- Stauffer & Aharony (1994). *Introduction to Percolation Theory*
- Sahimi (2011). *Flow and Transport in Porous Media and Fractured Rock*

### Computational Topology
- Edelsbrunner & Harer (2010). *Computational Topology: An Introduction*
- Carlsson (2009). "Topology and data." *Bull. AMS* 46, 255-308

### Information Theory & Physics
- Cover & Thomas (2006). *Elements of Information Theory*
- Landauer (1961). "Irreversibility and heat generation in the computing process"

### Graph Theory in Porous Media
- Dong & Blunt (2009). "Pore-network extraction from micro-computerized-tomography images"
- Raoof & Hassanizadeh (2013). "A new method for generating pore-network models"

### Fractal Dimension
- Mandelbrot (1982). *The Fractal Geometry of Nature*
- Our work (2025). "D = φ in salt-leached scaffolds" (in preparation)
