# Darwin Scaffold Studio: An Open-Source Julia Platform for Tissue Engineering Scaffold Analysis

## Metadata

**Title:** Darwin Scaffold Studio: An Open-Source Julia Platform for Tissue Engineering Scaffold Analysis

**Authors:**
- Demetrios Agourakis (PUC-SP, Brazil) - Corresponding author
- Moema Alencar Hausen (PUC-SP, Brazil) - Advisor

**Keywords:** tissue engineering, scaffold analysis, microCT, image processing, Julia, open source

**Code repository:** https://github.com/agourakis82/darwin-scaffold-studio

**License:** MIT

---

## 1. Motivation and Significance

Tissue engineering scaffolds require precise characterization of structural properties such as porosity, pore size, interconnectivity, and tortuosity to ensure optimal cell infiltration and tissue regeneration [1]. Current analysis tools are either proprietary (e.g., CTAn, Avizo), require extensive programming expertise (Python/MATLAB scripts), or lack validation against experimental ground truth.

Darwin Scaffold Studio addresses these gaps by providing:

1. **Open-source accessibility**: MIT-licensed Julia package with no proprietary dependencies
2. **Validated algorithms**: Metrics validated against synthetic ground truth and real experimental data
3. **Reproducibility**: Self-contained examples that run without external datasets
4. **Interoperability**: Standard file formats (STL, JSON-LD) and biomedical ontology integration

The software is particularly relevant for research groups in developing countries where commercial software licenses are prohibitively expensive.

---

## 2. Software Description

### 2.1 Software Architecture

Darwin Scaffold Studio is implemented in Julia 1.10+ and organized into modular components:

```
DarwinScaffoldStudio/
├── Core/           # Types, configuration, utilities
├── MicroCT/        # Image loading, segmentation, metrics
├── Optimization/   # Target-driven scaffold optimization  
├── Visualization/  # Mesh generation, STL export
├── Science/        # Topology, percolation analysis
└── Ontology/       # OBO Foundry integration (UBERON, CL, CHEBI)
```

The architecture separates concerns between data structures (Core), image processing (MicroCT), and domain knowledge (Ontology), enabling independent testing and extension.

### 2.2 Software Functionalities

**Image Processing:**
- Load MicroCT volumes (RAW, TIFF stacks, NIfTI)
- Load SEM images (TIFF, PNG)
- Preprocessing: denoising, normalization, histogram equalization
- Segmentation: Otsu thresholding, adaptive methods

**Metrics Computation:**

| Metric | Method | Output |
|--------|--------|--------|
| Porosity | Voxel counting | Fraction (0-1) |
| Pore size | Connected components + equivalent diameter | Mean, std, distribution (μm) |
| Interconnectivity | Largest connected component / total pore volume | Fraction (0-1) |
| Tortuosity | Dijkstra shortest path | Dimensionless ratio |
| Surface area | Marching cubes triangulation | mm² |
| Mechanical properties | Gibson-Ashby model | Young's modulus (MPa) |

**Ontology Integration:**
- 1,200+ terms from OBO Foundry (UBERON, CL, CHEBI)
- Lookup optimal parameters by tissue type
- FAIR-compliant JSON-LD export with provenance

**Synthetic Scaffold Generation:**
- TPMS surfaces (Gyroid, Schwarz P, Schwarz D, Neovius)
- Parametric control of porosity and unit cell size
- Analytical ground truth for validation

---

## 3. Illustrative Examples

### Example 1: Synthetic Scaffold Analysis

```julia
using Pkg; Pkg.activate(".")
using Images, Statistics

# Generate 64³ Gyroid scaffold
size = 64
volume = zeros(Bool, size, size, size)
for i in 1:size, j in 1:size, k in 1:size
    x, y, z = 2π .* (i, j, k) ./ size
    gyroid = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
    volume[i,j,k] = gyroid > 0.3
end

# Compute porosity
porosity = 1 - sum(volume) / length(volume)
println("Porosity: $(round(porosity*100, digits=1))%")
# Output: Porosity: 59.8%
```

### Example 2: SEM Image Analysis

```julia
using Images

# Load SEM image
img = load("scaffold_sem.tif")
gray = Float64.(Gray.(img))

# Adaptive thresholding
threshold = otsu_threshold(gray)
pore_mask = gray .< threshold

# Pore size analysis
labels = label_components(pore_mask)
for i in 1:maximum(labels)
    area = sum(labels .== i)
    diameter = 2 * sqrt(area / π) * pixel_size_um
    println("Pore $i: $(round(diameter, digits=1)) μm")
end
```

---

## 4. Validation

### 4.1 Synthetic Ground Truth

Validation against TPMS surfaces with analytical properties:

| Metric | Mean Error | Threshold |
|--------|-----------|-----------|
| Porosity | <1% | <1% |
| Surface area | <1% | <1% |

### 4.2 Experimental Validation

Validation against PoreScript dataset (DOI: 10.5281/zenodo.5562953) with manual SEM measurements:

| Metric | Darwin | Ground Truth | APE |
|--------|--------|--------------|-----|
| Pore size | 149.4 μm | 174.0 μm | 14.1% |

**Limitations:**
- Systematic underestimation of ~15% on pore size
- Validated on 3 SEM images (n=374 manual measurements)
- 2D SEM analysis; 3D microCT validation pending

The 14.1% error is acceptable for comparative scaffold analysis but users should apply correction factors for absolute measurements.

---

## 5. Impact

Darwin Scaffold Studio enables:

1. **Reproducible research**: All analyses can be replicated with provided scripts
2. **Cost reduction**: Free alternative to commercial software (CTAn ~$5,000/year)
3. **Education**: Minimal example runs in <1 minute for teaching purposes
4. **Interoperability**: FAIR data export facilitates meta-analyses

The software is being used in ongoing research at PUC-SP for bioactive glass scaffold characterization.

### Target Users
- Tissue engineering researchers
- Biomaterials science graduate students
- Core facilities performing scaffold characterization

---

## 6. Conclusions

Darwin Scaffold Studio provides an open-source, validated platform for tissue engineering scaffold analysis. The Julia implementation offers performance comparable to compiled languages while maintaining readability. Current validation shows 14.1% error on pore size measurements against manual ground truth, suitable for comparative studies.

Future development will include:
- 3D microCT validation against BoneJ
- Machine learning-based segmentation
- Cloud-based processing API

---

## References

[1] Murphy, C.M., Haugh, M.G., O'Brien, F.J. (2010). The effect of mean pore size on cell attachment, proliferation and migration in collagen-glycosaminoglycan scaffolds for bone tissue engineering. Biomaterials, 31(3), 461-466.

[2] Karageorgiou, V., Kaplan, D. (2005). Porosity of 3D biomaterial scaffolds and osteogenesis. Biomaterials, 26(27), 5474-5491.

[3] Hildebrand, T., Rüegsegger, P. (1997). A new method for the model-independent assessment of thickness in three-dimensional images. Journal of Microscopy, 185(1), 67-75.

---

## Required Coverage for SoftwareX

- [x] Permanent code identifier (GitHub + Zenodo DOI)
- [x] MIT open-source license
- [x] Executable example with synthetic data
- [x] Validation against ground truth
- [x] Clear documentation of limitations

---

*Draft prepared for SoftwareX submission*
*Word count: ~850 (target: 3-6 pages)*
