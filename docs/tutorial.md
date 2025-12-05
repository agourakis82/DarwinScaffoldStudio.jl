# Darwin Scaffold Studio - Tutorial

Complete guide for analyzing tissue engineering scaffolds.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Loading MicroCT Data](#loading-microct-data)
4. [Loading SEM Data](#loading-sem-data)
5. [Segmentation](#segmentation)
6. [Computing Metrics](#computing-metrics)
7. [Optimization](#optimization)
8. [Visualization](#visualization)
9. [Ontology Integration](#ontology-integration)
10. [FAIR Data Export](#fair-data-export)

---

## Installation

### Option 1: Julia Package

```julia
# Clone the repository
using Pkg
Pkg.develop(path="/path/to/darwin-scaffold-studio")

# Or directly from GitHub
Pkg.add(url="https://github.com/your-org/darwin-scaffold-studio")
```

### Option 2: Docker

```bash
# Build the image
docker build -t darwin-scaffold-studio .

# Run interactively
docker run -it -v /path/to/your/data:/app/user_data darwin-scaffold-studio

# Or use docker-compose
docker-compose up darwin-studio
```

---

## Quick Start

```julia
# Load the module
include("src/DarwinScaffoldStudio.jl")
using .DarwinScaffoldStudio

# Load your MicroCT data
img = load_microct("path/to/scaffold.raw", (512, 512, 512))

# Preprocess
processed = preprocess_image(img; denoise=true, normalize=true)

# Segment
binary = segment_scaffold(processed, "otsu")

# Compute metrics
metrics = compute_metrics(binary, 10.0)  # 10 μm voxel size

println("Porosity: $(metrics.porosity * 100)%")
println("Mean pore size: $(metrics.mean_pore_size_um) μm")
println("Interconnectivity: $(metrics.interconnectivity * 100)%")
```

---

## Loading MicroCT Data

### RAW Format

```julia
# Load RAW file (binary voxel data)
# Specify dimensions: (x, y, z)
img = load_microct("scaffold.raw", (512, 512, 512))

# With custom data type (default: UInt16)
img = load_microct("scaffold.raw", (512, 512, 512); dtype=UInt8)
```

### DICOM Stack

```julia
# Load DICOM directory
img = load_dicom_stack("path/to/dicom_folder/")
```

### NIfTI Format

```julia
# Load NIfTI file
img = load_nifti("scaffold.nii.gz")
```

### TIFF Stack

```julia
# Load TIFF image stack
img = load_tiff_stack("path/to/tiff_folder/")
```

---

## Loading SEM Data

For SEM (Scanning Electron Microscopy) images:

```julia
# Load single SEM image
sem_img = load_sem_image("sem_scaffold.tif")

# Load multiple SEM images
sem_stack = load_sem_stack("path/to/sem_images/")

# Analyze SEM for pore size distribution
# (2D analysis - different from MicroCT 3D)
pore_stats = analyze_sem_pores(sem_img; 
    scale_um_per_pixel=0.5,  # Calibration
    min_pore_size_um=10.0,
    max_pore_size_um=500.0
)

println("Mean pore size: $(pore_stats.mean_pore_size_um) μm")
println("Pore count: $(pore_stats.pore_count)")
```

---

## Segmentation

### Otsu's Method (Automatic)

```julia
# Automatic threshold using Otsu's method
binary = segment_scaffold(img, "otsu")
```

### Manual Threshold

```julia
# Manual threshold (0-1 normalized)
binary = segment_scaffold(img, "manual"; threshold=0.35)
```

### Adaptive Threshold

```julia
# Adaptive local thresholding
binary = segment_scaffold(img, "adaptive"; window_size=51)
```

### K-Means Clustering

```julia
# K-means with 2 clusters (scaffold/pore)
binary = segment_scaffold(img, "kmeans"; n_clusters=2)
```

---

## Computing Metrics

### Basic Metrics

```julia
# Compute all metrics
# voxel_size_um: size of one voxel in micrometers
metrics = compute_metrics(binary, 10.0)

# Access individual metrics
println("Porosity: ", metrics.porosity)
println("Mean pore size: ", metrics.mean_pore_size_um, " μm")
println("Interconnectivity: ", metrics.interconnectivity)
println("Tortuosity: ", metrics.tortuosity)
println("Specific surface area: ", metrics.specific_surface_area, " mm⁻¹")
println("Elastic modulus: ", metrics.elastic_modulus, " MPa")
println("Yield strength: ", metrics.yield_strength, " MPa")
println("Permeability: ", metrics.permeability, " m²")
```

### Literature Validation

Expected ranges for bone tissue engineering (Murphy et al., 2010; Karageorgiou, 2005):

| Metric | Optimal Range |
|--------|---------------|
| Porosity | 85-95% |
| Pore size | 100-300 μm |
| Interconnectivity | >90% |
| Tortuosity | 1.0-1.5 |

```julia
# Check if metrics meet requirements
function validate_for_bone(m::ScaffoldMetrics)
    valid = true
    
    if !(0.85 <= m.porosity <= 0.95)
        println("⚠️  Porosity $(m.porosity) outside optimal range [0.85-0.95]")
        valid = false
    end
    
    if !(100 <= m.mean_pore_size_um <= 300)
        println("⚠️  Pore size $(m.mean_pore_size_um)μm outside optimal range [100-300]")
        valid = false
    end
    
    if m.interconnectivity < 0.90
        println("⚠️  Interconnectivity $(m.interconnectivity) below 90%")
        valid = false
    end
    
    return valid
end

validate_for_bone(metrics)
```

---

## Optimization

### Define Target Parameters

```julia
# Define optimization targets based on tissue type
target = ScaffoldParameters(
    0.90,           # target porosity (90%)
    200.0,          # target pore size (200 μm)
    0.95,           # target interconnectivity (95%)
    1.15,           # target tortuosity
    (5.0, 5.0, 5.0), # volume (5×5×5 mm³)
    10.0            # resolution (10 μm)
)
```

### Run Optimization

```julia
# Create optimizer
optimizer = ScaffoldOptimizer(voxel_size_um=10.0)

# Optimize scaffold
results = optimize_scaffold(optimizer, binary, target)

# View results
println("Original porosity: ", results.original_metrics.porosity)
println("Optimized porosity: ", results.optimized_metrics.porosity)
println("Improvement: ", results.improvement_percent["porosity"], "%")

# Get optimized volume
optimized_scaffold = results.optimized_volume
```

---

## Visualization

### Create 3D Mesh

```julia
# Create mesh from binary volume
vertices, faces = create_mesh_simple(binary, 10.0)  # 10 μm voxel size

println("Vertices: ", size(vertices, 1))
println("Faces: ", size(faces, 1))
```

### Export STL

```julia
# Export to STL for 3D printing or visualization
export_stl("scaffold_mesh.stl", vertices, faces)
```

### Export PLY (with colors)

```julia
# Export to PLY with vertex colors
colors = compute_curvature_colors(vertices, faces)
export_ply("scaffold_colored.ply", vertices, faces, colors)
```

---

## Ontology Integration

### Lookup Tissue Parameters

```julia
using .DarwinScaffoldStudio.Ontology.OntologyManager

# Get optimal parameters for bone tissue
bone_info = lookup_tissue("bone")
println("Ontology ID: ", bone_info.ontology_id)  # UBERON:0002481
println("Optimal porosity: ", bone_info.optimal_porosity)
println("Optimal pore size: ", bone_info.optimal_pore_size)
```

### Lookup Cell Requirements

```julia
# Get cell information
osteoblast = lookup_cell("osteoblast")
println("Cell ID: ", osteoblast.ontology_id)  # CL:0000062
println("Size range: ", osteoblast.size_um, " μm")
```

### Lookup Material Properties

```julia
# Get material information
ha = lookup_material("hydroxyapatite")
println("Material ID: ", ha.ontology_id)  # CHEBI:52254
println("Elastic modulus: ", ha.elastic_modulus_gpa, " GPa")
println("Biocompatibility: ", ha.biocompatibility)
```

---

## FAIR Data Export

Export results in FAIR (Findable, Accessible, Interoperable, Reusable) format:

```julia
using .DarwinScaffoldStudio.Ontology.OntologyManager

# Export scaffold analysis to JSON-LD with ontology bindings
export_fair_json("scaffold_analysis.jsonld", 
    scaffold_name = "PCL_Gyroid_P85",
    metrics = metrics,
    tissue_type = "bone",
    material = "polycaprolactone",
    fabrication = "3d_bioprinting"
)
```

This produces a JSON-LD file with:
- Schema.org vocabulary
- OBO Foundry ontology IDs (UBERON, CL, CHEBI)
- Full provenance tracking
- Machine-readable metadata

---

## Complete Workflow Example

```julia
# ============================================
# Complete Scaffold Analysis Workflow
# ============================================

include("src/DarwinScaffoldStudio.jl")
using .DarwinScaffoldStudio

# 1. Load data
println("Loading MicroCT data...")
img = load_microct("data/my_scaffold.raw", (512, 512, 512))

# 2. Preprocess
println("Preprocessing...")
processed = preprocess_image(img; denoise=true, normalize=true)

# 3. Segment
println("Segmenting scaffold...")
binary = segment_scaffold(processed, "otsu")

# 4. Compute metrics
println("Computing metrics...")
metrics = compute_metrics(binary, 10.0)

println("\n=== SCAFFOLD METRICS ===")
println("Porosity: $(round(metrics.porosity * 100, digits=1))%")
println("Mean pore size: $(round(metrics.mean_pore_size_um, digits=1)) μm")
println("Interconnectivity: $(round(metrics.interconnectivity * 100, digits=1))%")
println("Tortuosity: $(round(metrics.tortuosity, digits=2))")

# 5. Check against literature
println("\n=== VALIDATION ===")
using .DarwinScaffoldStudio.Ontology.OntologyManager
bone = lookup_tissue("bone")

if bone.optimal_porosity[1] <= metrics.porosity <= bone.optimal_porosity[2]
    println("✅ Porosity within optimal range for bone")
else
    println("⚠️  Porosity outside optimal range")
end

# 6. Optimize if needed
if metrics.porosity < 0.85
    println("\nOptimizing scaffold...")
    
    target = ScaffoldParameters(0.90, 200.0, 0.95, 1.15, (5.0, 5.0, 5.0), 10.0)
    optimizer = ScaffoldOptimizer(voxel_size_um=10.0)
    results = optimize_scaffold(optimizer, binary, target)
    
    println("Optimization complete!")
    println("New porosity: $(round(results.optimized_metrics.porosity * 100, digits=1))%")
end

# 7. Export results
println("\nExporting mesh...")
vertices, faces = create_mesh_simple(binary, 10.0)
export_stl("results/scaffold.stl", vertices, faces)

println("\nExporting FAIR data...")
export_fair_json("results/scaffold_analysis.jsonld",
    scaffold_name = "My_Scaffold",
    metrics = metrics,
    tissue_type = "bone"
)

println("\n✅ Analysis complete!")
```

---

## Troubleshooting

### Memory Issues

For large datasets (>1GB):

```julia
# Use memory-mapped loading
img = load_microct("large_scaffold.raw", (1024, 1024, 1024); mmap=true)

# Process in chunks
for z in 1:100:1024
    chunk = img[:, :, z:min(z+99, 1024)]
    # Process chunk...
end
```

### GPU Acceleration

If available:

```julia
# Enable GPU processing
set_config(:enable_gpu, true)

# Check GPU status
println("GPU enabled: ", get_config(:enable_gpu))
```

### Common Errors

1. **"Module not found"**: Run `Pkg.instantiate()` to install dependencies
2. **"Voxel size mismatch"**: Ensure consistent units (always micrometers)
3. **"Segmentation failed"**: Try different methods or adjust threshold

---

## References

1. Murphy CM, O'Brien FJ (2010). Understanding the effect of mean pore size on cell activity in collagen-glycosaminoglycan scaffolds. Cell Adh Migr 4(3):377-381.

2. Karageorgiou V, Kaplan D (2005). Porosity of 3D biomaterial scaffolds and osteogenesis. Biomaterials 26(27):5474-5491.

3. Gibson LJ, Ashby MF (1997). Cellular Solids: Structure and Properties. Cambridge University Press.

---

*Darwin Scaffold Studio - Tissue Engineering Scaffold Analysis Platform*
