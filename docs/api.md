# Darwin Scaffold Studio - API Reference

Complete API documentation for all modules.

---

## Core Types

### ScaffoldMetrics

Structure containing computed scaffold metrics.

```julia
struct ScaffoldMetrics
    porosity::Float64              # Porosity ratio (0-1)
    mean_pore_size_um::Float64     # Mean pore size in micrometers
    interconnectivity::Float64     # Pore interconnectivity (0-1)
    tortuosity::Float64            # Path tortuosity (≥1.0)
    specific_surface_area::Float64 # Surface area per volume (mm⁻¹)
    elastic_modulus::Float64       # Estimated elastic modulus (MPa)
    yield_strength::Float64        # Estimated yield strength (MPa)
    permeability::Float64          # Estimated permeability (m²)
end
```

### ScaffoldParameters

Target parameters for scaffold optimization.

```julia
struct ScaffoldParameters
    porosity_target::Float64                    # Target porosity
    pore_size_target_um::Float64               # Target pore size (μm)
    interconnectivity_target::Float64          # Target interconnectivity
    tortuosity_target::Float64                 # Target tortuosity
    volume_mm3::Tuple{Float64,Float64,Float64} # Physical dimensions (mm)
    resolution_um::Float64                     # Voxel resolution (μm)
end
```

### OptimizationResults

Results from scaffold optimization.

```julia
struct OptimizationResults
    optimized_volume::Array{Bool,3}        # Optimized 3D binary volume
    original_metrics::ScaffoldMetrics      # Metrics before optimization
    optimized_metrics::ScaffoldMetrics     # Metrics after optimization
    improvement_percent::Dict{String,Float64}  # Improvement percentages
    fabrication_method::String             # Recommended fabrication method
    fabrication_parameters::Dict{String,Any}   # Fabrication parameters
end
```

---

## MicroCT Module

### load_microct

Load MicroCT data from RAW file.

```julia
load_microct(filepath::String, dimensions::Tuple{Int,Int,Int}; 
             dtype::DataType=UInt16, mmap::Bool=false) -> Array{Float64,3}
```

**Arguments:**
- `filepath`: Path to RAW file
- `dimensions`: (x, y, z) dimensions
- `dtype`: Data type (UInt8, UInt16, Float32)
- `mmap`: Use memory-mapped IO for large files

**Returns:** Normalized 3D array (values 0-1)

---

### preprocess_image

Apply preprocessing pipeline to image.

```julia
preprocess_image(image::Array{<:Real,3}; 
                 denoise::Bool=true, 
                 normalize::Bool=true,
                 remove_artifacts::Bool=false) -> Array{Float64,3}
```

**Arguments:**
- `image`: Input 3D image
- `denoise`: Apply Gaussian denoising
- `normalize`: Normalize to [0, 1]
- `remove_artifacts`: Remove ring artifacts

**Returns:** Preprocessed image

---

### segment_scaffold

Segment scaffold from background.

```julia
segment_scaffold(image::Array{<:Real,3}, method::String; 
                 kwargs...) -> Array{Bool,3}
```

**Arguments:**
- `image`: Preprocessed image
- `method`: Segmentation method ("otsu", "manual", "adaptive", "kmeans")

**Keyword Arguments (by method):**
- `manual`: `threshold::Float64` - Manual threshold value
- `adaptive`: `window_size::Int` - Local window size
- `kmeans`: `n_clusters::Int` - Number of clusters

**Returns:** Binary 3D volume (true = material, false = pore)

---

### compute_metrics

Compute comprehensive scaffold metrics.

```julia
compute_metrics(scaffold::Array{Bool,3}, voxel_size_um::Real) -> ScaffoldMetrics
```

**Arguments:**
- `scaffold`: Binary 3D volume
- `voxel_size_um`: Voxel size in micrometers

**Returns:** `ScaffoldMetrics` struct with all computed values

---

## Optimization Module

### ScaffoldOptimizer

Scaffold optimization engine.

```julia
ScaffoldOptimizer(; voxel_size_um::Float64=10.0,
                   max_iterations::Int=100,
                   convergence_threshold::Float64=0.01)
```

**Fields:**
- `voxel_size_um`: Voxel resolution
- `max_iterations`: Maximum optimization iterations
- `convergence_threshold`: Convergence criterion

---

### optimize_scaffold

Optimize scaffold towards target parameters.

```julia
optimize_scaffold(optimizer::ScaffoldOptimizer, 
                  scaffold::Array{Bool,3},
                  target::ScaffoldParameters) -> OptimizationResults
```

**Arguments:**
- `optimizer`: Configured optimizer
- `scaffold`: Input binary volume
- `target`: Target parameters

**Returns:** `OptimizationResults` with optimized volume and metrics

---

## Visualization Module

### create_mesh_simple

Create surface mesh from binary volume.

```julia
create_mesh_simple(scaffold::Array{Bool,3}, 
                   voxel_size_um::Real) -> Tuple{Matrix, Matrix}
```

**Arguments:**
- `scaffold`: Binary 3D volume
- `voxel_size_um`: Voxel size for scaling

**Returns:** `(vertices, faces)` - Nx3 matrices

---

### export_stl

Export mesh to STL format.

```julia
export_stl(filepath::String, vertices::Matrix, faces::Matrix)
```

**Arguments:**
- `filepath`: Output STL file path
- `vertices`: Vertex positions (Nx3)
- `faces`: Face indices (Mx3)

---

## Ontology Module

### OBOFoundry

Access to OBO Foundry ontology terms.

```julia
# Dictionaries with OBOTerm entries
OBOFoundry.UBERON  # Anatomy (tissues, organs)
OBOFoundry.CL      # Cell types
OBOFoundry.CHEBI   # Chemicals/materials
OBOFoundry.GO      # Biological processes
OBOFoundry.DOID    # Diseases
```

**Example:**
```julia
bone = OBOFoundry.UBERON["bone tissue"]
# bone.id => "UBERON:0002481"
# bone.name => "bone tissue"
```

---

### OntologyManager.lookup_tissue

Look up tissue information.

```julia
lookup_tissue(name::String) -> TissueInfo
```

**Arguments:**
- `name`: Tissue name (e.g., "bone", "cartilage", "skin")

**Returns:** `TissueInfo` struct with:
- `ontology_id`: UBERON ID
- `name`: Tissue name
- `optimal_porosity`: Tuple (min, max)
- `optimal_pore_size`: Tuple (min, max) in μm
- `cell_types`: Associated cell types
- `ecm_composition`: ECM components

---

### OntologyManager.lookup_cell

Look up cell type information.

```julia
lookup_cell(name::String) -> CellInfo
```

**Arguments:**
- `name`: Cell name (e.g., "osteoblast", "chondrocyte")

**Returns:** `CellInfo` struct with:
- `ontology_id`: CL ID
- `name`: Cell name
- `size_um`: Tuple (min, max) cell diameter
- `doubling_time_hours`: Proliferation rate
- `oxygen_consumption`: O₂ consumption rate
- `markers`: Surface markers

---

### OntologyManager.lookup_material

Look up biomaterial information.

```julia
lookup_material(name::String) -> MaterialInfo
```

**Arguments:**
- `name`: Material name (e.g., "hydroxyapatite", "pcl", "collagen")

**Returns:** `MaterialInfo` struct with:
- `ontology_id`: CHEBI ID
- `name`: Material name
- `material_class`: :polymer, :ceramic, :metal, :composite
- `elastic_modulus_gpa`: Elastic modulus
- `degradation_time_months`: Degradation timeline
- `biocompatibility`: :excellent, :good, :moderate, :poor

---

### OntologyManager.export_fair_json

Export analysis in FAIR JSON-LD format.

```julia
export_fair_json(filepath::String;
                 scaffold_name::String,
                 metrics::ScaffoldMetrics,
                 tissue_type::String,
                 material::String="unknown",
                 fabrication::String="unknown")
```

**Arguments:**
- `filepath`: Output JSON-LD file
- `scaffold_name`: Identifier for the scaffold
- `metrics`: Computed metrics
- `tissue_type`: Target tissue type
- `material`: Scaffold material
- `fabrication`: Fabrication method

**Output:** JSON-LD file with Schema.org vocabulary and OBO ontology IDs

---

## Science Module

### compute_euler_number

Compute Euler characteristic of scaffold.

```julia
compute_euler_number(scaffold::Array{Bool,3}) -> Int
```

**Returns:** Euler number (χ = vertices - edges + faces)

---

### compute_tortuosity

Compute path tortuosity through pore network.

```julia
compute_tortuosity(scaffold::Array{Bool,3}; 
                   axis::Symbol=:z) -> Float64
```

**Arguments:**
- `scaffold`: Binary volume
- `axis`: Direction for path analysis (:x, :y, :z)

**Returns:** Tortuosity value (≥1.0, where 1.0 = straight path)

---

### label_connected_components

Label connected components in volume.

```julia
label_connected_components(scaffold::Array{Bool,3}; 
                          connectivity::Int=6) -> Array{Int,3}
```

**Arguments:**
- `scaffold`: Binary volume
- `connectivity`: 6, 18, or 26 connectivity

**Returns:** Labeled volume (0 = background, 1..n = components)

---

## TPMS Functions

Triply Periodic Minimal Surface implicit functions for synthetic scaffold generation.

```julia
# Gyroid surface
gyroid(x, y, z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

# Diamond surface (Schwarz D)
diamond(x, y, z) = sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + 
                   cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z)

# Schwarz P surface
schwarz_p(x, y, z) = cos(x) + cos(y) + cos(z)

# Neovius surface
neovius(x, y, z) = 3*(cos(x) + cos(y) + cos(z)) + 4*cos(x)*cos(y)*cos(z)
```

**Usage:**
```julia
# Generate TPMS scaffold
function generate_tpms(func, size, porosity)
    scaffold = zeros(Bool, size, size, size)
    scale = 2π / size
    
    # Sample to find threshold
    samples = [func(i*scale, j*scale, k*scale) 
               for i=1:size, j=1:size, k=1:size]
    threshold = quantile(vec(samples), porosity)
    
    # Generate scaffold
    for i=1:size, j=1:size, k=1:size
        scaffold[i,j,k] = func(i*scale, j*scale, k*scale) > threshold
    end
    
    return scaffold
end

# Create 85% porosity Gyroid
scaffold = generate_tpms(gyroid, 100, 0.85)
```

---

## Configuration

### get_config / set_config

Global configuration management.

```julia
get_config(key::Symbol) -> Any
set_config(key::Symbol, value::Any)
```

**Available Keys:**
- `:enable_gpu` - Enable GPU acceleration (Bool)
- `:default_voxel_size` - Default voxel size in μm (Float64)
- `:num_threads` - Number of threads (Int)
- `:log_level` - Logging level (:debug, :info, :warn, :error)

---

## Error Handling

All functions use Julia's exception system:

```julia
try
    metrics = compute_metrics(scaffold, voxel_size)
catch e
    if e isa ArgumentError
        println("Invalid input: ", e.msg)
    elseif e isa DimensionMismatch
        println("Dimension error: ", e.msg)
    else
        rethrow()
    end
end
```

---

*Darwin Scaffold Studio v0.2.0*
