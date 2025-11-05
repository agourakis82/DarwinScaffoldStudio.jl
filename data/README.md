# 📊 Example Data

This directory contains example scaffold images for testing Darwin Scaffold Studio.

## 🔬 MicroCT Data

**File:** `biomaterials/microct/raw/demo_test.tif`
- **Type:** MicroCT 3D volume
- **Size:** 128³ voxels
- **Format:** TIFF (uncompressed)
- **Purpose:** Demonstration and testing

**Metrics (expected):**
- Porosity: ~65%
- Mean pore size: ~105 µm
- Interconnectivity: High

## 🔍 SEM Data

**File:** `biomaterials/sem/raw/D1_20x_sem.tiff`
- **Type:** SEM 2D image
- **Magnification:** 20x
- **Format:** TIFF
- **Sample:** D1 scaffold (PLDLA + 1% TEC)
- **Purpose:** Surface morphology analysis

**Metrics (expected):**
- Pore count: ~60
- Mean pore size: ~74 µm
- Circularity: Variable

## 🚀 How to Use

### In Darwin Scaffold Studio:

1. Run the app:
```bash
streamlit run apps/production/darwin_scaffold_studio.py --server.port 8600
```

2. Upload one of the example files
3. Run analysis
4. Explore results

### Via Python:

```python
from apps.production.scaffold_optimizer import ScaffoldOptimizer
import tifffile

# Load data
volume = tifffile.imread('data/biomaterials/microct/raw/demo_test.tif')

# Analyze
optimizer = ScaffoldOptimizer()
metrics = optimizer.analyze_scaffold(volume)

print(f"Porosity: {metrics['porosity']:.1f}%")
print(f"Pore size: {metrics['mean_pore_size']:.1f} µm")
```

## 📝 Adding Your Own Data

Place your files in:
- **MicroCT:** `biomaterials/microct/raw/`
- **SEM:** `biomaterials/sem/raw/`

**Supported formats:**
- TIFF (`.tif`, `.tiff`)
- NIfTI (`.nii`, `.nii.gz`)
- DICOM (`.dcm`)

**Note:** Large data files (>10 MB) should be added to `.gitignore` to avoid repository bloat.

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**

