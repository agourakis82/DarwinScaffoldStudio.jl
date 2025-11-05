# 🧬 Darwin Scaffold Studio

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17535484.svg)](https://doi.org/10.5281/zenodo.17535484)

**"Ciência rigorosa. Resultados honestos. Impacto real."**

## Q1-Level MicroCT and SEM Analysis Platform

Production-ready software for tissue engineering scaffold analysis with validation against Q1 literature (Murphy et al. 2010, Karageorgiou & Kaplan 2005, Gibson & Ashby 1997).

### 🚀 Features

- ✅ **MicroCT/SEM Analysis:** Upload TIFF/NIfTI/DICOM, preprocessing Q1-validated
- ✅ **3D Visualization:** Interactive Plotly with zoom, rotation, material styles
- ✅ **Parametric Optimization:** Target-based scaffold design
- ✅ **Mechanical Properties:** Gibson-Ashby predictions (E*, σ*, k)
- ✅ **Cell Viability:** Detection, morphology, coverage analysis
- ✅ **STL Export:** 3D printing ready
- ✅ **Q1 Validation:** Murphy 2010, Karageorgiou 2005, Gibson-Ashby 1997

### 📊 Metrics Calculated

- Porosity (%)
- Mean pore size (µm)
- Interconnectivity (%)
- Tortuosity
- Specific surface area (mm⁻¹)
- Elastic modulus, yield strength, permeability
- Cell viability, coverage, density

### 🌐 Public Access

- **Landing Page:** https://studio.agourakis.med.br
- **Files Upload:** https://files.agourakis.med.br

### 📚 Citation

If you use this software in your research, please cite:

```
Agourakis, D.C. (2025). Darwin Scaffold Studio: Q1-Level MicroCT and SEM 
Analysis Platform. Version 1.0.0 [Software]. Zenodo. 
https://doi.org/10.5281/zenodo.17535484
```

### 📖 References

- Murphy CM, Haugh MG, O'Brien FJ. The effect of mean pore size on cell attachment, proliferation and migration in collagen-glycosaminoglycan scaffolds for bone tissue engineering. *Biomaterials*. 2010;31(3):461-466.
- Karageorgiou V, Kaplan D. Porosity of 3D biomaterial scaffolds and osteogenesis. *Biomaterials*. 2005;26(27):5474-5491.
- Gibson LJ, Ashby MF. *Cellular Solids: Structure and Properties*. 2nd ed. Cambridge University Press; 1997.

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Darwin Scaffold Studio
streamlit run apps/production/darwin_scaffold_studio.py --server.port 8600
```

See [docs/INSTRUCOES_INICIAR_STUDIO.md](docs/INSTRUCOES_INICIAR_STUDIO.md) for complete instructions.

### 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

Developed with Q1 scientific rigor for tissue engineering research at PUCRS.

---

**"Rigorous science. Honest results. Real impact."**

