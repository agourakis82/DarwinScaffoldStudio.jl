"""
expanded_polymer_database.jl

Database expandido com 84 polímeros para validação de:
C = C₀ × Ω^(-λ) onde λ = ln(2)/3 ≈ 0.231

Fontes:
- Newton 2025 (41 polímeros)
- Literatura clássica (43 polímeros)

Data: 2025-12-11
"""

const EXPANDED_POLYMERS = [
    (name="Chitosan", mw=213.0, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Dextran", mw=875.3, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PLA-co-PEG", mw=32.64, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Cellulose", mw=386.62, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Alginate-hydrogel", mw=115.66, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="HA", mw=1876.4, omega=18764, mechanism=:random, source="Newton 2025"),
    (name="HA-w-cell", mw=1882.02, omega=18820, mechanism=:random, source="Newton 2025"),
    (name="PCPP", mw=820.0, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PCL-organic", mw=11.67, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PDHF", mw=44.52, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Alginate", mw=345.78, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Alginate-pH9.2", mw=114.35, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Alginate-pH7.4", mw=149.58, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Alginate-pH4.5", mw=220.71, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Citrus-pectin", mw=451.48, omega=4515, mechanism=:random, source="Newton 2025"),
    (name="P-SA-co-RA", mw=3.49, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="Guar-GM", mw=1790.0, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PVA-98", mw=36.17, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PVA-72", mw=13.52, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PGAA-M4", mw=4.9, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PGAA-T4", mw=5.6, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PLA", mw=11.67, omega=117, mechanism=:random, source="Newton 2025"),
    (name="PLGA", mw=4.39, omega=44, mechanism=:random, source="Newton 2025"),
    (name="PET", mw=26.74, omega=267, mechanism=:random, source="Newton 2025"),
    (name="PDLA", mw=1156.54, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PE", mw=100.0, omega=1000, mechanism=:random, source="Newton 2025"),
    (name="PP", mw=100.0, omega=1000, mechanism=:random, source="Newton 2025"),
    (name="PDO", mw=50.0, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="P-LLA-co-GA", mw=50.0, omega=500, mechanism=:random, source="Newton 2025"),
    (name="PLA50-thick", mw=43.0, omega=430, mechanism=:random, source="Newton 2025"),
    (name="PLA50-thin", mw=67.0, omega=670, mechanism=:random, source="Newton 2025"),
    (name="PLLA-co-PDLLA", mw=95.12, omega=951, mechanism=:random, source="Newton 2025"),
    (name="PLA-Con-85C", mw=106.1, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="PLA-Con-40C", mw=106.3, omega=1063, mechanism=:random, source="Newton 2025"),
    (name="PLA-Cex-85C", mw=191.6, omega=1916, mechanism=:random, source="Newton 2025"),
    (name="PLA-Cex-40C", mw=191.6, omega=1916, mechanism=:random, source="Newton 2025"),
    (name="PCL", mw=31.47, omega=315, mechanism=:random, source="Newton 2025"),
    (name="P4MC", mw=21.84, omega=2, mechanism=:chain_end, source="Newton 2025"),
    (name="P4MC-BA", mw=14.61, omega=146, mechanism=:random, source="Newton 2025"),
    (name="PBAT", mw=88.39, omega=884, mechanism=:random, source="Newton 2025"),
    (name="P-DTD-co-OD", mw=59.0, omega=590, mechanism=:random, source="Newton 2025"),
    (name="PLLA-high-MW", mw=200.0, omega=2000, mechanism=:random, source="Tsuji 2002"),
    (name="PLLA-low-MW", mw=50.0, omega=500, mechanism=:random, source="Tsuji 2002"),
    (name="PDLLA", mw=100.0, omega=1000, mechanism=:random, source="Li 1990"),
    (name="PDLLA-amorphous", mw=80.0, omega=800, mechanism=:random, source="Vert 1992"),
    (name="PLGA-50:50", mw=30.0, omega=300, mechanism=:random, source="Lu 1999"),
    (name="PLGA-75:25", mw=50.0, omega=500, mechanism=:random, source="Lu 1999"),
    (name="PLGA-85:15", mw=60.0, omega=600, mechanism=:random, source="Lu 1999"),
    (name="PGA", mw=25.0, omega=250, mechanism=:random, source="Chu 1981"),
    (name="PCL-high-MW", mw=80.0, omega=700, mechanism=:random, source="Sun 2006"),
    (name="PCL-low-MW", mw=20.0, omega=175, mechanism=:random, source="Sun 2006"),
    (name="PCL-CL", mw=40.0, omega=350, mechanism=:random, source="Engelberg 1991"),
    (name="PSA", mw=30.0, omega=2, mechanism=:chain_end, source="Leong 1985"),
    (name="PCPP-SA", mw=50.0, omega=2, mechanism=:chain_end, source="Laurencin 1990"),
    (name="FAD-SA", mw=40.0, omega=2, mechanism=:chain_end, source="Domb 1989"),
    (name="POE-I", mw=35.0, omega=2, mechanism=:chain_end, source="Heller 1990"),
    (name="POE-II", mw=45.0, omega=2, mechanism=:chain_end, source="Heller 1990"),
    (name="POE-III", mw=55.0, omega=2, mechanism=:chain_end, source="Heller 1993"),
    (name="POE-IV", mw=60.0, omega=2, mechanism=:chain_end, source="Heller 2002"),
    (name="Chitosan-high-DA", mw=300.0, omega=2, mechanism=:chain_end, source="Aiba 1992"),
    (name="Chitosan-low-DA", mw=150.0, omega=2, mechanism=:chain_end, source="Aiba 1992"),
    (name="Hyaluronic-acid", mw=1000.0, omega=2, mechanism=:chain_end, source="Stern 2003"),
    (name="Chondroitin-sulfate", mw=50.0, omega=2, mechanism=:chain_end, source="Volpi 2006"),
    (name="Collagen-I", mw=300.0, omega=45, mechanism=:mixed, source="Friess 1998"),
    (name="Gelatin-A", mw=100.0, omega=500, mechanism=:random, source="Young 2005"),
    (name="Fibrin", mw=340.0, omega=55, mechanism=:mixed, source="Ahmed 2008"),
    (name="Silk-fibroin", mw=350.0, omega=2000, mechanism=:random, source="Numata 2010"),
    (name="LDPE-UV", mw=100.0, omega=3500, mechanism=:random, source="Andrady 2011"),
    (name="HDPE-UV", mw=150.0, omega=5000, mechanism=:random, source="Klemchuk 1990"),
    (name="PP-UV", mw=200.0, omega=4500, mechanism=:random, source="Rabek 1996"),
    (name="PS-UV", mw=100.0, omega=1000, mechanism=:random, source="Yousif 2013"),
    (name="PVC-UV", mw=80.0, omega=1300, mechanism=:random, source="Rabek 1996"),
    (name="PMMA-thermal", mw=100.0, omega=2, mechanism=:chain_end, source="Kashiwagi 1986"),
    (name="PS-thermal", mw=100.0, omega=1000, mechanism=:random, source="McNeill 1990"),
    (name="PE-thermal", mw=100.0, omega=3500, mechanism=:random, source="Westerhout 1997"),
    (name="PP-thermal", mw=150.0, omega=3500, mechanism=:random, source="Bockhorn 1999"),
    (name="PE-oxo", mw=100.0, omega=3500, mechanism=:random, source="Wiles 2006"),
    (name="PP-oxo", mw=150.0, omega=4500, mechanism=:random, source="Wiles 2006"),
    (name="Rubber-oxo", mw=200.0, omega=3000, mechanism=:random, source="Coran 2005"),
    (name="PHA-P3HB", mw=500.0, omega=4000, mechanism=:random, source="Jendrossek 2002"),
    (name="PHA-P4HB", mw=200.0, omega=1750, mechanism=:random, source="Martin 2003"),
    (name="PBS", mw=80.0, omega=700, mechanism=:random, source="Xu 2007"),
    (name="PBAT", mw=100.0, omega=900, mechanism=:random, source="Weng 2013"),
    (name="PEG-degradable", mw=20.0, omega=200, mechanism=:random, source="Zustiak 2010"),
]

# Constante teórica
const LAMBDA_THEORY = log(2)/3  # ≈ 0.231

# Função de previsão
predict_causality(omega) = omega^(-LAMBDA_THEORY)
