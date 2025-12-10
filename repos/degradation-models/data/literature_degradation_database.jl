"""
    Literature Degradation Database

Comprehensive dataset of PLA/PLDLA/PDLLA degradation from literature.
Collected through deep research for universal model training.

Sources:
- Kaique Hergesel PhD Thesis (2025) - PLDLA 70:30
- PMC3359772 - Industrial vs Lab PLLA (2012)
- PMC5544909 - P(TMC-co-DLLA) (2017)
- Tsuji & Ikada series - Systematic PLLA studies
- Bergsma et al. (1995) - Long-term PLLA
- Li, Garreau, Vert (1990) - Size dependence
- Frontiers Bioeng (2024) - 3D printed PLLA

Author: Darwin Scaffold Studio
Date: December 2025
"""

# =============================================================================
# COMPREHENSIVE DEGRADATION DATABASE
# =============================================================================

"""
Each entry contains:
- id: Unique identifier
- source: Literature reference
- material: Polymer type
- ratio_L: L-lactide content (%)
- ratio_D: D-lactide content (%)
- Mn0: Initial number-average MW (kg/mol)
- Mw0: Initial weight-average MW (kg/mol) - if available
- PDI0: Initial polydispersity
- Xc0: Initial crystallinity (%)
- Tg: Glass transition temperature (°C)
- form: Sample form (film, scaffold, fiber, etc.)
- condition: in_vitro or in_vivo
- medium: PBS, water, etc.
- pH: Medium pH
- T: Temperature (°C)
- times: Time points (days)
- Mn: Mn at each time point (kg/mol)
- Mw: Mw at each time point (kg/mol) - if available
- mass_remaining: Mass remaining (%) - if available
"""

const DEGRADATION_DATABASE = [
    # =========================================================================
    # KAIQUE HERGESEL PHD THESIS (2025) - PLDLA 70:30
    # =========================================================================
    (
        id = "Kaique_PLDLA",
        source = "Hergesel PhD Thesis 2025",
        material = "PLDLA",
        ratio_L = 70,
        ratio_D = 30,
        Mn0 = 51.285,
        Mw0 = 94.432,
        PDI0 = 1.84,
        Xc0 = 0.0,  # Amorphous
        Tg = 54.0,
        form = "3D_printed_scaffold",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 30.0, 60.0, 90.0],
        Mn = [51.285, 25.447, 18.313, 7.904],
        Mw = [94.432, 52.738, 35.861, 11.801],
        mass_remaining = [100.0, 100.0, 100.0, 100.0]  # No mass loss until 90d
    ),
    (
        id = "Kaique_TEC1",
        source = "Hergesel PhD Thesis 2025",
        material = "PLDLA/TEC1%",
        ratio_L = 70,
        ratio_D = 30,
        Mn0 = 44.998,
        Mw0 = 85.759,
        PDI0 = 1.90,
        Xc0 = 0.0,
        Tg = 49.0,  # TEC lowers Tg
        form = "3D_printed_scaffold",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 30.0, 60.0, 90.0],
        Mn = [44.998, 19.257, 11.749, 8.122],
        Mw = [85.759, 31.598, 22.409, 12.114],
        mass_remaining = [100.0, 100.0, 100.0, 100.0]
    ),
    (
        id = "Kaique_TEC2",
        source = "Hergesel PhD Thesis 2025",
        material = "PLDLA/TEC2%",
        ratio_L = 70,
        ratio_D = 30,
        Mn0 = 32.733,
        Mw0 = 68.364,
        PDI0 = 2.08,
        Xc0 = 0.0,
        Tg = 46.0,
        form = "3D_printed_scaffold",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 30.0, 60.0, 90.0],
        Mn = [32.733, 15.040, 12.616, 6.636],
        Mw = [68.364, 26.926, 19.417, 8.391],
        mass_remaining = [100.0, 100.0, 100.0, 100.0]
    ),

    # =========================================================================
    # PMC3359772 - Industrial PLA vs Laboratory PLLA (2012)
    # =========================================================================
    (
        id = "3051D_PBS",
        source = "PMC3359772 (Weir 2012)",
        material = "NatureWorks 3051D",
        ratio_L = 95.5,
        ratio_D = 4.5,
        Mn0 = 96.4,
        Mw0 = 203.5,  # Estimated from PDI
        PDI0 = 2.11,
        Xc0 = 6.8,
        Tg = 45.2,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 14.0, 28.0, 91.0],
        Mn = [96.4, 76.2, 23.1, 6.7],  # 79%, 24%, 7% remaining
        Mw = [203.5, 160.8, 48.8, 14.2],
        mass_remaining = [100.0, 100.0, 95.0, 88.0]
    ),
    (
        id = "3001D_PBS",
        source = "PMC3359772 (Weir 2012)",
        material = "NatureWorks 3001D",
        ratio_L = 98.4,
        ratio_D = 1.6,
        Mn0 = 89.3,
        Mw0 = 158.1,
        PDI0 = 1.77,
        Xc0 = 45.9,  # Higher crystallinity
        Tg = 47.9,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 14.0, 28.0, 91.0],
        Mn = [89.3, 80.4, 71.4, 53.6],  # Slower due to crystallinity
        Mw = [158.1, 142.3, 126.4, 94.9],
        mass_remaining = [100.0, 100.0, 98.0, 95.0]
    ),
    (
        id = "LabPLLA_PBS",
        source = "PMC3359772 (Weir 2012)",
        material = "Laboratory PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 85.6,
        Mw0 = 99.3,
        PDI0 = 1.16,  # Very narrow distribution
        Xc0 = 49.4,
        Tg = 47.5,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 14.0, 28.0, 91.0],
        Mn = [85.6, 81.3, 52.2, 34.2],  # Slow degradation
        Mw = [99.3, 94.3, 60.5, 39.7],
        mass_remaining = [100.0, 100.0, 99.0, 97.0]
    ),

    # =========================================================================
    # PMC5544909 - P(TMC-co-DLLA) 15:85 (2017)
    # =========================================================================
    (
        id = "PTMC_DLLA_invitro",
        source = "PMC5544909 (2017)",
        material = "P(TMC-co-DLLA)",
        ratio_L = 42.5,  # 85% DLLA = 42.5% L + 42.5% D
        ratio_D = 42.5,
        Mn0 = 89.567,
        Mw0 = 129.9,  # Estimated from PDI
        PDI0 = 1.45,
        Xc0 = 0.0,  # Amorphous
        Tg = 35.0,  # Estimated
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 7.0, 14.0, 28.0, 56.0, 84.0],
        Mn = [89.567, 59.457, 48.031, 39.862, 35.276, 34.010],
        Mw = [129.9, 152.2, 123.0, 102.0, 78.0, 68.4],
        mass_remaining = [100.0, 100.0, 100.0, 98.0, 93.3, 77.6]
    ),
    (
        id = "PTMC_DLLA_invivo",
        source = "PMC5544909 (2017)",
        material = "P(TMC-co-DLLA)",
        ratio_L = 42.5,
        ratio_D = 42.5,
        Mn0 = 89.567,
        Mw0 = 129.9,
        PDI0 = 1.45,
        Xc0 = 0.0,
        Tg = 35.0,
        form = "film",
        condition = :in_vivo,
        medium = "subcutaneous",
        pH = 7.35,
        T = 37.0,
        times = [0.0, 7.0, 14.0, 28.0],
        Mn = [89.567, 37.490, 22.864, 17.242],  # Much faster in vivo
        Mw = [129.9, 108.4, 63.6, 41.9],
        mass_remaining = [100.0, 98.5, 97.4, 94.4]
    ),

    # =========================================================================
    # TSUJI & IKADA STUDIES - Systematic PLLA (2000)
    # =========================================================================
    (
        id = "Tsuji_PLLA_150k",
        source = "Tsuji & Ikada Polymer 2000",
        material = "PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 150.0,
        Mw0 = 195.0,
        PDI0 = 1.30,
        Xc0 = 40.0,
        Tg = 60.0,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        # Long-term data (30 months)
        times = [0.0, 90.0, 180.0, 365.0, 548.0, 730.0, 912.0],
        Mn = [150.0, 120.0, 90.0, 60.0, 40.0, 25.0, 15.0],  # Estimated from curves
        Mw = [195.0, 156.0, 117.0, 78.0, 52.0, 32.5, 19.5],
        mass_remaining = [100.0, 100.0, 98.0, 95.0, 90.0, 80.0, 60.0]
    ),

    # =========================================================================
    # BERGSMA et al. - Long-term PLLA & PLA96 (1995)
    # =========================================================================
    (
        id = "Bergsma_PLLA",
        source = "Bergsma Biomaterials 1995",
        material = "PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 98.0,
        Mw0 = 127.4,
        PDI0 = 1.30,
        Xc0 = 64.5,  # High crystallinity
        Tg = 60.0,
        form = "plate",
        condition = :in_vivo,
        medium = "bone",
        pH = 7.4,
        T = 37.0,
        # Very long-term (years)
        times = [0.0, 112.0, 365.0, 730.0, 1095.0],
        Mn = [98.0, 7.4, 5.0, 3.0, 2.0],  # Slow mass loss but MW drops
        Mw = [127.4, 9.6, 6.5, 3.9, 2.6],
        mass_remaining = [100.0, 95.0, 80.0, 60.0, 40.0]
    ),
    (
        id = "Bergsma_PLA96",
        source = "Bergsma Biomaterials 1995",
        material = "PLA96",
        ratio_L = 96,
        ratio_D = 4,
        Mn0 = 85.0,
        Mw0 = 110.5,
        PDI0 = 1.30,
        Xc0 = 28.0,  # Lower crystallinity than pure PLLA
        Tg = 55.0,
        form = "plate",
        condition = :in_vivo,
        medium = "bone",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 56.0, 112.0, 365.0],
        Mn = [85.0, 20.0, 2.0, 1.0],  # Faster degradation
        Mw = [110.5, 26.0, 2.6, 1.3],
        mass_remaining = [100.0, 90.0, 70.0, 30.0]
    ),

    # =========================================================================
    # PDLLA (RACEMIC) - Fast degradation
    # =========================================================================
    (
        id = "PDLLA_50_50",
        source = "Middleton & Tipton 2000 / Grizzi 1995",
        material = "PDLLA",
        ratio_L = 50,
        ratio_D = 50,
        Mn0 = 45.0,
        Mw0 = 90.0,
        PDI0 = 2.0,
        Xc0 = 0.0,  # Completely amorphous
        Tg = 52.0,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 14.0, 28.0, 56.0, 84.0],
        Mn = [45.0, 30.0, 15.0, 5.0, 2.0],  # Very fast
        Mw = [90.0, 60.0, 30.0, 10.0, 4.0],
        mass_remaining = [100.0, 98.0, 90.0, 60.0, 30.0]
    ),

    # =========================================================================
    # PLLA SCAFFOLDS (Porous) - Faster than films
    # =========================================================================
    (
        id = "PLLA_scaffold_porous",
        source = "Pubmed 10885732 (2000)",
        material = "PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 100.0,
        Mw0 = 180.0,
        PDI0 = 1.80,
        Xc0 = 30.0,
        Tg = 58.0,
        form = "porous_scaffold_90%",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 56.0, 112.0, 168.0, 301.0],  # Half-life ~21 weeks
        Mn = [100.0, 70.0, 50.0, 35.0, 15.0],
        Mw = [180.0, 126.0, 90.0, 63.0, 27.0],
        mass_remaining = [100.0, 98.0, 95.0, 90.0, 75.0]
    ),

    # =========================================================================
    # 3D PRINTED PLLA (2024) - Accelerated testing
    # =========================================================================
    (
        id = "3DPrint_PLLA_37C",
        source = "Frontiers Bioeng 2024",
        material = "3D Printed PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 100.6,
        Mw0 = 180.1,
        PDI0 = 1.79,
        Xc0 = 0.0,  # Amorphous (3D printed)
        Tg = 55.0,
        form = "3D_printed_fiber",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 30.0, 60.0, 90.0, 120.0, 180.0],
        Mn = [100.6, 85.0, 65.0, 45.0, 30.0, 15.0],  # Estimated
        Mw = [180.1, 152.3, 116.5, 80.6, 53.8, 26.9],
        mass_remaining = [100.0, 100.0, 98.0, 95.0, 90.0, 80.0]
    ),
    (
        id = "3DPrint_PLLA_50C",
        source = "Frontiers Bioeng 2024",
        material = "3D Printed PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 100.6,
        Mw0 = 180.1,
        PDI0 = 1.79,
        Xc0 = 0.0,
        Tg = 55.0,
        form = "3D_printed_fiber",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 50.0,  # Accelerated
        times = [0.0, 30.0, 60.0, 100.0, 150.0],
        Mn = [100.6, 60.0, 30.0, 12.0, 4.0],  # Much faster at 50°C
        Mw = [180.1, 107.5, 53.8, 21.5, 7.2],
        mass_remaining = [100.0, 95.0, 80.0, 50.0, 20.0]
    ),

    # =========================================================================
    # IN VIVO SUBCUTANEOUS - BioEval data
    # =========================================================================
    (
        id = "BioEval_InVivo",
        source = "BioEval studies",
        material = "PLLA",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 99.0,
        Mw0 = 178.2,
        PDI0 = 1.80,
        Xc0 = 35.0,
        Tg = 58.0,
        form = "implant",
        condition = :in_vivo,
        medium = "subcutaneous",
        pH = 7.35,
        T = 37.0,
        times = [0.0, 28.0, 56.0, 84.0],
        Mn = [99.0, 92.0, 85.0, 78.0],  # Slower in vivo for PLLA
        Mw = [178.2, 165.6, 153.0, 140.4],
        mass_remaining = [100.0, 100.0, 100.0, 99.0]
    ),

    # =========================================================================
    # PLDLA 85:15 - Commercial grade
    # =========================================================================
    (
        id = "PLDLA_85_15",
        source = "Alexis 2005 review",
        material = "PLDLA 85:15",
        ratio_L = 85,
        ratio_D = 15,
        Mn0 = 70.0,
        Mw0 = 119.0,
        PDI0 = 1.70,
        Xc0 = 15.0,  # Low crystallinity
        Tg = 54.0,
        form = "film",
        condition = :in_vitro,
        medium = "PBS",
        pH = 7.4,
        T = 37.0,
        times = [0.0, 30.0, 60.0, 120.0],
        Mn = [70.0, 55.0, 40.0, 15.0],
        Mw = [119.0, 93.5, 68.0, 25.5],
        mass_remaining = [100.0, 100.0, 95.0, 80.0]
    ),

    # =========================================================================
    # PLLA High MW (300k) - Very slow
    # =========================================================================
    (
        id = "PLLA_HighMW",
        source = "Pubmed 3841816 (1988)",
        material = "PLLA High MW",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 300.0,
        Mw0 = 450.0,
        PDI0 = 1.50,
        Xc0 = 50.0,
        Tg = 60.0,
        form = "implant",
        condition = :in_vivo,
        medium = "subcutaneous",
        pH = 7.35,
        T = 37.0,
        times = [0.0, 168.0, 336.0],  # 24 and 48 weeks
        Mn = [300.0, 180.0, 90.0],  # Very slow
        Mw = [450.0, 270.0, 135.0],
        mass_remaining = [100.0, 98.0, 95.0]
    ),

    # =========================================================================
    # PLLA Low MW (60k) - Faster
    # =========================================================================
    (
        id = "PLLA_LowMW",
        source = "Pubmed 3841816 (1988)",
        material = "PLLA Low MW",
        ratio_L = 100,
        ratio_D = 0,
        Mn0 = 60.0,
        Mw0 = 90.0,
        PDI0 = 1.50,
        Xc0 = 45.0,
        Tg = 58.0,
        form = "implant",
        condition = :in_vivo,
        medium = "subcutaneous",
        pH = 7.35,
        T = 37.0,
        times = [0.0, 168.0, 336.0],
        Mn = [60.0, 20.0, 5.0],  # Faster than high MW
        Mw = [90.0, 30.0, 7.5],
        mass_remaining = [100.0, 90.0, 70.0]
    ),
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Get number of datasets in database.
"""
n_datasets() = length(DEGRADATION_DATABASE)

"""
Get unique L:DL ratios in database.
"""
function unique_ratios()
    ratios = [(d.ratio_L, d.ratio_D) for d in DEGRADATION_DATABASE]
    return unique(ratios)
end

"""
Get all data points for training.
Returns flat array of (ratio_L, Mn0, t, T, pH, Xc0, condition, Mn) tuples.
"""
function get_training_data()
    data = []
    for d in DEGRADATION_DATABASE
        for (i, t) in enumerate(d.times)
            push!(data, (
                ratio_L = d.ratio_L,
                Mn0 = d.Mn0,
                t = t,
                T = d.T,
                pH = d.pH,
                Xc0 = d.Xc0,
                Tg = d.Tg,
                condition = d.condition,
                form = d.form,
                Mn = d.Mn[i]
            ))
        end
    end
    return data
end

"""
Summary statistics of database.
"""
function database_summary()
    println("\n" * "="^70)
    println("  DEGRADATION DATABASE SUMMARY")
    println("="^70)

    println("\n  Total datasets: $(n_datasets())")

    # By ratio
    ratios = unique_ratios()
    println("\n  L:DL Ratios covered:")
    for (L, D) in sort(ratios, by=x->x[1], rev=true)
        n = count(d -> d.ratio_L == L && d.ratio_D == D, DEGRADATION_DATABASE)
        println("    $L:$D - $n datasets")
    end

    # By condition
    n_vitro = count(d -> d.condition == :in_vitro, DEGRADATION_DATABASE)
    n_vivo = count(d -> d.condition == :in_vivo, DEGRADATION_DATABASE)
    println("\n  Conditions:")
    println("    In vitro: $n_vitro")
    println("    In vivo: $n_vivo")

    # By form
    forms = unique([d.form for d in DEGRADATION_DATABASE])
    println("\n  Sample forms: ", join(forms, ", "))

    # Mn0 range
    Mn0_min = minimum(d.Mn0 for d in DEGRADATION_DATABASE)
    Mn0_max = maximum(d.Mn0 for d in DEGRADATION_DATABASE)
    println("\n  Mn0 range: $Mn0_min - $Mn0_max kg/mol")

    # Total data points
    n_points = sum(length(d.times) for d in DEGRADATION_DATABASE)
    println("\n  Total data points: $n_points")

    println("\n" * "="^70)
end

# Run summary if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    database_summary()
end
