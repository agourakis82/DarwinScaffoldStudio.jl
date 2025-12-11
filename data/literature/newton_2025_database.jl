"""
newton_2025_database.jl

Dataset from Newton 2025: "Revealing chain scission modes in variable polymer degradation kinetics"
Cheng et al., Newton 1, 100168 (2025)
DOI: 10.1016/j.newton.2025.100168

41 polymers with degradation kinetics parameters extracted from meta-analysis.

The paper provides:
- Initial MW (kDa)
- Degradation timescale 1/k (days)
- R² for chain-end and random scission models
- Catalyst type and solubility

We reconstruct time series using the models:
- Chain-end scission: MW(t)/MW(0) = 1/(1 + kt)
- Random scission: MW(t)/MW(0) = exp(-kt)
"""

# ============================================================================
# DATA STRUCTURES
# ============================================================================

struct Newton2025Polymer
    id::Int
    name::String
    full_name::String
    r2_random::Float64
    r2_chain_end::Float64
    initial_mw_kda::Float64
    degradation_timescale_days::Float64  # 1/k
    catalyst::String
    solubility::Float64  # g per g solvent
    reference::String
    scission_mode::Symbol  # :chain_end or :random (based on higher R²)
end

# ============================================================================
# NEWTON 2025 DATASET - 41 POLYMERS
# ============================================================================

const NEWTON_2025_POLYMERS = [
    # Soluble polymers (chain-end scission dominant)
    Newton2025Polymer(1, "Chitosan", "Chitosan", 0.8818, 0.9974, 213.00, 0.006, "H2O2", 0.0050, "S3", :chain_end),
    Newton2025Polymer(2, "Dextran", "Dextran", 0.9040, 0.9863, 875.30, 0.001, "Dextranase", 0.0100, "S4", :chain_end),
    Newton2025Polymer(3, "PLA-co-PEG", "Poly(ethylene glycol)-co-polylactide", 0.8257, 0.9546, 32.64, 2.75, "H2O", 0.0020, "S5", :chain_end),
    Newton2025Polymer(4, "Cellulose", "Cellulose", 0.7671, 0.8463, 386.62, 0.01, "H+/H2O", 0.0010, "S9", :chain_end),
    Newton2025Polymer(5, "Alginate-hydrogel", "Alginate", 0.9651, 0.9714, 115.66, 16.16, "H2O", 0.0300, "S10", :chain_end),
    Newton2025Polymer(6, "HA", "Hyaluronic acid", 0.9875, 0.9553, 1876.40, 0.68, "Hyaluronidase", 0.0050, "S11", :random),
    Newton2025Polymer(7, "HA-w-cell", "Hyaluronic acid (with cell)", 0.9905, 0.9173, 1882.02, 1.10, "Hyaluronidase", 0.0050, "S11", :random),
    Newton2025Polymer(8, "PCPP", "Poly[di(carboxylatophenoxy)phosphazene]", 0.8416, 0.9494, 820.00, 21.93, "H2O", 0.0040, "S12", :chain_end),
    Newton2025Polymer(9, "PCL-organic", "Polycaprolactone (organic)", 0.9042, 0.9648, 11.67, 0.95, "H2O", 0.0020, "S13", :chain_end),
    Newton2025Polymer(10, "PDHF", "Poly(2,5-dihydrofuran)", 0.6967, 0.8692, 44.52, 0.01, "Grubbs", 0.0052, "S14", :chain_end),
    Newton2025Polymer(11, "Alginate", "Alginate", 0.7652, 0.7922, 345.78, 0.02, "H2O2", 0.0100, "S15", :chain_end),
    Newton2025Polymer(12, "Alginate-pH9.2", "Alginate (pH 9.2)", 0.7805, 0.8975, 114.35, 5.69, "H2O", 0.0200, "S16", :chain_end),
    Newton2025Polymer(13, "Alginate-pH7.4", "Alginate (pH 7.4)", 0.8485, 0.9502, 149.58, 8.61, "H2O", 0.0200, "S16", :chain_end),
    Newton2025Polymer(14, "Alginate-pH4.5", "Alginate (pH 4.5)", 0.8945, 0.9504, 220.71, 31.06, "H2O", 0.0200, "S16", :chain_end),
    Newton2025Polymer(15, "Citrus-pectin", "Citrus pectin", 0.9779, 0.9509, 451.48, 0.01, "H2O2/Fe3+", 0.0050, "S17", :random),
    Newton2025Polymer(16, "P-SA-co-RA", "Poly(sebacic acid-co-ricinoleic acid)", 0.8228, 0.8651, 3.49, 22.32, "H2O", 0.0040, "S18", :chain_end),
    Newton2025Polymer(17, "Guar-GM", "Guar galactomannan", 0.9310, 0.9911, 1790.00, 0.20, "Endo-mannanase", 0.0100, "S19", :chain_end),
    Newton2025Polymer(18, "PVA-98", "Polyvinyl alcohol 98% hydrolyzed", 0.7090, 0.7976, 36.17, 0.33, "Microbe", 0.0042, "S20", :chain_end),
    Newton2025Polymer(19, "PVA-72", "Polyvinyl alcohol 72% hydrolyzed", 0.7752, 0.8347, 13.52, 4.46, "Microbe", 0.0042, "S20", :chain_end),
    Newton2025Polymer(20, "PGAA-M4", "Poly(glycoamidoamine) D-mannaramide", 0.8350, 0.9137, 4.90, 0.97, "H2O", 0.0245, "S21", :chain_end),
    Newton2025Polymer(21, "PGAA-T4", "Poly(glycoamidoamine) L-tartaramide", 0.8060, 0.8842, 5.60, 0.65, "H2O", 0.0280, "S21", :chain_end),

    # Insoluble polymers (random scission dominant)
    Newton2025Polymer(22, "PLA", "Polylactic acid", 0.9983, 0.8630, 11.67, 7.87, "H2O", 1e-6, "S6", :random),
    Newton2025Polymer(23, "PLGA", "Poly(lactic-co-glycolic acid)", 0.9940, 0.9337, 4.39, 22.47, "H2O", 1e-6, "S7", :random),
    Newton2025Polymer(24, "PET", "Polyethylene terephthalate", 0.9915, 0.9834, 26.74, 13.76, "H2O", 1e-6, "S8", :random),
    Newton2025Polymer(25, "PDLA", "Poly(D,L-lactide)", 0.9307, 0.9621, 1156.54, 25.77, "H2O", 1e-6, "S22", :chain_end),
    Newton2025Polymer(26, "PE", "Polyethylene", 0.9890, 0.8375, 100.0, 7.17, "Microbe", 1e-6, "S23", :random),  # MW unknown, estimated
    Newton2025Polymer(27, "PP", "Polypropylene", 0.9923, 0.9129, 100.0, 7.67, "Microbe", 1e-6, "S23", :random),  # MW unknown, estimated
    Newton2025Polymer(28, "PDO", "Polydioxanone", 0.9738, 0.9971, 50.0, 9.06, "OH-/H2O", 1e-6, "S24", :chain_end),  # MW unknown, estimated
    Newton2025Polymer(29, "P-LLA-co-GA", "Poly(L-lactide-co-glycolide)", 0.9687, 0.9279, 50.0, 44.44, "OH-/H2O", 1e-6, "S24", :random),  # MW unknown
    Newton2025Polymer(30, "PLA50-thick", "Polylactic acid (thick)", 0.9774, 0.8579, 43.00, 58.02, "H2O", 1e-6, "S25", :random),
    Newton2025Polymer(31, "PLA50-thin", "Polylactic acid (thin)", 0.9966, 0.9409, 67.00, 119.25, "H2O", 1e-6, "S25", :random),
    Newton2025Polymer(32, "PLLA-co-PDLLA", "Poly(L-lactide)-co-poly(D,L-lactide)", 0.9833, 0.7453, 95.12, 400.00, "H2O", 1e-6, "S26", :random),
    Newton2025Polymer(33, "PLA-Con-85C", "Polylactic acid (85°C)", 0.8919, 0.9451, 106.10, 0.87, "H2O", 1e-6, "S27", :chain_end),
    Newton2025Polymer(34, "PLA-Con-40C", "Polylactic acid (40°C)", 0.9934, 0.9356, 106.30, 105.26, "H2O", 1e-6, "S27", :random),
    Newton2025Polymer(35, "PLA-Cex-85C", "Polylactic acid chain extender (85°C)", 0.9757, 0.9497, 191.60, 1.40, "H2O", 1e-6, "S27", :random),
    Newton2025Polymer(36, "PLA-Cex-40C", "Polylactic acid chain extender (40°C)", 0.9932, 0.8937, 191.60, 147.06, "H2O", 1e-6, "S27", :random),
    Newton2025Polymer(37, "PCL", "Polycaprolactone", 0.9962, 0.9854, 31.47, 434.78, "H2O", 1e-6, "S28", :random),
    Newton2025Polymer(38, "P4MC", "Poly(4-methylcaprolactone)", 0.9882, 0.9968, 21.84, 163.93, "H2O", 1e-6, "S28", :chain_end),
    Newton2025Polymer(39, "P4MC-BA", "Poly(4-methylcaprolactone) benzyl alcohol", 0.9825, 0.9787, 14.61, 144.93, "H2O", 1e-6, "S28", :random),
    Newton2025Polymer(40, "PBAT", "Polybutylene adipate-co-terephthalate", 0.9833, 0.9596, 88.39, 473.83, "Microbe", 1e-6, "S29", :random),
    Newton2025Polymer(41, "P-DTD-co-OD", "Poly(desaminotyrosyl-tyrosine dodecyl dodecanedioate)", 0.9944, 0.8889, 59.00, 57.66, "H2O", 1e-6, "S30", :random),
]

# ============================================================================
# FUNCTIONS TO GENERATE TIME SERIES
# ============================================================================

"""
Generate time series using chain-end scission model.
MW(t)/MW(0) = 1/(1 + kt)
"""
function generate_chain_end_series(polymer::Newton2025Polymer; n_points::Int=25)
    k = 1.0 / polymer.degradation_timescale_days
    MW0 = polymer.initial_mw_kda

    # Generate time points from 0 to ~90% degradation
    t_max = min(polymer.degradation_timescale_days * 10, 365.0)  # Cap at 1 year
    times = range(0, t_max, length=n_points)

    MW_values = [MW0 / (1 + k * t) for t in times]

    return (times=collect(times), MW=MW_values, model=:chain_end)
end

"""
Generate time series using random scission model.
MW(t)/MW(0) = exp(-kt)
"""
function generate_random_series(polymer::Newton2025Polymer; n_points::Int=25)
    k = 1.0 / polymer.degradation_timescale_days
    MW0 = polymer.initial_mw_kda

    # Generate time points from 0 to ~90% degradation
    t_max = min(polymer.degradation_timescale_days * 5, 365.0)  # Cap at 1 year
    times = range(0, t_max, length=n_points)

    MW_values = [MW0 * exp(-k * t) for t in times]

    return (times=collect(times), MW=MW_values, model=:random)
end

"""
Generate time series using the best-fit model for each polymer.
"""
function generate_best_fit_series(polymer::Newton2025Polymer; n_points::Int=25)
    if polymer.scission_mode == :chain_end
        return generate_chain_end_series(polymer; n_points=n_points)
    else
        return generate_random_series(polymer; n_points=n_points)
    end
end

"""
Get all time series for validation.
"""
function get_all_time_series(; n_points::Int=25)
    results = Dict{String, NamedTuple}()

    for polymer in NEWTON_2025_POLYMERS
        series = generate_best_fit_series(polymer; n_points=n_points)
        results[polymer.name] = (
            polymer=polymer,
            times=series.times,
            MW=series.MW,
            model=series.model
        )
    end

    return results
end

"""
Filter polymers by type.
"""
function get_pla_family()
    return filter(p -> occursin("PLA", p.name) || occursin("PLLA", p.name) ||
                       occursin("PDLA", p.name) || occursin("Polylactic", p.full_name),
                  NEWTON_2025_POLYMERS)
end

function get_pcl_family()
    return filter(p -> occursin("PCL", p.name) || occursin("caprolactone", lowercase(p.full_name)),
                  NEWTON_2025_POLYMERS)
end

function get_soluble_polymers()
    return filter(p -> p.solubility > 1e-4, NEWTON_2025_POLYMERS)
end

function get_insoluble_polymers()
    return filter(p -> p.solubility <= 1e-4, NEWTON_2025_POLYMERS)
end

# ============================================================================
# STATISTICS
# ============================================================================

function newton_database_stats()
    n_total = length(NEWTON_2025_POLYMERS)
    n_chain_end = count(p -> p.scission_mode == :chain_end, NEWTON_2025_POLYMERS)
    n_random = count(p -> p.scission_mode == :random, NEWTON_2025_POLYMERS)
    n_soluble = length(get_soluble_polymers())
    n_insoluble = length(get_insoluble_polymers())
    n_pla = length(get_pla_family())
    n_pcl = length(get_pcl_family())

    return Dict(
        "total_polymers" => n_total,
        "chain_end_scission" => n_chain_end,
        "random_scission" => n_random,
        "soluble" => n_soluble,
        "insoluble" => n_insoluble,
        "pla_family" => n_pla,
        "pcl_family" => n_pcl
    )
end

# ============================================================================
# PRINT SUMMARY
# ============================================================================

println("="^70)
println("  NEWTON 2025 DATABASE - Chain Scission Modes")
println("  Cheng et al., Newton 1, 100168 (2025)")
println("="^70)

stats = newton_database_stats()
println("  Total polymers: $(stats["total_polymers"])")
println("  - Chain-end scission: $(stats["chain_end_scission"])")
println("  - Random scission: $(stats["random_scission"])")
println("  - Soluble: $(stats["soluble"])")
println("  - Insoluble: $(stats["insoluble"])")
println("  - PLA family: $(stats["pla_family"])")
println("  - PCL family: $(stats["pcl_family"])")
println("="^70)
