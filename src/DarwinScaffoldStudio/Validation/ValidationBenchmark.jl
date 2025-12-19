"""
Validation Benchmark Module

Compare DarwinScaffoldStudio metrics against gold-standard tools:
- ImageJ/BoneJ
- CTAn (Bruker/Skyscan)
- Avizo
- Manual measurements (PoreScript)

This module is CRITICAL for dissertation defense - provides scientific validation.

References:
- Doube et al. (2010) "BoneJ: Free and extensible bone image analysis in ImageJ"
- Bruker CTAn User Manual (2019)
- Murphy et al. (2010) for pore size validation (100-200 μm optimal)
"""

module ValidationBenchmark

using Statistics
using Printf
using Dates

# Read version from Project.toml (avoid cross-module import issues)
const _VERSION = let
    project_file = joinpath(dirname(dirname(dirname(@__DIR__))), "Project.toml")
    version = "0.0.0"
    if isfile(project_file)
        for line in eachline(project_file)
            m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
            if m !== nothing
                version = m.captures[1]
                break
            end
        end
    end
    version
end

export ValidationResult, BenchmarkSuite
export run_validation, compare_with_reference, generate_validation_report
export validate_porosity, validate_pore_size, validate_interconnectivity

# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
    ValidationResult

Result of a single metric validation.

# Fields
- `metric_name::String`: Name of metric being validated
- `darwin_value::Float64`: Value from DarwinScaffoldStudio
- `reference_value::Float64`: Value from reference tool/measurement
- `reference_source::String`: Source of reference ("ImageJ", "CTAn", "Manual", etc.)
- `absolute_error::Float64`: |darwin - reference|
- `relative_error_percent::Float64`: |darwin - reference| / reference * 100
- `passed::Bool`: Whether within acceptable tolerance
- `tolerance_percent::Float64`: Acceptable error threshold
- `notes::String`: Additional notes
"""
struct ValidationResult
    metric_name::String
    darwin_value::Float64
    reference_value::Float64
    reference_source::String
    absolute_error::Float64
    relative_error_percent::Float64
    passed::Bool
    tolerance_percent::Float64
    notes::String
end

"""
    BenchmarkSuite

Complete benchmark results for a dataset.

# Fields
- `dataset_name::String`: Name/ID of test dataset
- `dataset_source::String`: Where dataset came from
- `timestamp::DateTime`: When benchmark was run
- `darwin_version::String`: Version of DarwinScaffoldStudio
- `results::Vector{ValidationResult}`: Individual metric results
- `overall_passed::Bool`: All metrics within tolerance
- `summary_stats::Dict`: Summary statistics
"""
struct BenchmarkSuite
    dataset_name::String
    dataset_source::String
    timestamp::DateTime
    darwin_version::String
    results::Vector{ValidationResult}
    overall_passed::Bool
    summary_stats::Dict{String,Any}
end

# ============================================================================
# TOLERANCE DEFINITIONS (based on literature)
# ============================================================================

# Acceptable relative errors for each metric
const METRIC_TOLERANCES = Dict{String,Float64}(
    "porosity" => 5.0,              # 5% relative error acceptable
    "mean_pore_size_um" => 10.0,    # 10% (validated at 1.4% vs PoreScript)
    "interconnectivity" => 10.0,    # 10%
    "tortuosity" => 15.0,           # 15% (harder to measure precisely)
    "specific_surface_area" => 15.0,
    "elastic_modulus" => 20.0,      # Model-based, higher tolerance
    "permeability" => 25.0          # Highly variable
)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

"""
    validate_porosity(darwin_value::Float64, reference_value::Float64;
                      source::String="Manual") -> ValidationResult

Validate porosity measurement against reference.

# Notes
Porosity = 1 - (solid volume / total volume)
Gold standard: gravimetric measurement or CTAn VOI analysis
"""
function validate_porosity(
    darwin_value::Float64,
    reference_value::Float64;
    source::String="Manual"
)::ValidationResult
    tolerance = METRIC_TOLERANCES["porosity"]
    abs_error = abs(darwin_value - reference_value)
    rel_error = reference_value > 0 ? (abs_error / reference_value) * 100 : 0.0
    passed = rel_error <= tolerance

    notes = if passed
        "Within $(tolerance)% tolerance"
    else
        "FAILED: $(round(rel_error, digits=2))% > $(tolerance)% tolerance"
    end

    return ValidationResult(
        "porosity",
        darwin_value,
        reference_value,
        source,
        abs_error,
        rel_error,
        passed,
        tolerance,
        notes
    )
end

"""
    validate_pore_size(darwin_value::Float64, reference_value::Float64;
                       source::String="PoreScript",
                       method::String="Feret") -> ValidationResult

Validate mean pore size against reference.

# Notes
Feret diameter method validated against PoreScript manual measurements.
Previous validation showed 1.4% relative error.
"""
function validate_pore_size(
    darwin_value::Float64,
    reference_value::Float64;
    source::String="PoreScript",
    method::String="Feret"
)::ValidationResult
    tolerance = METRIC_TOLERANCES["mean_pore_size_um"]
    abs_error = abs(darwin_value - reference_value)
    rel_error = reference_value > 0 ? (abs_error / reference_value) * 100 : 0.0
    passed = rel_error <= tolerance

    notes = "Method: $method. "
    notes *= if passed
        "Within $(tolerance)% tolerance. "
    else
        "FAILED: $(round(rel_error, digits=2))% > $(tolerance)% tolerance. "
    end

    # Add context about biological relevance
    if 100 <= darwin_value <= 200
        notes *= "Value in optimal range for bone tissue (Murphy 2010)."
    elseif darwin_value < 100
        notes *= "Below optimal range - may limit cell infiltration."
    else
        notes *= "Above optimal range for bone - may reduce mechanical strength."
    end

    return ValidationResult(
        "mean_pore_size_um",
        darwin_value,
        reference_value,
        source,
        abs_error,
        rel_error,
        passed,
        tolerance,
        notes
    )
end

"""
    validate_interconnectivity(darwin_value::Float64, reference_value::Float64;
                              source::String="CTAn") -> ValidationResult

Validate interconnectivity (open porosity) against reference.

# Notes
Interconnectivity = largest connected pore volume / total pore volume
CTAn reports this as "Open porosity" or through 3D analysis
"""
function validate_interconnectivity(
    darwin_value::Float64,
    reference_value::Float64;
    source::String="CTAn"
)::ValidationResult
    tolerance = METRIC_TOLERANCES["interconnectivity"]
    abs_error = abs(darwin_value - reference_value)
    rel_error = reference_value > 0 ? (abs_error / reference_value) * 100 : 0.0
    passed = rel_error <= tolerance

    notes = if passed
        "Within $(tolerance)% tolerance. "
    else
        "FAILED: $(round(rel_error, digits=2))% > $(tolerance)% tolerance. "
    end

    # Biological context
    if darwin_value >= 0.90
        notes *= "Meets Karageorgiou 2005 criterion (≥90%)."
    else
        notes *= "Below recommended 90% threshold."
    end

    return ValidationResult(
        "interconnectivity",
        darwin_value,
        reference_value,
        source,
        abs_error,
        rel_error,
        passed,
        tolerance,
        notes
    )
end

"""
    validate_tortuosity(darwin_value::Float64, reference_value::Float64;
                        source::String="Simulation") -> ValidationResult

Validate tortuosity measurement.

# Notes
Tortuosity = actual path length / straight-line distance
Darwin uses Gibson-Ashby approximation: τ ≈ 1 + 0.5 * relative_density
More accurate methods use random walk simulation or diffusion MRI
"""
function validate_tortuosity(
    darwin_value::Float64,
    reference_value::Float64;
    source::String="Simulation"
)::ValidationResult
    tolerance = METRIC_TOLERANCES["tortuosity"]
    abs_error = abs(darwin_value - reference_value)
    rel_error = reference_value > 0 ? (abs_error / reference_value) * 100 : 0.0
    passed = rel_error <= tolerance

    notes = "Gibson-Ashby approximation. "
    notes *= if passed
        "Within $(tolerance)% tolerance."
    else
        "FAILED: $(round(rel_error, digits=2))% > $(tolerance)% tolerance."
    end

    return ValidationResult(
        "tortuosity",
        darwin_value,
        reference_value,
        source,
        abs_error,
        rel_error,
        passed,
        tolerance,
        notes
    )
end

"""
    validate_surface_area(darwin_value::Float64, reference_value::Float64;
                          source::String="BoneJ") -> ValidationResult

Validate specific surface area.
"""
function validate_surface_area(
    darwin_value::Float64,
    reference_value::Float64;
    source::String="BoneJ"
)::ValidationResult
    tolerance = METRIC_TOLERANCES["specific_surface_area"]
    abs_error = abs(darwin_value - reference_value)
    rel_error = reference_value > 0 ? (abs_error / reference_value) * 100 : 0.0
    passed = rel_error <= tolerance

    notes = if passed
        "Within $(tolerance)% tolerance."
    else
        "FAILED: $(round(rel_error, digits=2))% > $(tolerance)% tolerance."
    end

    return ValidationResult(
        "specific_surface_area",
        darwin_value,
        reference_value,
        source,
        abs_error,
        rel_error,
        passed,
        tolerance,
        notes
    )
end

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

"""
    run_validation(darwin_metrics::Dict{String,Float64},
                   reference_metrics::Dict{String,Float64};
                   dataset_name::String="Unknown",
                   dataset_source::String="Unknown",
                   reference_source::String="Manual") -> BenchmarkSuite

Run full validation benchmark comparing Darwin metrics to reference.

# Arguments
- `darwin_metrics`: Dict with keys like "porosity", "mean_pore_size_um", etc.
- `reference_metrics`: Dict with same keys from reference measurement
- `dataset_name`: Identifier for the test dataset
- `dataset_source`: Where the dataset came from
- `reference_source`: Tool/method used for reference

# Returns
- `BenchmarkSuite` with all validation results
"""
function run_validation(
    darwin_metrics::Dict{String,Float64},
    reference_metrics::Dict{String,Float64};
    dataset_name::String="Unknown",
    dataset_source::String="Unknown",
    reference_source::String="Manual"
)::BenchmarkSuite
    results = ValidationResult[]

    # Validate each metric that has both Darwin and reference values
    for (metric, darwin_val) in darwin_metrics
        if haskey(reference_metrics, metric)
            ref_val = reference_metrics[metric]

            result = if metric == "porosity"
                validate_porosity(darwin_val, ref_val, source=reference_source)
            elseif metric == "mean_pore_size_um"
                validate_pore_size(darwin_val, ref_val, source=reference_source)
            elseif metric == "interconnectivity"
                validate_interconnectivity(darwin_val, ref_val, source=reference_source)
            elseif metric == "tortuosity"
                validate_tortuosity(darwin_val, ref_val, source=reference_source)
            elseif metric == "specific_surface_area"
                validate_surface_area(darwin_val, ref_val, source=reference_source)
            else
                # Generic validation
                tolerance = get(METRIC_TOLERANCES, metric, 20.0)
                abs_error = abs(darwin_val - ref_val)
                rel_error = ref_val > 0 ? (abs_error / ref_val) * 100 : 0.0
                ValidationResult(
                    metric, darwin_val, ref_val, reference_source,
                    abs_error, rel_error, rel_error <= tolerance,
                    tolerance, ""
                )
            end

            push!(results, result)
        end
    end

    # Summary statistics
    n_passed = count(r -> r.passed, results)
    n_total = length(results)
    overall_passed = n_passed == n_total

    rel_errors = [r.relative_error_percent for r in results]

    summary = Dict{String,Any}(
        "n_metrics" => n_total,
        "n_passed" => n_passed,
        "pass_rate" => n_total > 0 ? n_passed / n_total * 100 : 0.0,
        "mean_relative_error" => isempty(rel_errors) ? 0.0 : mean(rel_errors),
        "max_relative_error" => isempty(rel_errors) ? 0.0 : maximum(rel_errors),
        "min_relative_error" => isempty(rel_errors) ? 0.0 : minimum(rel_errors)
    )

    return BenchmarkSuite(
        dataset_name,
        dataset_source,
        now(),
        _VERSION,
        results,
        overall_passed,
        summary
    )
end

"""
    compare_with_reference(darwin_metrics, imagej_path::String) -> BenchmarkSuite

Compare Darwin metrics with ImageJ/BoneJ results file.

# Notes
Expects ImageJ results in CSV format with columns:
Label, Porosity, Surface Area, etc.
"""
function compare_with_reference(
    darwin_metrics::Dict{String,Float64},
    reference_file::String;
    format::String="csv"
)::BenchmarkSuite
    # Parse reference file based on format
    reference_metrics = if format == "csv"
        parse_imagej_csv(reference_file)
    elseif format == "ctan"
        parse_ctan_results(reference_file)
    else
        error("Unknown reference format: $format")
    end

    return run_validation(
        darwin_metrics,
        reference_metrics,
        dataset_name=basename(reference_file),
        dataset_source=reference_file,
        reference_source=format == "csv" ? "ImageJ" : "CTAn"
    )
end

"""
Parse ImageJ/BoneJ CSV results file.
"""
function parse_imagej_csv(filepath::String)::Dict{String,Float64}
    metrics = Dict{String,Float64}()

    # Map ImageJ column names to Darwin metric names
    column_map = Dict(
        "BV/TV" => "porosity",  # Note: BV/TV = 1 - porosity
        "Tb.Th" => "mean_pore_size_um",  # Trabecular thickness
        "Tb.Sp" => "mean_pore_size_um",  # Trabecular spacing (for pores)
        "Conn.D" => "interconnectivity",
        "BS/TV" => "specific_surface_area"
    )

    if !isfile(filepath)
        @warn "Reference file not found: $filepath"
        return metrics
    end

    lines = readlines(filepath)
    if isempty(lines)
        return metrics
    end

    # Parse header
    header = split(lines[1], ",")

    # Parse data (assume first data row)
    if length(lines) > 1
        values = split(lines[2], ",")

        for (i, col) in enumerate(header)
            col_clean = strip(col)
            if haskey(column_map, col_clean) && i <= length(values)
                val = tryparse(Float64, strip(values[i]))
                if !isnothing(val)
                    metric_name = column_map[col_clean]

                    # Special handling for BV/TV -> porosity
                    if col_clean == "BV/TV"
                        val = 1.0 - val
                    end

                    metrics[metric_name] = val
                end
            end
        end
    end

    return metrics
end

"""
Parse Bruker CTAn results file.
"""
function parse_ctan_results(filepath::String)::Dict{String,Float64}
    metrics = Dict{String,Float64}()

    # CTAn typically outputs in a specific format
    # This is a simplified parser

    if !isfile(filepath)
        @warn "Reference file not found: $filepath"
        return metrics
    end

    for line in readlines(filepath)
        # Look for key metrics
        if occursin("Percent object volume", line)
            val = extract_value_from_line(line)
            if !isnothing(val)
                metrics["porosity"] = 1.0 - val / 100  # Convert to porosity
            end
        elseif occursin("Object surface / volume ratio", line)
            val = extract_value_from_line(line)
            if !isnothing(val)
                metrics["specific_surface_area"] = val
            end
        elseif occursin("Structure thickness", line)
            val = extract_value_from_line(line)
            if !isnothing(val)
                metrics["mean_pore_size_um"] = val * 1000  # mm to μm
            end
        end
    end

    return metrics
end

"""
Extract numeric value from a line like "Metric name: 0.123"
"""
function extract_value_from_line(line::String)::Union{Nothing,Float64}
    # Try to find a number in the line
    m = match(r"[\d.]+", split(line, ":")[end])
    if !isnothing(m)
        return tryparse(Float64, m.match)
    end
    return nothing
end

# ============================================================================
# REPORT GENERATION
# ============================================================================

"""
    generate_validation_report(suite::BenchmarkSuite;
                              format::String="markdown") -> String

Generate validation report for dissertation/publication.

# Arguments
- `suite`: BenchmarkSuite from run_validation
- `format`: "markdown", "latex", or "html"

# Returns
- Formatted report string
"""
function generate_validation_report(
    suite::BenchmarkSuite;
    format::String="markdown"
)::String
    if format == "markdown"
        return generate_markdown_report(suite)
    elseif format == "latex"
        return generate_latex_report(suite)
    else
        return generate_markdown_report(suite)  # Default
    end
end

"""
Generate Markdown validation report.
"""
function generate_markdown_report(suite::BenchmarkSuite)::String
    io = IOBuffer()

    println(io, "# Validation Benchmark Report")
    println(io, "")
    println(io, "**Generated:** $(Dates.format(suite.timestamp, "yyyy-mm-dd HH:MM:SS"))")
    println(io, "**DarwinScaffoldStudio Version:** $(suite.darwin_version)")
    println(io, "**Dataset:** $(suite.dataset_name)")
    println(io, "**Dataset Source:** $(suite.dataset_source)")
    println(io, "")

    # Summary
    println(io, "## Summary")
    println(io, "")
    status = suite.overall_passed ? "**PASSED**" : "**FAILED**"
    println(io, "- Overall Status: $status")
    println(io, "- Metrics Validated: $(suite.summary_stats["n_metrics"])")
    println(io, "- Metrics Passed: $(suite.summary_stats["n_passed"])")
    println(io, "- Pass Rate: $(@sprintf("%.1f", suite.summary_stats["pass_rate"]))%")
    println(io, "- Mean Relative Error: $(@sprintf("%.2f", suite.summary_stats["mean_relative_error"]))%")
    println(io, "- Max Relative Error: $(@sprintf("%.2f", suite.summary_stats["max_relative_error"]))%")
    println(io, "")

    # Results table
    println(io, "## Detailed Results")
    println(io, "")
    println(io, "| Metric | Darwin | Reference | Source | Rel. Error | Status |")
    println(io, "|--------|--------|-----------|--------|------------|--------|")

    for r in suite.results
        status = r.passed ? "✓" : "✗"
        println(io, "| $(r.metric_name) | $(@sprintf("%.4f", r.darwin_value)) | $(@sprintf("%.4f", r.reference_value)) | $(r.reference_source) | $(@sprintf("%.2f", r.relative_error_percent))% | $status |")
    end
    println(io, "")

    # Notes
    println(io, "## Notes")
    println(io, "")
    for r in suite.results
        if !isempty(r.notes)
            println(io, "- **$(r.metric_name)**: $(r.notes)")
        end
    end
    println(io, "")

    # References
    println(io, "## References")
    println(io, "")
    println(io, "1. Murphy CM, O'Brien FJ (2010). Understanding the effect of mean pore size on cell activity in collagen-glycosaminoglycan scaffolds. Cell Adh Migr 4(3):377-381.")
    println(io, "2. Karageorgiou V, Kaplan D (2005). Porosity of 3D biomaterial scaffolds and osteogenesis. Biomaterials 26(27):5474-5491.")
    println(io, "3. Doube M et al. (2010). BoneJ: Free and extensible bone image analysis in ImageJ. Bone 47(6):1076-1079.")
    println(io, "")

    return String(take!(io))
end

"""
Generate LaTeX validation report (for dissertation).
"""
function generate_latex_report(suite::BenchmarkSuite)::String
    io = IOBuffer()

    println(io, "\\subsection{Validation Results}")
    println(io, "")
    println(io, "The DarwinScaffoldStudio metrics were validated against reference measurements.")
    println(io, "Table~\\ref{tab:validation} summarizes the results.")
    println(io, "")

    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{Validation of scaffold metrics against reference measurements.}")
    println(io, "\\label{tab:validation}")
    println(io, "\\begin{tabular}{lcccc}")
    println(io, "\\toprule")
    println(io, "Metric & Darwin & Reference & Rel. Error (\\%) & Status \\\\")
    println(io, "\\midrule")

    for r in suite.results
        status = r.passed ? "Pass" : "Fail"
        println(io, "$(r.metric_name) & $(@sprintf("%.4f", r.darwin_value)) & $(@sprintf("%.4f", r.reference_value)) & $(@sprintf("%.2f", r.relative_error_percent)) & $status \\\\")
    end

    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    println(io, "")

    # Summary paragraph
    pass_rate = suite.summary_stats["pass_rate"]
    mean_error = suite.summary_stats["mean_relative_error"]

    println(io, "The validation achieved a $(@sprintf("%.1f", pass_rate))\\% pass rate with a mean relative error of $(@sprintf("%.2f", mean_error))\\%.")

    if suite.overall_passed
        println(io, "All metrics were within acceptable tolerances, demonstrating the accuracy of the computational pipeline.")
    else
        println(io, "Some metrics exceeded tolerance thresholds, indicating areas for potential improvement.")
    end
    println(io, "")

    return String(take!(io))
end

"""
    save_validation_report(suite::BenchmarkSuite, filepath::String;
                          format::String="markdown")

Save validation report to file.
"""
function save_validation_report(
    suite::BenchmarkSuite,
    filepath::String;
    format::String="markdown"
)
    report = generate_validation_report(suite, format=format)

    open(filepath, "w") do io
        write(io, report)
    end

    @info "Validation report saved to $filepath"
end

# ============================================================================
# SYNTHETIC TEST DATA
# ============================================================================

"""
    create_validation_test_data() -> Tuple{Dict, Dict}

Create synthetic test data for validation (known ground truth).
Useful for unit testing the validation module itself.
"""
function create_validation_test_data()::Tuple{Dict{String,Float64}, Dict{String,Float64}}
    # Simulated Darwin metrics
    darwin = Dict{String,Float64}(
        "porosity" => 0.92,
        "mean_pore_size_um" => 156.3,
        "interconnectivity" => 0.94,
        "tortuosity" => 1.08,
        "specific_surface_area" => 12.5
    )

    # Simulated reference (slightly different to test validation)
    reference = Dict{String,Float64}(
        "porosity" => 0.91,
        "mean_pore_size_um" => 154.0,
        "interconnectivity" => 0.93,
        "tortuosity" => 1.10,
        "specific_surface_area" => 12.8
    )

    return (darwin, reference)
end

"""
Run self-test of validation module.
"""
function self_test()::Bool
    @info "Running ValidationBenchmark self-test..."

    darwin, reference = create_validation_test_data()

    suite = run_validation(
        darwin, reference,
        dataset_name="Self-Test",
        dataset_source="Synthetic",
        reference_source="Ground Truth"
    )

    @info "Self-test results:"
    @info "  Pass rate: $(suite.summary_stats["pass_rate"])%"
    @info "  Mean error: $(round(suite.summary_stats["mean_relative_error"], digits=2))%"
    @info "  Overall: $(suite.overall_passed ? "PASSED" : "FAILED")"

    return suite.overall_passed
end

end # module ValidationBenchmark
