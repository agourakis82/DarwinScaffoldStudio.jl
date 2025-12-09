#!/usr/bin/env julia
"""
Analyze Cambridge Scaffold Data
================================

Data from: https://www.repository.cam.ac.uk/handle/1810/303941
Paper: "MicroCT analysis of connectivity in porous structures"
Journal: J. R. Soc. Interface (2019)

This contains real scaffold measurements including:
- Porosity
- Pore size distribution
- Connectivity metrics

We'll analyze these to see if they support our D = φ prediction.
"""

using XLSX
using Statistics
using Printf

const DATA_DIR = joinpath(@__DIR__, "..", "data", "external_validation")
const φ = (1 + sqrt(5)) / 2

function analyze_scaffold_data()
    println("="^70)
    println("ANALYZING CAMBRIDGE SCAFFOLD DATA")
    println("="^70)
    println()

    filepath = joinpath(DATA_DIR, "ScaffoldDataAnalysis.xlsx")

    if !isfile(filepath)
        println("ERROR: File not found: $filepath")
        return nothing
    end

    println("Loading: $filepath")

    # Read Excel file
    xf = XLSX.readxlsx(filepath)

    # List all sheets
    println("\nAvailable sheets:")
    for sheet_name in XLSX.sheetnames(xf)
        println("  - $sheet_name")
    end

    # Analyze each sheet
    results = Dict()

    for sheet_name in XLSX.sheetnames(xf)
        println("\n" * "─"^60)
        println("Sheet: $sheet_name")
        println("─"^60)

        sheet = xf[sheet_name]

        # Get data range
        data = XLSX.getdata(sheet)

        if isempty(data)
            println("  (empty)")
            continue
        end

        # Print first few rows to understand structure
        n_rows = min(10, size(data, 1))
        n_cols = min(8, size(data, 2))

        println("\nFirst $n_rows rows, $n_cols columns:")
        for i in 1:n_rows
            row_str = join([string(data[i, j]) for j in 1:n_cols], " | ")
            println("  $row_str")
        end

        # Try to find porosity column
        headers = data[1, :]
        println("\nHeaders: $headers")

        results[sheet_name] = data
    end

    return results
end

function analyze_artificial_data()
    println("\n" * "="^70)
    println("ANALYZING ARTIFICIAL/SYNTHETIC DATA")
    println("="^70)
    println()

    filepath = joinpath(DATA_DIR, "ArtificialDataAnalysis.xlsx")

    if !isfile(filepath)
        println("ERROR: File not found: $filepath")
        return nothing
    end

    println("Loading: $filepath")

    xf = XLSX.readxlsx(filepath)

    println("\nAvailable sheets:")
    for sheet_name in XLSX.sheetnames(xf)
        println("  - $sheet_name")
    end

    # Analyze each sheet
    for sheet_name in XLSX.sheetnames(xf)
        println("\n" * "─"^60)
        println("Sheet: $sheet_name")
        println("─"^60)

        sheet = xf[sheet_name]
        data = XLSX.getdata(sheet)

        if isempty(data)
            println("  (empty)")
            continue
        end

        n_rows = min(10, size(data, 1))
        n_cols = min(8, size(data, 2))

        println("\nFirst $n_rows rows, $n_cols columns:")
        for i in 1:n_rows
            row_str = join([string(data[i, j]) for j in 1:n_cols], " | ")
            println("  $row_str")
        end

        headers = data[1, :]
        println("\nHeaders: $headers")
    end
end

function main()
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  CAMBRIDGE SCAFFOLD DATA ANALYSIS                                ║")
    println("╚══════════════════════════════════════════════════════════════════╝")
    println()
    println("Golden ratio φ = $(round(φ, digits=4))")
    println("Prediction: D = φ at ~95.8% porosity")
    println()

    # Analyze scaffold data
    scaffold_data = analyze_scaffold_data()

    # Analyze artificial data
    analyze_artificial_data()

    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("\nThe Cambridge data contains structural metrics from real scaffolds.")
    println("Key metrics to look for:")
    println("  - Porosity values")
    println("  - Pore size distributions")
    println("  - Connectivity measurements")
    println()
    println("If scaffolds have porosity ~95-96%, our prediction says D ≈ φ")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
