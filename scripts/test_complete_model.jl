#!/usr/bin/env julia
"""
Test script for the complete PLDLA degradation model.
"""

println("="^80)
println("Testing Complete PLDLA Degradation Model")
println("="^80)

# Include the module
include("../src/DarwinScaffoldStudio/Science/CompletePLDLADegradation.jl")
using .CompletePLDLADegradation

# Run validation
run_complete_validation()
