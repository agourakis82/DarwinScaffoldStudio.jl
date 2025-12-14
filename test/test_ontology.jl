"""
Ontology Module Tests
Tests for OBO Foundry integration and ontology lookups
"""

using Test
using DarwinScaffoldStudio

@testset "Ontology Module" begin
    @testset "OBO Foundry Terms" begin
        # Test UBERON (anatomy)
        @test haskey(DarwinScaffoldStudio.Ontology.OBOFoundry.UBERON, "bone tissue")
        bone = DarwinScaffoldStudio.Ontology.OBOFoundry.UBERON["bone tissue"]
        @test bone.id == "UBERON:0002481"
        @test occursin("bone", lowercase(bone.name))

        # Test CL (cell types)
        @test haskey(DarwinScaffoldStudio.Ontology.OBOFoundry.CL, "osteoblast")
        osteoblast = DarwinScaffoldStudio.Ontology.OBOFoundry.CL["osteoblast"]
        @test osteoblast.id == "CL:0000062"

        # Test CHEBI (chemicals)
        @test haskey(DarwinScaffoldStudio.Ontology.OBOFoundry.CHEBI, "hydroxyapatite")
        ha = DarwinScaffoldStudio.Ontology.OBOFoundry.CHEBI["hydroxyapatite"]
        @test startswith(ha.id, "CHEBI:")
    end

    @testset "Tissue Library" begin
        TL = DarwinScaffoldStudio.Ontology.TissueLibrary

        # Test bone tissue entry using TISSUES dict
        @test haskey(TL.TISSUES, "UBERON:0002481")
        bone = TL.TISSUES["UBERON:0002481"]
        @test bone.id == "UBERON:0002481"

        # Test get_tissue function
        bone_info = TL.get_tissue("UBERON:0002481")
        @test bone_info !== nothing
        @test bone_info.id == "UBERON:0002481"
    end

    @testset "Cell Library" begin
        CL = DarwinScaffoldStudio.Ontology.CellLibrary

        # Test osteoblast entry using CELLS dict
        @test haskey(CL.CELLS, "CL:0000062")
        osteoblast = CL.CELLS["CL:0000062"]
        @test osteoblast.id == "CL:0000062"
    end

    @testset "Material Library" begin
        ML = DarwinScaffoldStudio.Ontology.MaterialLibrary

        # Test hydroxyapatite
        @test haskey(ML.MATERIALS, "CHEBI:46662")
        ha = ML.MATERIALS["CHEBI:46662"]
        @test ha.id == "CHEBI:46662"
    end

    @testset "Disease Library" begin
        # Disease Library uses DISEASES constant
        DL = DarwinScaffoldStudio.Ontology.DiseaseLibrary

        # Test that DISEASES dict exists and has content
        @test !isempty(DL.DISEASES)

        # Test bone disorders exist
        @test !isempty(DL.BONE_DISORDERS)

        # Test get_disease function
        osteoporosis = DL.get_disease("NCIT:C3298")
        @test osteoporosis !== nothing || haskey(DL.DISEASES, "NCIT:C3298")
    end

    @testset "Cross-Ontology Relations" begin
        COR = DarwinScaffoldStudio.Ontology.CrossOntologyRelations

        # Test tissue-cell relations
        @test haskey(COR.TISSUE_CELL_RELATIONS, "UBERON:0002481")  # bone
        bone_cells = COR.TISSUE_CELL_RELATIONS["UBERON:0002481"]
        @test "CL:0000062" in bone_cells  # osteoblast

        # Test tissue-material relations
        @test haskey(COR.TISSUE_MATERIAL_RELATIONS, "UBERON:0002481")
    end
end

println("✅ Ontology tests passed!")
