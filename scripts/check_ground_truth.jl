using XLSX
using Statistics

xf = XLSX.readxlsx("data/validation/porescript/Manual_Salt_Leached.xlsx")
println("Sheets: ", XLSX.sheetnames(xf))

sheet = xf["x27"]
println("\nColumn headers (row 1):")
for col in 1:15
    try
        v = sheet[1, col]
        if v !== missing
            println("  Col ", col, ": ", v)
        end
    catch
    end
end

# Read pore sizes from column H (index 8)
pore_sizes = Float64[]
for row in 2:500
    try
        v = sheet[row, 8]
        if v !== missing && v isa Number && v > 0
            push!(pore_sizes, Float64(v))
        end
    catch
        break
    end
end

n = length(pore_sizes)
println("\nGround truth statistics (n=", n, "):")
println("  Mean: ", round(mean(pore_sizes), digits=1), " um")
println("  Std:  ", round(std(pore_sizes), digits=1), " um")
println("  Min:  ", round(minimum(pore_sizes), digits=1), " um")
println("  Max:  ", round(maximum(pore_sizes), digits=1), " um")
