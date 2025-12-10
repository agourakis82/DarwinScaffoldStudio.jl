# Gnuplot script for Figure 1: Multi-Polymer Validation
set terminal pdfcairo enhanced font 'Arial,12' size 6,4
set output 'fig1_multipolymer_validation.pdf'

set xlabel 'Time (days)'
set ylabel 'M_n / M_{n0}'
set key outside right top
set grid

set style line 1 lc rgb '#E41A1C' pt 7 ps 1.2 lw 2
set style line 2 lc rgb '#377EB8' pt 5 ps 1.2 lw 2
set style line 3 lc rgb '#4DAF4A' pt 9 ps 1.2 lw 2
set style line 4 lc rgb '#984EA3' pt 11 ps 1.2 lw 2
set style line 5 lc rgb '#FF7F00' pt 13 ps 1.2 lw 2

set xrange [0:*]
set yrange [0:1.1]

# Plot using the CSV data
# You'll need to filter by polymer name

