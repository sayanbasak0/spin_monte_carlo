
set size ratio -1

set title font "Times New Roman,24"
set key font "Times New Roman,20"
set xlabel font "Times New Roman,18"
set ylabel font "Times New Roman,18"
set ytics font "Times New Roman,16"
set xtics font "Times New Roman,16"

set title "Zero Temperature limit cycle of periodicity 7*2π,\n robust to small finite temperature fluctuations"

set xlabel "M_x"
set ylabel "M_y"

set obj 1 rect from -0.6,1.0 to -0.25,0.9 fc rgb "white" back
set label 1 at -0.425,0.95 "160x160 | H = 0.048J" center font "Times New Roman,22"

set key invert

plot 'O(2)_2D_hys_rot_64x64_0.010_{0.000,(0.048)}_{0.500,0.000}_{0.000}_o_r_ALL.dat' u 4:5 w lp pt 7 title "T = 0J",\
'data_16_0.00.dat' u 4:5 w lp pt 7 ps 0.5 title "T=0.01J (1 MC checkerboard sweep / {/Symbol d}{/Symbol f}=2π*0.0001)"