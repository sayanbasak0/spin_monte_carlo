# Plot of file.dat 

# This command works for a linux computer. In linux, you need to specify the exact location of
# the font you want to use
set terminal png notransparent rounded giant enhanced font "Times New Roman" 24 \
  size 1200,960 

# nomirror means do not put tics on the opposite side of the plot
set xtics nomirror
set ytics nomirror

# On the Y axis put a major tick every 5
set xtics 0.1
set ytics 0.2

# On both the x and y axes split each space in half and put a minor tic there
set mxtics 2
set mytics 2


# Line style for axes
# Define a line style (we're calling it 80) and set 
# lt = linetype to 0 (dashed line)
# lc = linecolor to a gray defined by that number
set style line 80 lt 0 lc rgb "#808080"

# Set the border using the linestyle 80 that we defined
# 3 = 1 + 2 (1 = plot the bottom line and 2 = plot the left line)
# back means the border should be behind anything else drawn
set border 3 back ls 80 

# Line style for grid
# Define a new linestyle (81)
# linetype = 0 (dashed line)
# linecolor = gray
# lw = lineweight, make it half as wide as the axes lines
set style line 81 lt 0 lc rgb "#808080" lw 0.5

# Draw the grid lines for both the major and minor tics
set grid xtics
set grid ytics
set grid mxtics
set grid mytics

# Put the grid behind anything drawn and use the linestyle 81
set grid back ls 81

# Add line at -3db
# Draw a line from the right end of the graph to the left end of the graph at
# the y value of -3
# The line should not have an arrowhead
# Linewidth = 2
# Linecolor = black
# It should be in front of anything else drawn
# set arrow from graph 0,first -3 to graph 1, first -3 nohead lw 2 lc rgb "#000000" front

# Put a label -3db at 80% the width of the graph and y = -2 (it will be just above the line drawn)
# set label "-3dB" at graph 0.8, first -2

# Create some linestyles for our data
# pt = point type (triangles, circles, squares, etc.)
# ps = point size
set style line 1 lt 1 lc rgb "#A00000" lw 2 pt 7 ps 1.5
set style line 2 lt 1 lc rgb "#00A000" lw 2 pt 11 ps 1.5
set style line 3 lt 1 lc rgb "#5060D0" lw 2 pt 9 ps 1.5
set style line 4 lt 1 lc rgb "#0000A0" lw 2 pt 8 ps 1.5
set style line 5 lt 1 lc rgb "#D0D000" lw 2 pt 13 ps 1.5
set style line 6 lt 1 lc rgb "#00D0D0" lw 2 pt 12 ps 1.5
set style line 7 lt 1 lc rgb "#B200B2" lw 2 pt 5 ps 1.5

# Name our output file
set output "mag_y_abs_64.png"

# Put X and Y labels
set xlabel "T/J"
set ylabel "|<M_y>|"

# Set the range of our x and y axes
set xrange [0:2]
set yrange [-0.1:1]

# Give the plot a title
set title "Magnetisation vs Temperature upon cooling (N=64^2)"

# Put the legend at the bottom left of the plot
set key right top

# Plot the actual data
# u 1:2 = using column 1 for X axis and column 2 for Y axis
# w lp = with linepoints, meaning put a point symbol and draw a line
# ls 1 = use our defined linestyle 1
# t "Test 1" = title "Test 1" will go in the legend
# The rest of the lines plot columns 3, 5 and 7
plot \
"mean_std_abs_0.062500_c.dat" u 1:8:9 w yerrorbars ls 5 t "old sim, h_x = R_x/8", \
"mean_std_abs_0.071429_c.dat" u 1:8:9 w yerrorbars ls 4 t "new sim, h_x = R_x/7", \
"mean_std_abs_0.062500_c.dat" u 1:8 w l ls 5 notitle, \
"mean_std_abs_0.071429_c.dat" u 1:8 w l ls 4 notitle, \

# "mean_std_abs_0.000000_c.dat" u 1:6:7 w yerrorbars ls 1 t "h_x = 0.0", \
# "mean_std_abs_0.007812_c.dat" u 1:6:7 w yerrorbars ls 2 t "h_x = R_x/64", \
# "mean_std_abs_0.015625_c.dat" u 1:6:7 w yerrorbars ls 3 t "h_x = R_x/32", \
# "mean_std_abs_0.031250_c.dat" u 1:6:7 w yerrorbars ls 4 t "h_x = R_x/16", \
# "mean_std_abs_0.062500_c.dat" u 1:6:7 w yerrorbars ls 5 t "h_x = R_x/8", \
# "mean_std_abs_0.125000_c.dat" u 1:6:7 w yerrorbars ls 6 t "h_x = R_x/4", \
# "mean_std_abs_0.250000_c.dat" u 1:6:7 w yerrorbars ls 7 t "h_x = R_x/2"

# This is important because it closes our output file.
set output 
