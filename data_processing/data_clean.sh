

if [ $# -ne 3 ]
then
    echo 'Syntax : $ bash data_clean.sh L config_start config_end '
    echo "Choice 1 : (L=64,  0 <= config_start <= config_end < 75) "
    echo "Choice 2 : (L=80,  0 <= config_start <= config_end < 60) "
    echo "Choice 3 : (L=100, 0 <= config_start <= config_end < 50) "
    echo "Choice 4 : (L=128, 0 <= config_start <= config_end < 40) "
    echo "Choice 5 : (L=160, 0 <= config_start <= config_end < 30) "
    
    exit 1
fi

if [ `ls config_*/o_r_O2_2D_$1_*.dat | grep -c ".dat"` -ne 0 ]
then
    echo -n "No. of Files 'config_*/o_r_O2_2D_$1_*.dat' : "
    ls config_*/o_r_O2_2D_$1_*.dat | grep -c ".dat" 
    echo "You may want to clean: \$ rm config_*/o_r_O2_2D_$1_*.dat"
    echo -n "Press 0 to Abort, 1 to Continue, 2 to Clean and Continue : "
    read input
    if [ $input -eq 2 ]
    then
        rm config_*/o_r_O2_2D_$1_*.dat
    fi
    if [ $input -eq 0 ]
    then
        exit 1
    fi
    
fi

# rm config_*/o_r_O2_2D_$1_*.dat

for (( k=$2; k<=$3; k++ ))
do
    echo -e -n "\rconfig_$2-${k}/                                        "
    cd "config_${k}/"
    for i in $1 
    do
        echo -e -n "\rconfig_$2-${k}/L=${i},                             "
        # echo "$i" >> o_r_O2_2D_all.dat
        # echo 'L{N=LxL} |h|, SUM[del(m_x)^2] SUM[del(m_y)^2] SUM[del(E)^2] SUM[del(|m|)^2]' >> o_r_O2_2D_all.dat
        for j in 0.010000 0.012000 0.014000 0.016000 0.018000 0.020000 0.022000 0.024000 0.026000 0.028000 0.030000 0.032000 0.034000 0.035000 0.036000 0.037000 0.038000 0.039000 0.040000 0.041000 0.042000 0.043000 0.044000 0.045000 0.046000 0.048000 0.050000 0.052000 0.054000 0.056000 0.058000 0.060000 0.064000 0.070000 0.080000 0.090000 0.100000 0.110000 0.120000 0.130000 0.140000 0.150000 
        do
            echo -e -n "\rconfig_$2-${k}/L=${i}/sigma_h=${j}...          "
            sed '/^[^0-9]/d' *_${i}x${i}_*\(${j}\)*_o_r.dat > o_r_O2_2D_${i}_${j}.dat
            sed -i '/^$/d' o_r_O2_2D_${i}_${j}.dat 
            # tail -n 20002 o_r_O2_2D_${i}_${j}_temp.dat > o_r_O2_2D_${i}_${j}.dat
        done
        echo -e -n "\rconfig_$2-${k}/L=${i}/sigma_h=0.01-0.15                "
        # echo "" >> o_r_O2_2D_all.dat
    done
    cd ..
    echo -e -n "\rconfig_$2-${k}/L=$1/sigma_h=0.01-0.15                "
done
echo -e "\rconfig_$2-$3/L=$1/sigma_h=0.01-0.15                "
