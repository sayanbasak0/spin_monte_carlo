
mkdir before_periodicity_data

max=0
for (( k=0; k<=$2; k++ ))
do
    echo "config_${k}/"
    cd "config_${k}/"
    for i in "$1" 
    do
        # echo "$i" >> o_r_O2_2D_all.dat
        # echo 'L{N=LxL} |h|, SUM[del(m_x)^2] SUM[del(m_y)^2] SUM[del(E)^2] SUM[del(|m|)^2]' >> o_r_O2_2D_all.dat
        for j in 0.010000 0.012000 0.014000 0.016000 0.018000 0.020000 0.022000 0.024000 0.026000 0.028000 0.030000 0.032000 0.034000 0.035000 0.036000 0.037000 0.038000 0.039000 0.040000 0.041000 0.042000 0.043000 0.044000 0.045000 0.046000 0.048000 0.050000 0.052000 0.054000 0.056000 0.058000 0.060000 0.064000 0.070000 0.080000 0.090000 0.100000 0.110000 0.120000 0.130000 0.140000 0.150000 
        do
            echo -n "h=${j}, "
            sed '1,26d' O*_${i}x${i}_*\(${j}\)*_o_r.dat > temp_O.dat
            sed -i '/----/d' temp_O.dat
            sed -i '/Updating/d' temp_O.dat

            a=`tail -n 1 temp_O.dat | grep -o -E '[0-9]+' | head -n 1`
            b=`tail -n 1 temp_O.dat | grep -o -E '[0-9]+' | tail -n 1`
            c=$(($a-$b))
            sed -n '1p' temp_O.dat > data_file_temp.dat
            d=`cat temp_O.dat | grep -n -m 1 "loop $c" | sed 's/\([0-9]*\).*/\1/'`
            let d=d+1
            sed -n "${d}p" temp_O.dat >> data_file_temp.dat
            CUTOFF=0.000000000001
            nbef=`awk -v cutoff="${CUTOFF}" -v b="$b" 'NR>1 {if(b>1) print b; else if($4-p4>cutoff) print 0; else if($5-p5>cutoff) print 0; else if(p4-$4>cutoff) print 0; else if(p5-$5>cutoff) print 0; else print b;}' data_file_temp.dat`
            echo ${nbef} >> ../before_periodicity_data/period_${i}_${j}_${nbef}.dat
            # echo "$c" >> ../before_periodicity_data/period_${i}_${j}_${c}.dat
            if [ "${nbef}" -gt "${max}" ]
            then 
                let max=nbef
            fi
        done
        # echo "" >> o_r_O2_2D_all.dat
    done
    cd ..
done

# for (( k=1; k<=$max; k++ ))
# do
#     for i in "$1" 
#     do
#         # echo "$i" >> o_r_O2_2D_all.dat
#         # echo 'L{N=LxL} |h|, SUM[del(m_x)^2] SUM[del(m_y)^2] SUM[del(E)^2] SUM[del(|m|)^2]' >> o_r_O2_2D_all.dat
#         for j in 0.010000 0.012000 0.014000 0.016000 0.018000 0.020000 0.022000 0.024000 0.026000 0.028000 0.030000 0.032000 0.034000 0.035000 0.036000 0.037000 0.038000 0.039000 0.040000 0.041000 0.042000 0.043000 0.044000 0.045000 0.046000 0.048000 0.050000 0.052000 0.054000 0.056000 0.058000 0.060000 0.064000 0.070000 0.080000 0.090000 0.100000 0.110000 0.120000 0.130000 0.140000 0.150000 
#         do
#             echo -n "h=${j}, "
            
#             d=`cat before_periodicity_data/period_${i}_${j}_${k}.dat | grep -c "${k}"`
#             echo "$i $j $k $d" >> period_${k}.dat
#         done
#         # echo "" >> o_r_O2_2D_all.dat
#     done
# done
