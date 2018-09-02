#!/bin/bash


# cd "new_${i}"
# sed -i '1,4d' config_*/*_c_r.dat

# sed -i '/loop/d' config_*/*_c_r.dat
# sed -i '/loop/d' new_*/*_c_r.dat
# tail -n 10002 new_*/*_c_r.dat
# sed -i '1,2d' config_*/*_c.dat
# sed -i '1,2d' new_*/*_c.dat
# for j in {0..128}
# do
#     for i in 0.000000 0.007812 0.015625 0.031250 0.062500 0.125000 0.250000 
#     do
#         tail -n 10002 config_${j}/*\(${i}\)*_c_r.dat > config_${j}/${j}_c_r_last.dat
#         head -n 10006 config_${j}/*\(${i}\)*_c_r.dat |tail -n 10002 > config_${j}/${i}_c_r_first.dat
#         head -n 5 config_${j}/*\(${i}\)*_c_r.dat |tail -n 4 > config_${j}/${i}_c_r_m_vs_h.dat

#     done
# done

for i in 0.000000 0.007812 0.015625 0.031250 0.062500 0.125000 0.250000
do
    # awk '{a1[FNR]=$1;a2[FNR]=$2;a3[FNR]=$3;a4m[FNR]+=$4;a4d[FNR]+=$4*$4;a5m[FNR]+=$5;a5d[FNR]+=$5*$5;a6m[FNR]+=$6;a6d[FNR]+=$6*$6;n[FNR]++;}END{for(i=1;i<=FNR;i++)print a1[i],a2[i],a3[i],a4m[i]/n[i],sqrt(sqrt((a4d[i]/n[i]-(a4m[i]/n[i])*(a4m[i]/n[i]))^2)),a5m[i]/n[i],sqrt(sqrt((a5d[i]/n[i]-(a5m[i]/n[i])*(a5m[i]/n[i]))^2)),a6m[i]/n[i],sqrt(sqrt((a6d[i]/n[i]-(a6m[i]/n[i])*(a6m[i]/n[i]))^2));}' config_*/*\(${i}\)*_c_r.dat > mean_std_${i}_c_r.dat
    awk '{a1[FNR]=$1;a2[FNR]=$2;a3[FNR]=$3;a4m[FNR]+=sqrt($4*$4);a4d[FNR]+=$4*$4;a5m[FNR]+=sqrt($5*$5);a5d[FNR]+=$5*$5;a6m[FNR]+=sqrt($6*$6);a6d[FNR]+=$6*$6;n[FNR]++;}END{for(i=1;i<=FNR;i++)print a1[i],a2[i],a3[i],a4m[i]/n[i],sqrt(sqrt((a4d[i]/n[i]-(a4m[i]/n[i])*(a4m[i]/n[i]))^2)),a5m[i]/n[i],sqrt(sqrt((a5d[i]/n[i]-(a5m[i]/n[i])*(a5m[i]/n[i]))^2)),a6m[i]/n[i],sqrt(sqrt((a6d[i]/n[i]-(a6m[i]/n[i])*(a6m[i]/n[i]))^2));}' config_*/${i}_c_r_m_vs_h.dat > mean_std_abs_${i}_c_r_m_vs_h.dat
    
    # awk '{a1[FNR]=$1;a2m[FNR]+=sqrt($2*$2);a2d[FNR]+=$2*$2;a3m[FNR]+=sqrt($3*$3);a3d[FNR]+=$3*$3;a4m[FNR]+=sqrt($4*$4);a4d[FNR]+=$4*$4;a5m[FNR]+=sqrt($5*$5);n[FNR]++;a5d[FNR]+=$5*$5;}END{for(i=1;i<=FNR;i++)print a1[i],a2m[i]/n[i],sqrt(sqrt((a2d[i]/n[i]-(a2m[i]/n[i])*(a2m[i]/n[i]))^2)),a3m[i]/n[i],sqrt(sqrt((a3d[i]/n[i]-(a3m[i]/n[i])*(a3m[i]/n[i]))^2)),a4m[i]/n[i],sqrt(sqrt((a4d[i]/n[i]-(a4m[i]/n[i])*(a4m[i]/n[i]))^2)),a5m[i]/n[i],sqrt(sqrt((a5d[i]/n[i]-(a5m[i]/n[i])*(a5m[i]/n[i]))^2));}' config_*/*\(${i}\)*_c.dat > mean_std_abs_${i}_c.dat
    # awk '{a1[FNR]=$1;a2m[FNR]+=$2;a2d[FNR]+=$2*$2;a3m[FNR]+=$3;a3d[FNR]+=$3*$3;a4m[FNR]+=$4;a4d[FNR]+=$4*$4;a5m[FNR]+=$5;n[FNR]++;a5d[FNR]+=$5*$5;}END{for(i=1;i<=FNR;i++)print a1[i],a2m[i]/n[i],sqrt(sqrt((a2d[i]/n[i]-(a2m[i]/n[i])*(a2m[i]/n[i]))^2)),a3m[i]/n[i],sqrt(sqrt((a3d[i]/n[i]-(a3m[i]/n[i])*(a3m[i]/n[i]))^2)),a4m[i]/n[i],sqrt(sqrt((a4d[i]/n[i]-(a4m[i]/n[i])*(a4m[i]/n[i]))^2)),a5m[i]/n[i],sqrt(sqrt((a5d[i]/n[i]-(a5m[i]/n[i])*(a5m[i]/n[i]))^2));}' new_*/*\(${i}\)*_c.dat > mean_std_${i}_c.dat
    
done
# awk '{a[FNR]+=$2;b[FNR]++;}END{for(i=1;i<=FNR;i++)print i,a[i]/b[i];}' *_c.dat > mean_c.dat
# cd ..