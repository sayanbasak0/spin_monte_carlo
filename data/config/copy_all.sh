#!/bin/bash
for (( j=0; j<=64; j++ ))
do
    echo "config_${j}/"
    cd "config_${j}/"
    for i in 5 6 7 10
    do
        echo "a_${i}.out"
        cp "../a_${i}.out" ./

        echo "Mp_Batch_File_${i}.job"
        cp "../Mp_Batch_File_${i}.job" ./
        pwdesc=$(echo $PWD | sed 's_/_\\/_g')
        sed -i "13s/.*/cd ${pwdesc}/" "Mp_Batch_File_${i}.job" 
        sbatch Mp_Batch_File_${i}.job
        sleep 0.1
    done
    path=$(pwd)
    echo $path
    cd ..
    path=$(pwd)
    echo $path
    
done
