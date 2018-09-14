#!/bin/bash
for (( j=86; j<=128; j++ ))
do
    echo "config_${j}/"
    cd "config_${j}/"
    path=$(pwd)
    echo $path
    for i in 5 6 7 10
    do
        echo "a_${i}.out"
        cp "../a_${i}.out" ./

        echo "Mp_Batch_File_${i}.job"
        cp "../Mp_Batch_File_${i}.job" ./

        sbatch Mp_Batch_File_${i}.job
        sleep 0.1
    done
    cd ..
    path=$(pwd)
    echo $path
    
done
