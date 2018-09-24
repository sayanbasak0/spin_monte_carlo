#!/bin/bash
for (( j=0; j<=128; j++ ))
do
    echo "config_${j}/"
    cd "config_${j}/"
    for i in 7 8
    do
        echo "a_${i}.out"
        cp "../a_${i}.out" ./

        echo "batch_${i}.sub"
        cp "../batch_${i}.sub" ./
        # pwdesc=$(echo $PWD | sed 's_/_\\/_g')
        # sed -i "13s/.*/cd ${pwdesc}/" "batch_${i}.job" 
        qsub batch_${i}.sub
        sleep 0.1
    done
    path=$(pwd)
    echo $path
    cd ..
    path=$(pwd)
    echo $path
    
done
