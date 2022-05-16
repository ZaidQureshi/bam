#!/usr/bin/env bash
set -x

if [ $# -ne 3 ]
then
	echo Usage $0 numssd gpuid tbsize && exit 1
fi

CTRL=$1
GPU=$2
TB=32
CS=8589934592
for P in 32768 131072
do
    echo "++++++++++++++++++ $P Page size ++++++++++++++++++" 
    for S in 512 4096
    do
        echo "++++++++++++++++++ $S Sector size ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controller ++++++++++++++++++"
            for T in 32768 65536 131072 262144 524288 1048576 2097152 4194304
            do
                echo "++++++++++++++++++ $T Threads ++++++++++++++++++"
                ./bin/nvm-sectorvalid-bench --n_ctrls $C -p $P --sector_size $S --gpu $GPU --threads $T --blk_size 32
            done
        done
    done 
done