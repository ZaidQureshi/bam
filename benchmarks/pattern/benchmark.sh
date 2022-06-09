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
T=4194304
E=137438953472
#for P in 32768 131072
#do
#    echo "++++++++++++++++++ $P Page size ++++++++++++++++++" 
    for S in 512 1024 2048 4096
    do
        echo "++++++++++++++++++ $S Sector size ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controller ++++++++++++++++++"
            for P in 1024 2048 4096 8192 16384 32768 65536 131072 262144
            do
                echo "++++++++++++++++++ $P Page size ++++++++++++++++++"
                ./bin/nvm-pattern-bench --input_a /home/vsm2/bafsdata/GAP-kron.bel --memalloc 6 --n_ctrls $C -p $P --sectorsize $S --gpu $GPU --threads $T --n_elems $E --impl_type 7 --queue_depth 1024 --num_queues 128 --blk_size 1024 | grep -e P:0 -e P:10
            done
        done
    done 
#done
