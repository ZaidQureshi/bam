#!/usr/bin/env bash
set -eux


P=512
R=0
B=64
G=1
A=0
RT=50
NB=$((128*1024*1024))
for C in $1
do
    echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
    for T in 65536 131072 262144
    do
        echo "------------------ $T Threads ------------------"
        ../../build/bin/nvm-block-bench --threads=$T --blk_size=$B --reqs=1 --pages=$T --queue_depth=1024 --num_queues=128 --page_size=$P --n_ctrls=$C --gpu=$G --num_blks=$NB --access_type=$R
    done

done
