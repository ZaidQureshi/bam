#!/usr/bin/env bash
set -eux


P=4096
R=1
B=1024
G=2
for C in 1 2 3 4 5 6 7
do
    echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
    for T in 1024 2048 4096 8192 16384 32768 65536 131072 262144
    do
        echo "------------------ $T Threads ------------------"
        ../../build/bin/nvm-cuda-bench --threads=$T --blk_size=$B --reqs=$R --pages=$T --queue_depth=1024 --num_queues=128 --page_size=$P --n_ctrls=$C --gpu=$G | grep "IO"
    done

done
