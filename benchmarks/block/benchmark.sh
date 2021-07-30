#!/usr/bin/env bash
set -x

if [ $# -ne 1 ]
then
	echo Usage $0 numssd  && exit 1
fi

set -eux
P=512
R=0
B=64
G=1
A=0
RT=50
NB=$((128*1024*1024))
CTRLS=$1
for C in 1 .. $CTRLS
do
    echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
    for T in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
    do
        echo "------------------ $T Threads ------------------"
        ../../build/bin/nvm-block-bench --threads=$T --blk_size=$B --reqs=1 --pages=$T --queue_depth=1024 --num_queues=128 --page_size=$P --n_ctrls=$C --gpu=$G --num_blks=$NB --access_type=$R
    done

done
