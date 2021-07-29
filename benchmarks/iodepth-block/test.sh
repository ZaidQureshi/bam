#!/usr/bin/env bash
set -euo pipefail

for i in {1..100000}
do
    ../../build/bin/nvm-block-bench --threads=$((1024*256*4)) --blk_size=64 --reqs=1 --pages=$((256*1024*4)) --queue_depth=1024  --page_size=$((512)) --num_blks=$((2097152)) --gpu=0 --n_ctrls=1 --num_queues=128

    echo "******************** $i *********************"
done
