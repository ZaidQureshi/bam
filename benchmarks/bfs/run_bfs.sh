#!/usr/bin/env bash
set -x

if [ $# -ne 3 ]
then
	echo Usage $0 numssd gpuid tbsize && exit 1
fi


NUMDATASET=6
declare -a GraphFileArray=(
"/home/vsm2/bafsdata/GAP-kron.bel"
"/home/vsm2/bafsdata/GAP-urand.bel"
"/home/vsm2/bafsdata/com-Friendster.bel"
"/home/vsm2/bafsdata/MOLIERE_2016.bel"
"/home/vsm2/bafsdata/uk-2007-05.bel"
"/home/vsm2/bafsdata/sk-2005.bel"
)
declare -a GraphFileOffset=(
"$((1024*1024*1024*0))"
"$((1024*1024*1024*64))"
"$((1024*1024*1024*160))"
"$((1024*1024*1024*224))"
"$((1024*1024*1024*320))"
"$((1024*1024*1024*384))"
)
declare -a GraphRootNode=(
"58720242"
"58720256"
"28703654"
"13229860"
"46329738"
"37977096"
)

#echo "${GraphFileArray[5]} offset is ${GraphFileOffset[5]}"


CTRL=4
MEMTYPE=6  #BAFS_DIRECT
GPU=0
TB=128
S=4096
IMPLTYPE=20

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]} located at offset ${GraphFileOffset[gfid]} ++++++++++++++++++"
    for S in 4096    ##baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $S Sector size ++++++++++++++++++"
        for P in 65536
        do
            echo "++++++++++++++++++ $P Page size ++++++++++++++++++"
            for ((C=1; C<=$CTRL; C++))
            do
                echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
                ./bin/nvm-bfs-bench -f ${GraphFileArray[gfid]} -l ${GraphFileOffset[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --src ${GraphRootNode[gfid]} --n_ctrls $C -p $P --gpu $GPU --threads $TB --sector_size 4096 --COARSE 8 --queue_depth 256 --num_queues 128 | grep -e time -e IO -e Hit
            done
        done
    done
done

