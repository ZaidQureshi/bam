#!/usr/bin/env bash
set -x

if [ $# -ne 3 ]
then
	echo Usage $0 numssd gpuid tbsize && exit 1
fi

#No MOILERE as it is multigraph and not right to run PR.
NUMDATASET=5
declare -a GraphFileArray=(
"/home/vsm2/bafsdata/GAP-kron.bel"
"/home/vsm2/bafsdata/GAP-urand.bel"
"/home/vsm2/bafsdata/com-Friendster.bel"
"/home/vsm2/bafsdata/uk-2007-05.bel"
"/home/vsm2/bafsdata/sk-2005.bel"
)
declare -a GraphFileOffset=(
"$((1024*1024*1024*0))"
"$((1024*1024*1024*64))"
"$((1024*1024*1024*160))"
"$((1024*1024*1024*320))"
"$((1024*1024*1024*384))"
)

#echo "${GraphFileArray[5]} offset is ${GraphFileOffset[5]}"


CTRL=$1
MEMTYPE=6  #BAFS_DIRECT
GPU=$2
TB=128

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]} located at offset ${GraphFileOffset[gfid]} ++++++++++++++++++"
    for IMPLTYPE in 4 10 #3 9    ##baseline, coalesced, hash, hash coalesced
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
            for P in 512 4096 8192
            #for P in 512 1024 2048 4096 8192 16384
            do
                echo "++++++++++++++++++ $P Page size ++++++++++++++++++"
                ./bin/nvm-pagerank-bench -f ${GraphFileArray[gfid]} -l ${GraphFileOffset[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --n_ctrls $C -p $P --gpu $GPU --threads $TB
            done
        done
    done
done

