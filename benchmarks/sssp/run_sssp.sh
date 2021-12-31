#!/usr/bin/env bash
set -x

if [ $# -ne 3 ]
then
	echo Usage $0 numssd gpuid tbsize && exit 1
fi


NUMDATASET=5 #1 dataset MOLIERE is a floating type. Will be used separately later.
declare -a GraphFileArray=(
"/home/vsm2/bafsdata/GAP-kron.bel"
"/home/vsm2/bafsdata/GAP-urand.bel"
"/home/vsm2/bafsdata/com-Friendster.bel"
"/home/vsm2/bafsdata/uk-2007-05.bel"
"/home/vsm2/bafsdata/sk-2005.bel"
"/home/vsm2/bafsdata/MOLIERE_2016.bel"
)
declare -a GraphFileOffset=(
"$((1024*1024*1024*0))"
"$((1024*1024*1024*64))"
"$((1024*1024*1024*160))"
"$((1024*1024*1024*320))"
"$((1024*1024*1024*384))"
"$((1024*1024*1024*224))"
)

declare -a GraphWeightOffset=(
"$((1024*1024*1024*32))"
"$((1024*1024*1024*128))"
"$((1024*1024*1024*192))"
"$((1024*1024*1024*352))"
"$((1024*1024*1024*416))"
"$((1024*1024*1024*288))"
)


declare -a GraphRootNode=(
"58720242"
"58720256"
"28703654"
"46329738"
"37977096"
"13229860"
)
#echo "${GraphFileArray[5]} offset is ${GraphFileOffset[5]}"


CTRL=$1
MEMTYPE=6  #BAFS_DIRECT
GPU=$2
TB=128

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]} located at offset ${GraphFileOffset[gfid]} and weight at ${GraphWeightOffset[gfid]}++++++++++++++++++"
    for IMPLTYPE in 4 #3 8 9    ##baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
            for P in 512 4096 8192
            #for P in 512 1024 2048 4096 8192 16384
            do
                echo "++++++++++++++++++ $P Page size ++++++++++++++++++"
                ./bin/nvm-sssp-bench -f ${GraphFileArray[gfid]} -l ${GraphFileOffset[gfid]} -w ${GraphWeightOffset[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --src ${GraphRootNode[gfid]} --n_ctrls $C -p $P --gpu $GPU --threads $TB
            done
        done
    done
done



echo "++++++++++++++++++ ${GraphFileArray[6]} located at offset ${GraphFileOffset[6]} and weight at ${GraphWeightOffset[6]}++++++++++++++++++"
for IMPLTYPE in 4 #3 8 9    ##baseline, coalesced, frontier, frontier coaslesced.
do
    echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
    for ((C=1; C<=$CTRL; C++))
    do
        echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
        for P in 512 4096 8192
        #for P in 512 1024 2048 4096 8192 16384
        do
            echo "++++++++++++++++++ $P Page size ++++++++++++++++++"
            ./bin/nvm-sssp_float-bench -f ${GraphFileArray[6]} -l ${GraphFileOffset[6]} -w ${GraphWeightOffset[6]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --src ${GraphRootNode[6]} --n_ctrls $C -p $P --gpu $GPU --threads $TB
        done
    done
done
