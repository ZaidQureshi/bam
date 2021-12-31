#!/usr/bin/env bash
set -x

if [ $# -ne 2 ]
then
	echo Usage $0 gpuid tbsize && exit 1
fi


NUMDATASET=6
declare -a GraphFileArray=(
"/mnt/graphs/GAP-kron.bel"
"/mnt/graphs/GAP-urand.bel"
"/mnt/graphs/com-Friendster.bel"
"/mnt/graphs/MOLIERE_2016.bel"
"/mnt/graphs/uk-2007-05.bel"
"/mnt/graphs/sk-2005.bel"
)

SSSPNUMDATASET=5
declare -a GraphFileArraySSSP=(
"/mnt/graphs/GAP-kron.bel"
"/mnt/graphs/GAP-urand.bel"
"/mnt/graphs/com-Friendster.bel"
"/mnt/graphs/uk-2007-05.bel"
"/mnt/graphs/sk-2005.bel"
)
#echo "${GraphFileArray[5]} offset is ${GraphFileOffset[5]}"

#CTRL=$1
MEMTYPE=2  #BAFS_DIRECT
GPU=$1
TB=128


make benchmark -j

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]}  ++++++++++++++++++"
    sysctl vm.drop_caches=3
    for IMPLTYPE in 0 1 6 7 #baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        ./bin/nvm-bfs-bench -f ${GraphFileArray[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --repeat 32 --gpu $GPU --threads $TB
    done
done

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]}  ++++++++++++++++++"
    for IMPLTYPE in 0 1 6 7 #baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        ./bin/nvm-cc-bench -f ${GraphFileArray[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --gpu $GPU --threads $TB
    done
done

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]}  ++++++++++++++++++"
    for IMPLTYPE in 0 1 6 7 #baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        ./bin/nvm-pagerank-bench -f ${GraphFileArray[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --gpu $GPU --threads $TB
    done
done

for ((gfid=0; gfid<SSSPNUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArraySSSP[gfid]}  ++++++++++++++++++"
    sysctl vm.drop_caches=3
    for IMPLTYPE in 0 1 #baseline, coalesced, frontier, frontier coaslesced.
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        ./bin/nvm-sssp-bench -f ${GraphFileArraySSSP[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --repeat 32 --gpu $GPU --threads $TB
    done
done

for IMPLTYPE in 0 1 #baseline, coalesced, frontier, frontier coaslesced.
do
     sysctl vm.drop_caches=3
     echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
     ./bin/nvm-sssp_float-bench -f /mnt/graphs/MOLIERE_2016.bel --impl_type $IMPLTYPE --memalloc $MEMTYPE --repeat 32 --gpu $GPU --threads $TB
done
