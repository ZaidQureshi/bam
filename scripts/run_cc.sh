#!/usr/bin/env bash
set -x

if [ $# -ne 3 ]
then
	echo Usage $0 numssd gpuid tbsize && exit 1
fi


#Initialize set of files are taken from EMOGI and graphBIG.

NUMDATASET=4
declare -a GraphFileArray=(
"/home/vmailthody/data/GAP-kron.bel"
"/home/vmailthody/data/GAP-urand.bel"
"/home/vmailthody/data/com-Friendster.bel"
"/home/vmailthody/data/MOLIERE_2016.bel"
)
declare -a GraphFileOffset=(
"$((1024*1024*1024*0))"
"$((1024*1024*1024*64))"
"$((1024*1024*1024*160))"
"$((1024*1024*1024*224))"
)


CTRL=$1
MEMTYPE=6  #BAFS_DIRECT
GPU=$2
TB=128

for ((gfid=0; gfid<NUMDATASET; gfid++))
do
    echo "++++++++++++++++++ ${GraphFileArray[gfid]} ++++++++++++++++++"
    for IMPLTYPE in 4 10
    do
        echo "++++++++++++++++++ $IMPLTYPE Type ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controller ++++++++++++++++++"
            for P in 4096 512
            do
                echo "++++++++++++++++++ $P page size++++++++++++++++++"
                #for stride in 1 16 32 128 512 1024 4096 16384 131072 262144 1048576 4194304
                #for stride in 1 16 32 128 512 1024 4096
                #for stride in 128 512
                #do
                   #echo "++++++++++++++++++ $CS stride factor ++++++++++++++++++"
                   #for coarse in 1 2 4 8 16 32
                   #for coarse in 1
                   #do
                   #echo "++++++++++++++++++ $COARSE coarsened ++++++++++++++++++"
                        ./bin/nvm-cc-bench -f ${GraphFileArray[gfid]} -l ${GraphFileOffset[gfid]} --impl_type $IMPLTYPE --memalloc $MEMTYPE --n_ctrls $C -p $P --gpu $GPU --threads $TB -M $((8*1024*1024*1024)) #-C $coarse -P $stride
                    #done
                #done
            done
        done
    done
done

