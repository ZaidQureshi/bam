#set -x

if [ $# -ne 2 ]
then
	echo Usage $0 logfile numssd && exit 1
fi

logfile=$1
CTRL=$2

NUMDATASET=6
declare -a GraphFileName=(
"GAP-kron.bel"
"GAP-urand.bel"
"com-Friendster.bel"
"MOLIERE_2016.bel"
"uk-2007-05.bel"
"sk-2005.bel"
"Dummy"
)

IMPLSIZE=2
declare -a ImplType=(
"3"
"4"
"5"
#"8"
#"9"
)

NUMPAGESIZE=3
declare -a PageSize=(
"512"
"4096"
"8192"
)

TYPE=Accesses
for((gid=0;gid<NUMDATASET;gid++))
do
    echo "++++++++++++++++++ ${GraphFileName[gid]} ++++++++++++++++++"
    for ((impl=0;impl<$IMPLSIZE;impl++))
    do
        echo "++++++++++++++++++ ${ImplType[impl]} Type ++++++++++++++++++"
        for ((C=1; C<=$CTRL; C++))
        do
            echo "++++++++++++++++++ $C Controllers ++++++++++++++++++"
            for ((pg=0; pg<$NUMPAGESIZE; pg++))
            do
                echo "++++++++++++++++++ ${PageSize[pg]} PageSize ++++++++++++++++++"
                cat ${logfile} | grep -v "GraphFile"| sed -n "/${GraphFileName[gid]}/,/${GraphFileName[gid+1]}/p" | sed -n "/impl_type ${ImplType[impl]}/,/impl_type ${ImplType[impl+1]}/p"| sed -n "/n_ctrls ${C}/,/n_ctrls ${C+1}/p"| sed -n "/-p ${PageSize[pg]}/,/-p ${PageSize[pg+1]}/p" | grep -v "+"  |grep ${TYPE} | awk '{sum+=$3;n++} END {if(n>0) printf "%.2f\n",sum/n}'
            done
        done
    done
done
