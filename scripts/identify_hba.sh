#!/bin/bash
for a in `lspci -D -d 1000:c010 | cut -d" " -f1`

do
        if [ $(lspci -vv -s $a | egrep -i "Upstream | 00-80-5e" | wc -l) == 2 ]; then
                if
                        [ $(lspci -vv -s $a | egrep -i "Power budget" | wc -l) == 1 ]; then
                        echo "Falcon_HBA_BUS#" $a
                fi
        fi
done


