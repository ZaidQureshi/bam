#!/bin/bash

find -L . /lib/modules/$(uname -r)/build/ -type f -regex '.+[^/].*\.\(c\|h\|cc\|cu\|cuh\|cpp\)' > cscope.files
cscope -qRb
