#!/bin/bash

if [ "$1" == "-B" ]; then
    all=1
fi

for i in *.svg; do
    newname=${i/svg/pdf}
    if [ $all ] || [ ! -f $newname ]; then
        echo $newname
        inkscape --export-pdf=$newname $i 2> /dev/null
    fi
done
