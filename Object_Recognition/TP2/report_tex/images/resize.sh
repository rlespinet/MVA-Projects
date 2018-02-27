#!/bin/sh

for i in $(ls airplane*); do
    convert $i -crop 844x267+212+19 cropped_$i;
    convert cropped_$i -chop 177x0+270 cropped_$i;
done
