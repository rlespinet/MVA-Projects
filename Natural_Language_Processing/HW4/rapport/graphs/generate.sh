#!/bin/bash

for i in *.viz; do
    dot $i -Tsvg -o "../imgs/"${i/viz/svg}
done
