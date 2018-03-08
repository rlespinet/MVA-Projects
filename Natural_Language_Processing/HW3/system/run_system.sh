#!/bin/bash

if [ ! -h context2vec ]; then
    echo "Creating symbolic link for contex2vec"
    if [ ! "$CONTEXT2VECDIR" ]; then
        echo '$CONTEXT2VECDIR is not set. Please set it and relaunch the script'
        exit 1
    fi
    ln -s $CONTEXT2VECDIR context2vec
fi


cat <<EOF
You can either run the program on all the corpus or in interactive
mode. To run in interactive use

    python interactive.py <contex2vec_param>

To run the program on all the corpus use

    python process_corpus.py <contex2vec_param> <corpus>

You can read the README for a more detailed help. you will also fall
back on help by running the scripts with missing arguments
EOF
