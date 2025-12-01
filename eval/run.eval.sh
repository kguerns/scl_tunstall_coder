#!/bin/bash

for c in $(seq 8 18); do
    echo "Running code_length=$c"
    python eval.py ../data/ --code_length "$c"
done