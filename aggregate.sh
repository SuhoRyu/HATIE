#!/bin/bash

python evaluation/aggregate.py \
    --query_file "queries/queries_wo_remove.pkl" \
    --prefix "modelA" \
    --outdir "outputs" \
    --compute_errors 1 