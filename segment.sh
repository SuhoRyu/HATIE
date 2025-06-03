#!/bin/bash

python evaluation/segment_targets.py \
    --query_file "queries/queries_wo_remove.pkl" \
    --prefix "modelA" \
    --edited_images "path/to/edited/images" \
    --outdir "outputs" \
    --image_format "jpg" \
    --use_gpu 1