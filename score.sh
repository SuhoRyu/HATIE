#!/bin/bash

python evaluation/score.py \
    --query_file "queries/queries_wo_remove.pkl" \
    --prefix "imodelA" \
    --original_seg_file "HATIE/editable_objs_mask.pkl" \
    --original_images "original_images" \
    --edited_images "path/to/edited/images" \
    --outdir "outputs" \
    --use_gpu 1 \
    --compute_err 1