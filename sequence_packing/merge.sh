#!/bin/bash

corpus=$1
seq_len=$2

log_dir=/script/logs/${corpus}
mkdir -p $log_dir
output="${log_dir}/${SLURM_PROCID}.txt"

echo "Output: $output"

python3 merge_engine.py \
    --corpus=$corpus \
    --max_seq_len=$seq_len \
    --n_shards=$SLURM_NPROCS \
    --shard_idx=$SLURM_PROCID \
    > $output