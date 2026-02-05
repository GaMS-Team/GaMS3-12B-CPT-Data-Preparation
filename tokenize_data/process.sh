#!/bin/bash

corpus=$1
seq_length=$2

log_dir=/script/logs/${corpus}
mkdir -p $log_dir
output="${log_dir}/${SLURM_PROCID}.txt"

echo "Output: $output"

python3 processing_engine.py \
    --corpus=$corpus \
    --tokenizer_path=/tokenizer \
    --n_shards=$SLURM_NPROCS \
    --shard_idx=$SLURM_PROCID \
    --max_seq_len=$seq_length \
    > $output