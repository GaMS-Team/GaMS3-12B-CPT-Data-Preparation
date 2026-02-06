#!/bin/bash
#SBATCH --job-name=create_mmap
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=mmap_err.txt

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-CPT-Data-Preparation/nemo_data_indexing

# TODO: Add path to your data
DATA_DIR=

# TODO: Add path to the output dir (where the final data will be saved)
OUTPUT_DIR=

CONTAINER_DIR=$WORK_DIR/containers
NEMO_VERSION=25.09
CONTAINER=${CONTAINER_DIR}/nemo_${NEMO_VERSION}.sqsh

input_data=$1
output_prefix=$2
tokenizer_dir=$3
seq_len=$4

srun \
  --cpu-bind=verbose \
  --output="$SCRIPT_DIR/logs/${input_data}.txt" \
  --container-mounts DATA_DIR:/data,$OUTPUT_DIR:/output,$tokenizer_dir:/tokenizer,$SCRIPT_DIR:/script \
  --container-image CONTAINER \
  --container-workdir /script \
  python preprocess_data_for_megatron.py \
    --input=/data/$input_data \
    --json-keys=text \
    --tokenized-data \
    --tokenizer-library=huggingface \
    --tokenizer-type=/tokenizer \
    --dataset-impl=mmap \
    --output-prefix=/output/$output_prefix \
    --max-seq-len=$seq_len \
    --workers=$SLURM_CPUS_PER_TASK
