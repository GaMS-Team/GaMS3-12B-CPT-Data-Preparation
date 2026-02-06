#!/bin/bash
#SBATCH --job-name=count_tokens
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=count_err.txt

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-CPT-Data-Preparation/count_tokens

# TODO: Add path to your data
DATA_DIR=

CONTAINER_DIR=$WORK_DIR/containers
NEMO_VERSION=25.09
CONTAINER=${CONTAINER_DIR}/nemo_${NEMO_VERSION}.sqsh

corpus_file=$1

# run with singularity
srun \
  --cpu-bind=verbose \
  --output="$SCRIPT_DIR/logs/${corpus_file}.txt" \
  --container-mounts $DATA_DIR:/data,$SCRIPT_DIR:/script \
  --container-workdir /script \
  --container-image $CONTAINER \
  python count_for_corpus.py \
    --data_path=/data/${corpus_file}_text_document
