#!/bin/bash
#SBATCH --job-name=merge_corpus
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --output=merge_err.txt

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-CPT-Data-Preparation/sequence_packing

# TODO: Add path to the container
CONTAINER_PATH=

# TODO: Add path to your data
DATA_DIR=

corpus=$1
seq_length=$2

srun \
  --cpu-bind=verbose \
  --container-mounts $DATA_DIR:/data,$SCRIPT_DIR:/script \
  --container-workdir /script \
  --container-image $CONTAINER_PATH \
  ./merge.sh $corpus $seq_len
