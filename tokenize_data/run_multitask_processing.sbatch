#!/bin/bash
#SBATCH --job-name=tokenize_corpus
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --output=process_err.txt

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-CPT-Data-Preparation/tokenize_data

# TODO: Add path to the container
CONTAINER_PATH=

# TODO: Add path to your data
DATA_DIR=

corpus=$1
tokenizer_dir=$2
seq_length=$3

srun \
  --cpu-bind=verbose \
  --container-mounts $DATA_DIR:/data,$tokenizer_dir:/tokenizer,$SCRIPT_DIR:/script \
  --container-workdir /script \
  --container-image $CONTAINER_PATH \
  ./process.sh $corpus $seq_length