#!/bin/bash
#SBATCH --job-name=tokenize_corpus
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --output=process_hf_err.txt

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

workers=$SLURM_CPUS_PER_TASK
echo "Running preprocessing with $workers workers"

srun \
  --cpu-bind=verbose \
  --output="$SCRIPT_DIR/logs/${corpus}.txt" \
  --container-mounts $DATA_DIR:/data,$tokenizer_dir:/tokenizer,$SCRIPT_DIR:/script \
  --container-workdir /script \
  --container-image $CONTAINER_PATH \
  python3 processing_engine.py \
    --corpus=$corpus \
    --tokenizer_path=/tokenizer \
    --max_seq_len=$seq_length \
    --n_shards=$workers \
    --hf_dataset