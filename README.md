# GaMS3-12B CPT Data Preparation

This repository contains the data preparation pipeline for the Continual Pre-Training (CPT) of the **GaMS3-12B** model. The scripts provided here streamline the process of converting raw datasets into the memory-mapped format required by the **NVIDIA NeMo** framework for efficient training.

## 📋 Pipeline Overview

The data preparation workflow follows a three-step sequential pipeline:

1. **Tokenization**: Raw text data is tokenized and split into chunks that fit within the model's context window.
2. **Sequence Packing**: Tokenized chunks are packed (merged) together to maximize training efficiency.
3. **Indexing**: The packed data is indexed and converted into NeMo-compatible mmap datasets.

Additionally, utilities are provided to inspect the processed data and count tokens.

---

## 📂 Repository Structure

```text
GaMS3-12B-CPT-Data-Preparation/
├── tokenize_data/         # Scripts for tokenizing and chunking raw datasets
├── sequence_packing/      # Scripts for packing tokenized chunks
├── nemo_data_indexing/    # Scripts for creating mmap datasets and indices for NeMo
├── inspect_indexed_data/  # Script to inspect the final mmap data
├── count_tokens/          # Script to count tokens in the final mmap data
└── README.md

```

---

## 🖥️ Code Usage & Infrastructure

This repository is designed for execution on high-performance computing (HPC) environments using the **Slurm** workload manager. Each training stage includes specialized Sbatch scripts tailored to the infrastructure used during development.

### 📦 Environment & Containers

The latest stages (data indexing, token counting and data inspection) rely on **NVIDIA NeMo Framework**. We suggest using official NVIDIA container. We tested our code with version 25.07 (`nvcr.io/nvidia/nemo:25.07`).

The previous stages are optimized for multi-CPU parallelism. Hence, running NeMo container can be overkill or lead to problems if the scripts are not run on GPU optimized nodes. Hence, we suggest setting up local environment for such libraries. The only requirement besides Python is installing `transformers`, `datasets` and `nltk` libraries. 

### 2. Path Setup

Before running any jobs, you **must** configure the file paths in the `sbatch` scripts. Open the scripts in each directory and locate the `TODO` sections at the top:

```bash
# TODO: Add path to the root dir
WORK_DIR=/path/to/your/workspace
# TODO: Add path to the container
CONTAINER_PATH=/path/to/containers/nemo_25.09.sqsh
# TODO: Add path to your data
DATA_DIR=/path/to/raw/data

```

*Ensure these paths are correct for your specific cluster environment.*

---

## 🚀 Usage

### Step 1: Tokenize Data

**Script:** `tokenize_data/run_multitask_processing.sbatch`

Tokenizes the raw corpus and splits it into chunks fitting the sequence length. The script runs several parallel tasks, each tokenizing different parts of dataset.

**Usage:**

```bash
cd tokenize_data
sbatch run_multitask_processing.sbatch <corpus_name> <tokenizer_dir> <seq_length>

```

* `<corpus_name>`: The name of the input corpus file (located in your configured `DATA_DIR`).
* `<tokenizer_dir>`: Path to the directory containing the tokenizer model (HF format).
* `<seq_length>`: Maximum sequence length (e.g., 4096, 8192).

**Alternative**: the data can be tokenized using`tokenize_data/run_hf_processing.sbatch` script, which relies on the parallelism by HF datasets instead of running multiple tasks. To use this script, the corpus processor class must have implemented HF data loading and `dataset_chop_fn` function.

---

### Step 2: Sequence Packing

**Script:** `sequence_packing/run_merge.sbatch`

Merges the tokenized chunks from Step 1 to minimize padding and maximize training efficiency.

**Usage:**

```bash
cd sequence_packing
sbatch run_merge.sbatch <corpus_name> <seq_length>

```

* `<corpus_name>`: The same corpus name used in Step 1.
* `<seq_length>`: Must match the sequence length used in Step 1.

**Alternative**: sequence packing can be performed in a parallel way by running `sequence_packing/run_merge_shards.sbatch` script. This enables faster and more memory efficient sequence packing, but might result in less optimal packing.

---

### Step 3: Data Indexing (NeMo Format)

**Script:** `nemo_data_indexing/create_mmap_tokenized.sbatch`

Creates the final memory-mapped datasets (`.bin` and `.idx`) required by NeMo. This step utilizes `preprocess_data_for_megatron.py` with the `--tokenized-data` flag, as the input is already tokenized.

**Usage:**

```bash
cd nemo_data_indexing
sbatch create_mmap_tokenized.sbatch <input_data_name> <output_prefix> <tokenizer_dir> <seq_length>

```

* `<input_data_name>`: The filename of the packed data from Step 2.
* `<output_prefix>`: The desired prefix for the final output files (e.g., `my_dataset_cpt`).
* `<tokenizer_dir>`: Path to the tokenizer directory.
* `<seq_length>`: The sequence length.

**Corpora split in multiple files**: for corpora that were sequence packed using parallel approach and consists out of multiple JSONL files must be indexed using the `nemo_data_indexing/create_mmap_tokenized_dir.sbatch` script.

---

## 🛠 Validation Utilities

### Inspect Indexed Data

**Script:** `inspect_indexed_data/run_inspection.sbatch`

Decodes and prints a few examples from the final `.bin` files to verify data integrity.

**Usage:**

```bash
cd inspect_indexed_data
sbatch run_inspection.sbatch <corpus_prefix> <tokenizer_dir>

```

* `<corpus_prefix>`: The prefix used in Step 3 (the script expects to find `${corpus_prefix}_text_document`).
* `<tokenizer_dir>`: Path to the tokenizer directory.

### Count Tokens

**Script:** `count_tokens/run_count.sbatch`

Calculates the total number of tokens in the indexed dataset. This is crucial for calculating training epochs and compute budget.

**Usage:**

```bash
cd count_tokens
sbatch run_count.sbatch <corpus_prefix>

```

* `<corpus_prefix>`: The prefix used in Step 3 (the script expects to find `${corpus_prefix}_text_document`).

---

## 🏛️ Acknowledgments

Developed by researchers at the **University of Ljubljana, Faculty of Computer and Information Science**, within the **PoVeJMo** research program.

The project was supported by:

* **ARIS** (Slovenian Research and Innovation Agency).
* **NextGenerationEU**.
* **NVIDIA Sovereign AI initiative**.
* **EuroHPC JU**.
* **SLING** (Slovenian National Supercomputing Network).
* **SLAIF** (Slovenian AI Factory).

---

## **Contact**

**Domen Vreš**  
domen.vres@fri.uni-lj.si

---

## 📄 License

This project is licensed under the **Apache-2.0 licensee**.
