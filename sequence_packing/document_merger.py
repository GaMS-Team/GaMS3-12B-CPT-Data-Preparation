import os
import json
from tqdm import tqdm
import sys


class DocumentMerger:
    def __init__(self, input_path, max_seq_len):
        self.input_path = input_path
        self.max_seq_len = max_seq_len

        self.documents, self.lens = self.load_data()

    def load_data(self):
        input_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        documents = []
        lens = []

        print("Loading data ...")
        document_idx = 0
        for file in input_files:
            f_in = open(os.path.join(self.input_path, file))
            for line in f_in:
                document = json.loads(line)
                documents.append(document["text"])
                lens.append((document_idx, len(document["text"])))
                document_idx += 1

        return documents, lens

    def minimize_documents(self):
        print("Running merging ...")

        # Sort the lengths in descending order to maximize packing efficiency.
        self.lens.sort(key=lambda el: -el[1])
        print("Maximum length:", self.lens[0][1])

        # List to store the packages.
        packages = []
        capacities = []
        packages_to_check = {}
        n_packages = 0

        for idx, doc_len in tqdm(self.lens, file=sys.stdout):
            # Try to fit the value into an existing package.
            placed = False
            for i in packages_to_check.keys():
                if capacities[i] >= doc_len:
                    packages[i] += self.documents[idx]
                    capacities[i] -= doc_len
                    if capacities[i] == 0:
                        del packages_to_check[i]
                    placed = True
                    break

            # If it doesn't fit into any existing package, create a new one.
            if not placed:
                packages.append(self.documents[idx])
                capacities.append(self.max_seq_len - doc_len)
                if doc_len < self.max_seq_len:
                    packages_to_check[n_packages] = True
                n_packages += 1

        return packages


class SeparateDocumentMerger(DocumentMerger):
    def __init__(self, input_path, max_seq_len):
        self.input_path = input_path
        self.max_seq_len = max_seq_len

        self.corpora_documents, self.corpora_lens = self.load_data()

    def load_data(self):
        input_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        if input_files == []:
            subdirs = os.listdir(self.input_path)
            for subdir in subdirs:
                for file in os.listdir(os.path.join(self.input_path, subdir)):
                    if file.endswith(".jsonl"):
                        input_files.append(os.path.join(subdir, file))
        documents = {}
        lens = {}
        document_idx = {}

        print("Loading data ...")
        for file in input_files:
            corpus = file.removesuffix(".jsonl")
            documents[corpus] = []
            lens[corpus] = []
            document_idx[corpus] = 0
            f_in = open(os.path.join(self.input_path, file))
            for line in f_in:
                document = json.loads(line)
                documents[corpus].append(document["text"])
                lens[corpus].append((document_idx[corpus], len(document["text"])))
                document_idx[corpus] += 1

        return documents, lens

    def get_corpora(self):
        return self.corpora_documents.keys()

    def minimize_documents(self, corpus):
        self.documents = self.corpora_documents[corpus]
        self.lens = self.corpora_lens[corpus]
        return super().minimize_documents()


class MetafidaDocumentMerger(SeparateDocumentMerger):
    def load_data(self):
        input_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        documents = {}
        lens = {}
        document_idx = {}

        print("Loading data ...")
        for file in input_files:
            f_in = open(os.path.join(self.input_path, file))
            for line in f_in:
                document = json.loads(line)
                corpus = document["corpus_id"]
                if corpus not in documents:
                    documents[corpus] = []
                    lens[corpus] = []
                    document_idx[corpus] = 0
                documents[corpus].append(document["text"])
                lens[corpus].append((document_idx[corpus], len(document["text"])))
                document_idx[corpus] += 1

        return documents, lens


class ShardDocumentMerger(DocumentMerger):
    def __init__(self, input_path, max_seq_len, n_shards, shard_idx):
        self.input_path = input_path
        self.max_seq_len = max_seq_len

        self.documents, self.lens = self.load_data(n_shards, shard_idx)

    def load_data(self, n_shards, shard_idx):
        input_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        input_files.sort()
        input_files = input_files[shard_idx::n_shards]
        print(input_files)

        documents = []
        lens = []

        print("Loading data ...")
        document_idx = 0
        for file in input_files:
            f_in = open(os.path.join(self.input_path, file))
            for line in f_in:
                document = json.loads(line)
                documents.append(document["text"])
                lens.append((document_idx, len(document["text"])))
                document_idx += 1

        print("Number of documents", len(documents))

        return documents, lens
