from argparse import ArgumentParser
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron import MMapIndexedDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the corpus file."
    )
    return parser.parse_args()


def count_tokens(data_path):
    dataset = MMapIndexedDataset(data_path)
    print("Number of documents:", len(dataset))

    ntokens = 0
    for doc in tqdm(dataset):
        ntokens += len(doc)

    print("Number of tokens:", ntokens)


if __name__=="__main__":
    args = parse_args()
    count_tokens(args.data_path)
