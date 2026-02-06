from argparse import ArgumentParser

from nemo.collections.nlp.data.language_modeling.megatron import MMapIndexedDataset
from transformers import AutoTokenizer


def print_example(example, tokenizer):
    print(50*"-")
    print("Tokens:", example)
    print("Starts with bos:", example[0] == tokenizer.bos_token_id)
    print("Ends with eos:", example[-1] == tokenizer.eos_token_id)
    text = tokenizer.decode(example)
    print("Text:", text)
    print(50*"-")


def get_few_examples(data_path, tokenizer_path, n=10):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data = MMapIndexedDataset(data_path)

    for example in data[:n]:
        print_example(example, tokenizer)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the mmap data file."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/tokenizer/gemma2_tokenizer"
    )
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    get_few_examples(args.data_path, args.tokenizer_path)
