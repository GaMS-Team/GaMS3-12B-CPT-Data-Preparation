from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
import sys

from document_merger import DocumentMerger, SeparateDocumentMerger, MetafidaDocumentMerger, ShardDocumentMerger


def run_merge(corpus, max_seq_len, n_shards, shard_idx):
    if corpus == "DGT":
        input_path = os.path.join("/data", "parallel_corpora_processed", "DGT")
        output_path = os.path.join("/data", "parallel_corpora_merged", "DGT.jsonl")
    elif corpus == "cc_news":
        input_path = os.path.join("/data", "parallel_corpora_processed", "cc_news")
        output_path = os.path.join("/data", "parallel_corpora_merged", "cc_news.jsonl")
    elif corpus == "cc_stories":
        input_path = os.path.join("/data", "parallel_corpora_processed", "cc_stories")
        output_path = os.path.join("/data", "parallel_corpora_merged", "cc_stories.jsonl")
    elif corpus == "macocu":
        input_path = os.path.join("/data", "parallel_corpora_processed", "MaCoCu")
        output_path = os.path.join("/data", "parallel_corpora_merged", "macocu.jsonl")
    elif corpus == "kas":
        input_path = os.path.join("/data", "parallel_corpora_processed", "KAS")
        output_path = os.path.join("/data", "parallel_corpora_merged", "KAS.jsonl")
    elif corpus == "wikipedia_sl":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_sl")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_sl.jsonl")
    elif corpus == "wikipedia_en":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_en")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_en.jsonl")
    elif corpus == "kas_20":
        input_path = os.path.join("/data", "slovene_corpora_processed", "KAS")
        output_path = os.path.join("/data", "slovene_corpora_merged", "KAS.jsonl")
    elif corpus == "wikipedia_hr":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_hr")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_hr.jsonl")
    elif corpus == "wikipedia_bs":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_bs")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_bs.jsonl")
    elif corpus == "wikipedia_sr_latin":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_sr_latin")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_sr_latin.jsonl")
    elif corpus == "wikipedia_en_sl_translated":
        input_path = os.path.join("/data", "wikipedia_processed", "wikipedia_en_sl_translated")
        output_path = os.path.join("/data", "wikipedia_merged", "wikipedia_en_sl_translated.jsonl")
    elif corpus == "nemotron_pretraining_code":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-Pretraining-Code")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_pretraining_code.jsonl")
    elif corpus == "nemotron_diverse_qa":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-CC-DiverseQA")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_diverse_qa.jsonl")
    elif corpus == "nemotron_high_quality":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-High-Quality-Synthetic")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_high_quality.jsonl")
    elif corpus == "nemotron_pretraining_sft":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-Pretraining-SFT")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_pretraining_sft.jsonl")
    elif corpus == "nemotron_math_4_plus":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-CC-Math-4plus")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_math_4_plus.jsonl")
    elif corpus == "nemotron_math_3":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-CC-Math-3")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_math_3.jsonl")
    elif corpus == "nemotron_math_4_plus_translated":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-CC-Math-4plus_translated")
        output_path = os.path.join("/data", "nemotron_merged", "math_sl", "nemotron_math_4_plus_translated.jsonl")
    elif corpus == "nemotron_pretraining_sft_translated":
        input_path = os.path.join("/data", "nemotron_processed", "Nemotron-Pretraining-SFT_translated")
        output_path = os.path.join("/data", "nemotron_merged", "nemotron_pretraining_sft_translated.jsonl")
    elif corpus == "sodna_praksa":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_legal", "sodna_praksa")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_legal", "sodna_praksa.jsonl")
    elif corpus == "usrs":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_legal", "usrs")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_legal", "usrs.jsonl")
    elif corpus == "ul_uredbeni":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_legal", "uradni_list", "uredbeni")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_legal", "ul_uredbeni.jsonl")
    elif corpus == "ul_razglasni":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_legal", "uradni_list", "razglasni")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_legal", "ul_razglasni.jsonl")
    elif corpus == "pisrs":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_legal", "PISRS")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_legal", "PISRS.jsonl")
    elif corpus == "nuk_col":
        input_path = os.path.join("/data", "nuk_processed", "col")
        output_path = os.path.join("/data", "nuk_merged", "nuk_col.jsonl")
    elif corpus == "nuk_doc":
        input_path = os.path.join("/data", "nuk_processed", "doc")
        output_path = os.path.join("/data", "nuk_merged", "nuk_doc.jsonl")
    elif corpus == "vemo_med":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_med", "vemomed")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_med", "vemo_med.jsonl")
    elif corpus == "crawled_med":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_med", "crawled")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_med", "crawled_med.jsonl")
    elif corpus == "oss_med":
        input_path = os.path.join("/data", "slovene_corpora_processed", "sl_med", "ossMED")
        output_path = os.path.join("/data", "slovene_corpora_merged", "sl_med", "oss_med.jsonl")
    elif corpus == "classla_base":
        input_path = os.path.join("/data", "slovene_corpora_processed", "CLASSLA_base")
        output_path = os.path.join("/data", "slovene_corpora_merged", "classla_base.jsonl")
    elif corpus == "classla_rewrite":
        input_path = os.path.join("/data", "slovene_corpora_processed", "CLASSLA_rewrite")
        output_path = os.path.join("/data", "slovene_corpora_merged", "classla_rewrite.jsonl")
    elif corpus == "fineweb2":
        input_path = os.path.join("/data", "slovene_corpora_processed", "fineweb2")
        output_path = os.path.join("/data", "slovene_corpora_merged", "fineweb2.jsonl")
    elif corpus == "fineweb2_edu":
        input_path = os.path.join("/data", "slovene_corpora_processed", "fineweb2_edu")
        output_path = os.path.join("/data", "slovene_corpora_merged", "fineweb2_edu.jsonl")
    elif corpus == "finepdfs_slv":
        input_path = os.path.join("/data", "slovene_corpora_processed", "finepdfs_slv")
        output_path = os.path.join("/data", "slovene_corpora_merged", "finepdfs_slv.jsonl")
    elif corpus == "finepdfs_hrv":
        input_path = os.path.join("/data", "hbs_corpora_processed", "finepdfs_hrv")
        output_path = os.path.join("/data", "hbs_corpora_merged", "finepdfs_hrv.jsonl")
    elif corpus == "finepdfs_bos":
        input_path = os.path.join("/data", "hbs_corpora_processed", "finepdfs_bos")
        output_path = os.path.join("/data", "hbs_corpora_merged", "finepdfs_bos.jsonl")
    elif corpus == "finepdfs_srp":
        input_path = os.path.join("/data", "hbs_corpora_processed", "finepdfs_srp")
        output_path = os.path.join("/data", "hbs_corpora_merged", "finepdfs_srp.jsonl")
    elif corpus == "omf":
        input_path = os.path.join("/data", "slovene_corpora_processed", "omf")
        output_path = os.path.join("/data", "slovene_corpora_merged", "omf.jsonl")
    elif corpus == "presek":
        input_path = os.path.join("/data", "slovene_corpora_processed", "presek")
        output_path = os.path.join("/data", "slovene_corpora_merged", "presek.jsonl")
    else:
        raise ValueError("Unsupported corpus", corpus)

    if n_shards == 1:
        merger = DocumentMerger(input_path, max_seq_len)
    else:
        merger = ShardDocumentMerger(input_path, max_seq_len, n_shards=n_shards, shard_idx=shard_idx)
    document_list = merger.minimize_documents()

    print("Writing documents ...")
    if n_shards > 1:
        output_dir = output_path.removesuffix(".jsonl")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{shard_idx}.jsonl")
    f_out = open(output_path, "w")
    for i, document in enumerate(tqdm(document_list, file=sys.stdout)):
        f_out.write(json.dumps({"id": i, "text": document}) + "\n")
    f_out.close()
    print("Done!")


def run_separate_merge(corpus, max_seq_len):
    if corpus == "metafida":
        input_path = os.path.join("/data", "slovene_corpora_processed", "Metafida")
        output_path = os.path.join("/data", "slovene_corpora_merged", "Metafida")

        merger = MetafidaDocumentMerger(input_path, max_seq_len)
    else:
        if corpus == "nuk_clanki":
            input_path = os.path.join("/data", "nuk_processed", "clanki")
            output_path = os.path.join("/data", "nuk_merged", "clanki")
        elif corpus == "nuk_casopisje":
            input_path = os.path.join("/data", "nuk_processed", "casopisje")
            output_path = os.path.join("/data", "nuk_merged", "casopisje")
        elif corpus == "nuk_knjige":
            input_path = os.path.join("/data", "nuk_processed", "knjige")
            output_path = os.path.join("/data", "nuk_merged", "knjige")
        elif corpus == "trendi":
            input_path = os.path.join("/data", "slovene_corpora_processed", "Trendi")
            output_path = os.path.join("/data", "slovene_corpora_merged", "Trendi")
        elif corpus == "kas_extension":
            input_path = os.path.join("/data", "slovene_corpora_processed", "KAS_Extension")
            output_path = os.path.join("/data", "slovene_corpora_merged", "KAS_Extension")

        merger = SeparateDocumentMerger(input_path, max_seq_len)

    os.makedirs(output_path, exist_ok=True)

    subcorpora = merger.get_corpora()
    for subcorpus in subcorpora:
        if corpus == "metafida" and subcorpus == "classlawiki_sl":
            print("Skipping Classla Wikipedia as newer Wikipedia dump was obtained")
            continue

        subcorpus_split = subcorpus.split(os.path.sep)
        if len(subcorpus_split) > 1:
            os.makedirs(os.path.join(output_path, *subcorpus_split[:-1]), exist_ok=True)

        print(f"Merging {subcorpus} ...")
        document_list = merger.minimize_documents(subcorpus)
        print("Writing documents ...")
        if len(subcorpus_split) > 1:
            f_out = open(os.path.join(output_path, f"{subcorpus}.jsonl"), "w")
        else:
            f_out = open(os.path.join(output_path, f"{corpus}_{subcorpus}.jsonl"), "w")
        for i, document in enumerate(tqdm(document_list)):
            f_out.write(json.dumps({"id": i, "text": document}) + "\n")
        f_out.close()

    print("Done!")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Name of the corpus to process."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="Maximum length of merged documents."
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="Number of processes to split the data merging to."
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of the process in shard processing."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.corpus in ["metafida", "nuk_clanki", "nuk_casopisje", "nuk_knjige", "trendi", "kas_extension"]:
        run_separate_merge(args.corpus, args.max_seq_len)
    else:
        run_merge(args.corpus, args.max_seq_len, args.n_shards, args.shard_idx)
