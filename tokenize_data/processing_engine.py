import sys
from argparse import ArgumentParser
from tqdm import tqdm

from transformers import AutoTokenizer

from corpus_processor import *


def get_corpus_processor(data_dir, corpus, tokenizer_name, tokenizer, max_seq_len, n_shards) -> CorpusProcessor:
    if corpus == "DGT":
        return DGTProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "cc_news":
        return CCNewsProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "cc_stories":
        return CCStoriesProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "KAS":
        return KASParallelProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "macocu":
        return MaCoCuParallelProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "kas_20":
        return KASProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "metafida":
        return MetafidaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_en":
        return EnglishWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_sl":
        return SloveneWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_hr":
        return CroatianWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_bs":
        return BosnianWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_sr_latin":
        return SerbianWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "wikipedia_en_sl_translated":
        return TranslatedWikipediaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_pretraining_code":
        return NemotronPretrainingCodeProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_pretraining_sft":
        return NemotronPretrainingSFTProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_diverse_qa":
        return NemotronDiverseQAProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_high_quality":
        return NemotronHighQualityProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_math_4_plus":
        return NemotronMath4PlusProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_math_3":
        return NemotronMath3Processor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_math_4_plus_translated":
        return NemotronMath4PlusTranslatedProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nemotron_pretraining_sft_translated":
        return NemotronPretrainingSFTTranslatedProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "sodna_praksa":
        return SodnaPraksaProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "usrs":
        return USRSProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "ul_uredbeni":
        return ULUredbeniProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "ul_razglasni":
        return ULRazglasniProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "vemo_med":
        return VemoMedProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "crawled_med":
        return CrawledMedProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "oss_med":
        return OSSMedProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "pisrs":
        return PISRSProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_clanki":
        return NukClankiProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_kmetijske":
        return NukKmetijskeProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_nasa_sodobnost":
        return NukNasaSodobnostProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_novi_svet":
        return NukNoviSvetProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_sodobnost":
        return NukSodobnostProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_casopisje":
        return NukCasopisjeProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_knjige":
        return NukKnjigeProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_col":
        return NukColProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "nuk_doc":
        return NukDocProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "trendi":
        return TrendiProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "finepdfs_slv":
        return SloveneFinePDFsProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "finepdfs_hrv":
        return CroatianFinePDFsProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "finepdfs_bos":
        return BosnianFinePDFsProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "finepdfs_srp":
        return SerbianFinePDFsProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "fineweb2":
        return FineWebProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "fineweb2_edu":
        return FineWeb2EduProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "classla_base":
        return CLASSLABaseProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "classla_rewrite":
        return CLASSLARewriteProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "kas_extension":
        return KASExtensionProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "omf":
        return OMFProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
    if corpus == "presek":
        return PresekProcessor(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)

    raise ValueError("Unsopported corpus", corpus)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus",
        required=True,
        type=str,
        help="Name of the corpus to process."
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        type=str,
        help="Path to the Huggingface tokenizer."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="Max sequence length of the document."
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="Number of shards the data is split into"
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of shard to process"
    )
    parser.add_argument(
        "--hf_dataset",
        action="store_true",
        help="If provided HF datasets map function is used for processing"
    )

    return parser.parse_args()


def run_processing(corpus, tokenizer_name, tokenizer, max_seq_len, shard_idx, n_shards, hf_dataset):
    if hf_dataset:
        corpus_processor = get_corpus_processor("/data", corpus, tokenizer_name, tokenizer, max_seq_len, n_shards=1)
        output_file = corpus_processor.load_data(data_index=0)[0]

        split_data = corpus_processor.map_dataset(n_shards)
        print("Saving data ...")
        metadata_cols = [column for column in split_data.column_names if column != "split_parts"]
        with open(output_file, "w") as f_out:
            for example in tqdm(split_data):
                metadata = {col: example[col] for col in metadata_cols}
                for tokens in example["split_parts"]:
                    output_example = metadata.copy()
                    output_example["text"] = tokens
                    f_out.write(json.dumps(output_example) + "\n")
    else:
        corpus_processor = get_corpus_processor("/data", corpus, tokenizer_name, tokenizer, max_seq_len, n_shards)
        print("Number of shards:", args.n_shards)
        print("Shard index:", args.shard_idx)
        output_files = corpus_processor.load_data(shard_idx)

        for i, output_file in enumerate(output_files):
            f_out = open(output_file, "w")

            print(f"Processing {output_file} ...")

            for content in tqdm(corpus_processor.get_data_iter(i), total=corpus_processor.get_size(i), file=sys.stdout):
                processed_content = corpus_processor.process_document(content)
                for document in processed_content:
                    f_out.write(json.dumps(document, ensure_ascii=False) + "\n")

            f_out.close()

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    run_processing(args.corpus, args.tokenizer_path, tokenizer, args.max_seq_len, args.shard_idx, args.n_shards,
                   args.hf_dataset)
