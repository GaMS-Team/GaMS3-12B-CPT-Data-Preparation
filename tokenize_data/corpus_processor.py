import os
import json
import warnings
import random
import math
from datetime import datetime

from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from nltk import PunktSentenceTokenizer


def get_slovene_date(dt):
    day_map = {
        "Monday": "Ponedeljek",
        "Tuesday": "Torek",
        "Wednesday": "Sreda",
        "Thursday": "Četrtek",
        "Friday": "Petek",
        "Saturday": "Sobota",
        "Sunday": "Nedelja"
    }

    month_map = {
        "January": "januar",
        "February": "februar",
        "March": "marec",
        "April": "april",
        "May": "maj",
        "June": "junij",
        "July": "julij",
        "August": "avgust",
        "September": "september",
        "October": "oktober",
        "November": "november",
        "December": "december"
    }

    day = dt.strftime("%A")
    month = dt.strftime("%B")
    day_number = dt.strftime("%d")
    if day_number.startswith("0"):
        day_number = day_number[1:]
    year = dt.strftime("%Y")

    return f"{day_map[day]}, {day_number}. {month_map[month]} {year}"


class CorpusProcessor:
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len) -> None:
        self.input_path = None
        self.output_path = os.path.join(data_dir, "tokenized_corpora", tokenizer_name)
        self.size = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def process_document(self, content):
        pass

    def get_data_iter(self, file_idx):
        pass

    def load_data(self):
        pass

    def get_size(self, i):
        return self.size[i]

    def chop_document(self, document_units, delimiter, prefix=""):
        chopped_document = []
        delimiter_len = len(self.tokenizer.tokenize(delimiter))
        prefix_len = len(self.tokenizer.tokenize(prefix))
        curr_text = ""
        curr_len = 0
        for unit in document_units:
            tokens = self.tokenizer.tokenize(unit)
            token_len = len(tokens)

            # Truncate the unit to fit in the sequence length - no need for EOS token, but leave the space for BOS
            if token_len + prefix_len >= self.max_seq_len:
                tokens = tokens[:self.max_seq_len - 1 - prefix_len]
                unit = self.tokenizer.convert_tokens_to_string(tokens)
                token_len = self.max_seq_len - 1 - prefix_len

            if curr_text == "":
                curr_text = prefix + unit
                curr_len = token_len + prefix_len + 1
                continue

            combined_len = curr_len + delimiter_len + token_len
            if combined_len > self.max_seq_len:
                tokenized_text = self.tokenizer.encode(curr_text)
                if len(tokenized_text) < self.max_seq_len:
                    tokenized_text.append(self.tokenizer.eos_token_id)
                chopped_document.append(tokenized_text)
                curr_text = prefix + unit
                curr_len = token_len + prefix_len + 1
            else:
                curr_text += delimiter + unit
                curr_len = combined_len

        # Add the last document
        tokenized_text = self.tokenizer.encode(curr_text)
        if len(tokenized_text) < self.max_seq_len:
            tokenized_text.append(self.tokenizer.eos_token_id)
        chopped_document.append(tokenized_text)

        # Final check that everything was ok
        assert all([len(tokens) <= self.max_seq_len for tokens in
                    chopped_document]), f"All document lengths must be smaller than {self.max_seq_len}"

        return chopped_document

    def dataset_chop_fn(self):
        pass

    def map_dataset(self, workers):
        return self.data.map(self.dataset_chop_fn(), num_proc=workers)


class DGTProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "parallel_corpora", "DGT")
        self.output_path = os.path.join(self.output_path, "parallel_corpora_processed", "DGT")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []
        metadata = {"file": content["file"]}

        def process_lang(lang):
            lang_text = content[lang].strip()
            if lang_text == "":
                return
            document_units = lang_text.split("\n")
            texts = self.chop_document(document_units, delimiter="\n")
            for i, text in enumerate(texts):
                new_document = metadata.copy()
                new_document["lang"] = lang
                new_document["part"] = i + 1
                new_document["text"] = text
                new_documents.append(new_document)

        for lang in ["en", "sl", "hr"]:
            process_lang(lang)

        return new_documents

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class CCNewsProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "parallel_corpora", "cc_news_translation_corrected")
        self.output_path = os.path.join(self.output_path, "parallel_corpora_processed", "cc_news")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        en_paragraphs = content["text"].replace("\n\n", "\n").split("\n")
        sl_paragraphs = content["text_sl"].replace("\n\n", "\n").split("\n")
        if len(en_paragraphs) != len(sl_paragraphs):
            warnings.warn("Different number of English and Slovene paragraphs.")

        document_units = [en_p + "\n" + sl_p for (en_p, sl_p) in zip(en_paragraphs, sl_paragraphs)]
        # Add the title to the first unit
        document_units[0] = f"# {content['title']}\n# {content['title_sl']}\n\n" + document_units[0]

        meta_info = ["domain", "url"]
        metadata = {key: content[key] for key in meta_info}

        texts = self.chop_document(document_units, delimiter="\n\n")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"cc_news_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class CCStoriesProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "parallel_corpora", "cc_stories_translation_corrected")
        self.output_path = os.path.join(self.output_path, "parallel_corpora_processed", "cc_stories")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        document_units = [content["text"].strip(), content["text_sl"].strip()]

        texts = self.chop_document(document_units, delimiter="\n\n")

        for text in texts:
            new_document = {}
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"cc_stories_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class KASParallelProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "parallel_corpora", "KAS")
        self.output_path = os.path.join(self.output_path, "parallel_corpora_processed", "KAS")

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "src"]
        metadata = {key: content[key] for key in meta_info}

        document_units = [content["sl"].strip(), content["en"].strip()]
        texts = self.chop_document(document_units, delimiter="\n\n")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_idx):
        input_file = os.path.join(self.input_path, "kas.jsonl")
        f_in = open(input_file, "r")
        self.data = [json.loads(line) for line in f_in.readlines()]
        f_in.close()

        output_file = os.path.join(self.output_path, "kas.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class MaCoCuParallelProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "parallel_corpora", "MaCoCu")
        self.output_path = os.path.join(self.output_path, "parallel_corpora_processed", "MaCoCu")
        self.N_DOCS = 251174
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(os.path.join(data_dir, "parallel_corpora_processed"), exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        sentence_tokenizer = PunktSentenceTokenizer()
        en_text = content["en"].strip()
        sl_text = content["sl"].strip()
        if "\n" in en_text or "\n" in sl_text:
            warnings.warn("Document contains more paragraphs.")
        en_sentences = sentence_tokenizer.tokenize(en_text)
        sl_sentences = sentence_tokenizer.tokenize(sl_text)

        meta_info = ["id", "src"]
        metadata = {key: content[key] for key in meta_info}

        def process_language(sentences):
            texts = self.chop_document(sentences, delimiter="\n")
            for i, text in enumerate(texts):
                new_document = metadata.copy()
                new_document["part"] = i + 1
                new_document["text"] = text
                new_documents.append(new_document)

        process_language(en_sentences)
        process_language(sl_sentences)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        print("Loading data ...")
        input_file = os.path.join(self.input_path, "macocu_good.jsonl")
        f_in = open(input_file, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            self.data.append(json.loads(line))

        f_in.close()
        print("Data loaded!")

        output_file = os.path.join(self.output_path, f"macocu_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class KASProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "slovene_corpora")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "KAS")
        self.N_DOCS = 82129
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        meta_info = ["id"]
        metadata = {key: content[key] for key in meta_info}

        document_units = content["text"].split("\n")
        texts = self.chop_document(document_units, delimiter="\n\n")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        input_file = os.path.join(self.input_path, "kas_gr_math.jsonl")
        f_in = open(input_file, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            self.data.append(json.loads(line))

        f_in.close()
        print("Data loaded!")
        f_in.close()

        output_file = os.path.join(self.output_path, f"kas_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class MetafidaProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "slovene_corpora")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "Metafida")
        self.N_DOCS = 15294373
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(os.path.join(data_dir, "slovene_corpora_processed"), exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content):
        new_documents = []

        skip_corpora = ["dgt15_sl"]
        title_check_corpora = ["classlawiki_sl", "dsi", "konji", "korp", "lemonde_sl", "maks"]
        title_and_author_corpora = ["eltec_slv", "gfida20_dedup", "imp", "ispac_sl", "maj68", "rsdo5", "sbsj",
                                    "trans5_sl"]
        title_and_year_corpora = ["siparl30"]
        text_corpora = ["gos20", "janes_blog", "janes_forum", "janes_news", "janes_norm30", "janes_tweet", "janes_wiki",
                        "jaslo_sl", "jezkor", "kost10_orig", "prilit", "slwac", "solar30_orig", "suss", "tweet_sl",
                        "vayna"]
        special_corpora = ["filmi"]

        corpus = content["corpus_id"]

        if corpus in skip_corpora:
            return new_documents

        meta_info = ["corpus_id", "id", "corpus"]
        metadata = {key: content[key] for key in meta_info}
        document_units = content["text"].split("\n")

        text_prefix = ""
        if corpus in title_check_corpora:
            first_paragraph = document_units[0]
            text_prefix = "# "
            if "title" in content and content["title"] is not None and content[
                "title"].lower() != first_paragraph.lower():
                text_prefix = f"# {content['title']}\n\n"

        elif corpus in title_and_author_corpora:
            title_prefix = f"# {content['title']}\n"
            author_prefix = f"*Avtor: {content['author']}*\n\n"
            text_prefix = title_prefix + author_prefix

        elif corpus in title_and_year_corpora:
            title_prefix = f"# {content['title']}\n"
            year_prefix = f"*Leto: {content['year']}*\n\n"
            text_prefix = title_prefix + year_prefix

        elif corpus in special_corpora:
            title_prefix = f"# Kritika filma {content['title']}\n"
            author_prefix = f"*Avtor kritike: {content['author']}*\n\n"
            text_prefix = title_prefix + author_prefix

        elif corpus not in text_corpora:
            raise ValueError("Invalid corpus", corpus)

        texts = self.chop_document(document_units, delimiter="\n", prefix=text_prefix)

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        print("Loading data ...")
        input_file = os.path.join(self.input_path, "metafida_exact_dedup.jsonl")
        f_in = open(input_file, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            self.data.append(json.loads(line))

        f_in.close()
        print("Data loaded!")

        output_file = os.path.join(self.output_path, f"metafida_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class WikipediaProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.language = language
        self.input_path = os.path.join(data_dir, "wikipedia", f"wikipedia_{self.language}", "train")
        self.output_path = os.path.join(self.output_path, "wikipedia_processed", f"wikipedia_{self.language}")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def process_document(self, content, removed_headers):
        new_documents = []

        meta_info = ["id", "url"]
        metadata = {key: content[key] for key in meta_info}

        # Remove unnecessary footers
        text = content["text"]
        for header in removed_headers:
            text = text.split(header)[0].rstrip()
        document_units = [unit for unit in text.split("\n\n") if unit.strip() != ""]
        # Add title
        title_prefix = f"# {content['title']}\n\n"
        texts = self.chop_document(document_units, delimiter="\n\n", prefix=title_prefix)

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        print("Data loaded!")

        output_file = os.path.join(self.output_path, f"wikipedia_{self.language}_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class EnglishWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="en")

    def process_document(self, content):
        removed_headers = ["## References", "## Reference", "## External links", "## See also"]
        return super().process_document(content, removed_headers)


class SloveneWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="sl")

    def process_document(self, content):
        removed_headers = ["## Glej tudi", "## Zunanje povezave", "## Sklici"]
        return super().process_document(content, removed_headers)


class CroatianWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="hr")

    def process_document(self, content):
        removed_headers = ["## Vidi još", "## Vanjske poveznice", "## Izvori"]
        return super().process_document(content, removed_headers)


class BosnianWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="bs")

    def process_document(self, content):
        removed_headers = ["## Također pogledajte", "## Vanjski linkovi", "## Reference"]
        return super().process_document(content, removed_headers)


class SerbianWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="sr_latin")
        self.input_path = os.path.join(data_dir, "wikipedia", "wikipedia_sr_latin")

    def process_document(self, content):
        removed_headers = ["## Vidi još", "## Spoljašnje veze", "## Reference"]
        return super().process_document(content, removed_headers)


class TranslatedWikipediaProcessor(WikipediaProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, seed=5) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="en_sl_translated")
        self.input_path = os.path.join(data_dir, "wikipedia", "wikipedia_en_sl_translated")
        self.rng = random.Random(seed)

    def process_document(self, content):
        en_removed_headers = ["## References", "## Reference", "## External links", "## See also"]
        sl_removed_headers = ["## Reference", "## Referenca", "## Zunanje povezave", "## Glej tudi", "## Sklici"]

        new_documents = []

        meta_info = ["id", "url"]
        metadata = {key: content[key] for key in meta_info}

        # Remove unnecessary footers
        en_text = content["text"]
        for header in en_removed_headers:
            en_text = en_text.split(header)[0].rstrip()
        en_document_units = [unit for unit in en_text.split("\n\n") if unit.strip() != ""]

        sl_text = content["text_sl"]
        for header in sl_removed_headers:
            sl_text = sl_text.split(header)[0].rstrip()
        sl_document_units = [unit for unit in sl_text.split("\n\n") if unit.strip() != ""]

        if self.rng.random() < 0.5:
            document_units = en_document_units + sl_document_units
            title_prefix = f"# {content['title']}\n\n"
        else:
            document_units = sl_document_units + en_document_units
            title_prefix = ""

        texts = self.chop_document(document_units, delimiter="\n\n", prefix=title_prefix)

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class NemotronWholeProcessor(CorpusProcessor):
    def load_data(self, data_index, corpus_name):
        print("Loading data ...")
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"{corpus_name}_{data_index}.jsonl")

        return [output_file]

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def get_size(self, i):
        return len(self.data)

    def process_document(self, content):
        new_document = content.copy()
        text = self.chop_document([content["text"]], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            split_doc = self.chop_document([example["text"]], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class NemotronPretrainingCodeProcessor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-Pretraining-Code")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-Pretraining-Code")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_pretraining_code")


class NemotronPretrainingSFTProcessor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-Pretraining-SFT")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-Pretraining-SFT")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_pretraining_sft")


class NemotronDiverseQAProcessor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-CC-DiverseQA")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-CC-DiverseQA")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_diverse_qa")


class NemotronHighQualityProcessor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-High-Quality-Synthetic")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-High-Quality-Synthetic")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_high_quality")


class NemotronMath4PlusProcessor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-CC-Math-4plus")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-CC-Math-4plus")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_math_4_plus")


class NemotronMath3Processor(NemotronWholeProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-CC-Math-3")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-CC-Math-3")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_math_3")


class NemotronTranslatedProcessor(CorpusProcessor):
    def load_data(self, data_index, corpus_name):
        print("Loading data ...")
        subdirs = os.listdir(self.input_path)
        subdirs.sort()
        subdirs = subdirs[data_index::self.n_shards]
        datasets = [load_from_disk(os.path.join(self.input_path, subdir)) for subdir in subdirs]
        if len(datasets) > 1:
            self.data = concatenate_datasets(datasets)
        else:
            self.data = datasets[0]

        output_file = os.path.join(self.output_path, f"{corpus_name}_{data_index}.jsonl")

        return [output_file]

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def get_size(self, i):
        return len(self.data)

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "metadata"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text_sl"]
        document_units = [unit for unit in text.split("\n\n") if unit.strip() != ""]
        texts = self.chop_document(document_units, delimiter="\n\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class NemotronMath4PlusTranslatedProcessor(NemotronTranslatedProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-CC-Math-4plus_translated")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-CC-Math-4plus_translated")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_math_4_plus_translated")


class NemotronPretrainingSFTTranslatedProcessor(NemotronTranslatedProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nemotron_selected", "Nemotron-Pretraining-SFT_translated")
        self.output_path = os.path.join(self.output_path, "nemotron_processed", "Nemotron-Pretraining-SFT_translated")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        return super().load_data(data_index, corpus_name="nemotron_pretraining_sft_translated")


class NUKProcessor(CorpusProcessor):
    def process_document(self, content):
        new_documents = []

        meta_info = ["corpus", "subcorpus", "filename", "ocr_model"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n\n") if unit.strip() != ""]
        texts = self.chop_document(document_units, delimiter="\n\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class NukClankiProcessor(NUKProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nuk_filtered", "clanki")
        self.output_path = os.path.join(self.output_path, "nuk_processed", "clanki")
        self.n_shards = n_shards

        os.makedirs(os.path.join(data_dir, "nuk_processed"), exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class NukKmetijskeProcessor(NukClankiProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "clanki_year", "kmetijske_in_rokodelske_novice")

    def load_data(self, shard_idx):
        output_files = super().load_data(shard_idx)
        new_output_files = []
        for file in output_files:
            file_split = file.split(os.path.sep)
            file_split[-1] = "kmetijske_" + file_split[-1]
            new_output_files.append(os.path.sep.join(file_split))

        return new_output_files


class NukNasaSodobnostProcessor(NukClankiProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "clanki_year", "nasa_sodobnost")

    def load_data(self, shard_idx):
        output_files = super().load_data(shard_idx)
        new_output_files = []
        for file in output_files:
            file_split = file.split(os.path.sep)
            file_split[-1] = "nasa_sodobnost_" + file_split[-1]
            new_output_files.append(os.path.sep.join(file_split))

        return new_output_files


class NukNoviSvetProcessor(NukClankiProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "clanki_year", "novi_svet")

    def load_data(self, shard_idx):
        output_files = super().load_data(shard_idx)
        new_output_files = []
        for file in output_files:
            file_split = file.split(os.path.sep)
            file_split[-1] = "novi_svet_" + file_split[-1]
            new_output_files.append(os.path.sep.join(file_split))

        return new_output_files


class NukSodobnostProcessor(NukClankiProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "clanki_year", "sodobnost")

    def load_data(self, shard_idx):
        output_files = super().load_data(shard_idx)
        new_output_files = []
        for file in output_files:
            file_split = file.split(os.path.sep)
            file_split[-1] = "sodobnost_" + file_split[-1]
            new_output_files.append(os.path.sep.join(file_split))

        return new_output_files


class NukCasopisjeProcessor(NUKProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "casopisje")
        self.output_path = os.path.join(self.output_path, "nuk_processed", "casopisje")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class NukKnjigeProcessor(NUKProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "knjige")
        self.output_path = os.path.join(self.output_path, "nuk_processed", "knjige")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class NukColProcessor(NUKProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "col")
        self.output_path = os.path.join(self.output_path, "nuk_processed", "col")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class NukDocProcessor(NUKProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "nuk_cleaned", "doc")
        self.output_path = os.path.join(self.output_path, "nuk_processed", "doc")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files


class SodnaPraksaProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "sl_legal", "sodnapraksa", "sp_courts.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "sl_legal", "sodna_praksa")
        self.n_shards = n_shards
        self.N_DOCS = 216220
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        f_in = open(self.input_path, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            example = json.loads(line)
            keys_to_keep = ["sodisce", "oddelek", "datum_odlocbe", "institut"]
            example["metadata"] = {key: example["metadata"][key] for key in keys_to_keep}
            self.data.append(example)

        self.data = Dataset.from_list(self.data)
        output_file = os.path.join(self.output_path, f"sodnapraksa_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def process_document(self, content):
        new_document = content.copy()
        document_text = f"# {content['metadata']['sodisce']} - {content['metadata']['oddelek']}\n\n"
        document_text += f"*Datum: {content['metadata']['datum_odlocbe']}*\nKljučne besede: {', '.join(content['metadata']['institut'])}\n\n"
        document_text += f"## Jedro\n\n{content['jedro']}\n\n## Izrek\n\n{content['izrek']}\n\n## Obrazložitev\n\n{content['obrazlozitev']}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            document_text = f"# {example['metadata']['sodisce']} - {example['metadata']['oddelek']}\n\n"
            document_text += f"*Datum: {example['metadata']['datum_odlocbe']}*\nKljučne besede: {', '.join(example['metadata']['institut'])}\n\n"
            document_text += f"## Jedro\n\n{example['jedro']}\n\n## Izrek\n\n{example['izrek']}\n\n## Obrazložitev\n\n{example['obrazlozitev']}"
            split_doc = self.chop_document([document_text], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class USRSProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "sl_legal", "usrs", "usrs.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "sl_legal", "usrs")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_dataset("json", data_files=self.input_path)["train"]
        data = data.remove_columns(["dateOfApplication", "dateOfDecision"])
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"usrs_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def process_document(self, content):
        new_document = content.copy()
        document_text = f"# {content['act']}\n\n"
        document_text += f"## Sklep\n\n{content['operationalProvisions']}\n\n"
        document_text += f"## Pravna podlaga\n\n{content['legalBasis']}\n\n"
        document_text += f"## Vsebina\n\n{content['fullText']}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            document_text = f"# {example['act']}\n\n"
            document_text += f"## Sklep\n\n{example['operationalProvisions']}\n\n"
            document_text += f"## Pravna podlaga\n\n{example['legalBasis']}\n\n"
            document_text += f"## Vsebina\n\n{example['fullText']}"
            split_doc = self.chop_document([document_text], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class ULUredbeniProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "sl_legal", "uradnilist", "ul-uredbeni.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "sl_legal", "uradni_list",
                                        "uredbeni")
        self.n_shards = n_shards
        self.N_DOCS = 148008
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        f_in = open(self.input_path, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            example = json.loads(line)
            keys_to_keep = ["id", "score", "ul_num", "title", "basis", "text"]
            example = {key: example[key] for key in keys_to_keep}
            self.data.append(example)

        output_file = os.path.join(self.output_path, f"ul_uredbeni_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def process_document(self, content):
        new_document = content.copy()
        document_text = f"# {content['title']}\n\n{content['text']}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            document_text = f"# {example['title']}\n\n{example['text']}"
            split_doc = self.chop_document([document_text], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class ULRazglasniProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "sl_legal", "uradnilist", "ul-razglasni.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "sl_legal", "uradni_list",
                                        "razglasni")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_dataset("json", data_files=self.input_path)["train"]
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"ul_razglasni_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def process_document(self, content):
        new_document = content.copy()
        core_text = content["text"].removesuffix(content["odered_by"]).strip()
        document_text = f"# {content['title']}\n*{content['type']}*\n\n"
        document_text += f"## Naročnik\n\n{content['odered_by']}\n\n"
        document_text += f"## Besedilo\n\n{core_text}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            core_text = example["text"].removesuffix(example["odered_by"]).strip()
            document_text = f"# {example['title']}\n*{example['type']}*\n\n"
            document_text += f"## Naročnik\n\n{example['odered_by']}\n\n"
            document_text += f"## Besedilo\n\n{core_text}"
            split_doc = self.chop_document([document_text], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class PISRSProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "sl_legal", "PISRS")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "sl_legal", "PISRS")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = []
        for file in os.listdir(self.input_path):
            if not file.endswith(".jsonl") or file in ["PISRS-neveljavni-predpisi.jsonl",
                                                       "PISRS-obsoletni-in-konzumirani-predpisi.jsonl"]:
                continue

            data.append(load_dataset("json", data_files=os.path.join(self.input_path, file))["train"])

        data = concatenate_datasets(data)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"PISRS_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def process_document(self, content):
        new_document = content.copy()
        document_text = content['text']
        if "naziv" in content:
            document_text = f"# {content['naziv']}\n\n{document_text}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]

    def dataset_chop_fn(self):
        def process_document(example):
            document_text = example['text']
            if "naziv" in example:
                document_text = f"# {example['naziv']}\n\n{document_text}"
            split_doc = self.chop_document([document_text], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class TrendiProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "slovene_corpora", "Trendi")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "Trendi")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def process_document(self, content):
        new_document = content.copy()
        en_date = datetime.strptime(content["date"].removesuffix(" topic="), "%Y-%m-%d")
        sl_date = get_slovene_date(en_date)
        if "url" in content:
            header = f"# {content['title']}\n\n*Datum: {sl_date}*\n*Vir: {content['source']}*\n*URL: {content['url']}*"
        else:
            header = f"# {content['title']}\n\n*Datum: {sl_date}*\n*Vir: {content['source']}*"
        document_text = f"{header}\n\n{content['text']}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]


class FinePDFsProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.language = language
        if self.language == "slv":
            base_dir = "slovene_corpora"
        else:
            base_dir = "hbs_corpora"
        self.input_path = os.path.join(data_dir, base_dir, f"finepdfs_{self.language}")
        self.output_path = os.path.join(self.output_path, f"{base_dir}_processed", f"finepdfs_{self.language}")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"finepdfs_{self.language}_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def dataset_chop_fn(self):
        def process_document(example):
            document_units = example["text"].split("\n\n")

            dt = datetime.fromisoformat(example["date"].replace("Z", "+00:00"))
            header = f"*{get_slovene_date(dt)}*"
            document_units = [header] + document_units

            split_doc = self.chop_document(document_units, delimiter="\n\n")
            example["split_parts"] = split_doc

            return example

        return process_document

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "dump", "url", "date", "file_path"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n") if unit.strip() != ""]
        dt = datetime.fromisoformat(content["date"].replace("Z", "+00:00"))
        header = f"*{get_slovene_date(dt)}*"
        document_units = [header] + document_units
        texts = self.chop_document(document_units, delimiter="\n\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class SloveneFinePDFsProcessor(FinePDFsProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="slv")


class CroatianFinePDFsProcessor(FinePDFsProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="hrv")


class BosnianFinePDFsProcessor(FinePDFsProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="bos")


class SerbianFinePDFsProcessor(FinePDFsProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, language="srp")


class FineWebProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "fineweb_2_curated", "slv_Latn")
        self.output_path = os.path.join(self.output_path, f"slovene_corpora_processed", "fineweb2")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_from_disk(self.input_path)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"fineweb2_slv_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def dataset_chop_fn(self):
        def process_document(example):
            document_units = example["text"].split("\n")

            dt = datetime.fromisoformat(example["date"].replace("Z", "+00:00"))
            header = f"*{get_slovene_date(dt)}*"
            document_units = [header] + document_units

            split_doc = self.chop_document(document_units, delimiter="\n")
            example["split_parts"] = split_doc

            return example

        return process_document

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "dump", "url", "date", "file_path"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n") if unit.strip() != ""]
        dt = datetime.fromisoformat(content["date"].replace("Z", "+00:00"))
        header = f"*{get_slovene_date(dt)}*"
        document_units = [header] + document_units
        texts = self.chop_document(document_units, delimiter="\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class FineWeb2EduProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "fineweb2_edu_3plus", "3plus")
        self.output_path = os.path.join(self.output_path, f"slovene_corpora_processed", "fineweb2_edu")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data_files = [os.path.join(self.input_path, filename) for filename in os.listdir(self.input_path) if
                      filename.endswith(".parquet")]
        data_files.sort()
        data_files = data_files[data_index::self.n_shards]
        print("Data files:", data_files)
        self.data = load_dataset("parquet", data_files=data_files)["train"]

        output_file = os.path.join(self.output_path, f"fineweb2_slv_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def dataset_chop_fn(self):
        def process_document(example):
            document_units = example["text"].split("\n")

            dt = datetime.fromisoformat(example["date"].replace("Z", "+00:00"))
            header = f"*{get_slovene_date(dt)}*"
            document_units = [header] + document_units

            split_doc = self.chop_document(document_units, delimiter="\n")
            example["split_parts"] = split_doc

            return example

        return process_document

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "dump", "url", "date", "file_path"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n") if unit.strip() != ""]
        dt = datetime.fromisoformat(content["date"].replace("Z", "+00:00"))
        header = f"*{get_slovene_date(dt)}*"
        document_units = [header] + document_units
        texts = self.chop_document(document_units, delimiter="\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class SloveneMedProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, corpus) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.corpus = corpus
        self.input_path = os.path.join(data_dir, "sl-med-1", f"{self.corpus}.jsonl")
        self.output_path = os.path.join(self.output_path, f"slovene_corpora_processed", "sl_med", self.corpus)
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_index):
        print("Loading data ...")
        data = load_dataset("json", data_files=self.input_path)["train"]
        data = data.filter(self.filter_fn)
        self.N_DOCS = len(data)
        self.shard_size = int(math.ceil(self.N_DOCS / self.n_shards))
        start_index = data_index * self.shard_size
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        print("Number of documents:", self.N_DOCS)
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        self.data = data.select(range(start_index, end_index))

        output_file = os.path.join(self.output_path, f"{self.corpus}_{data_index}.jsonl")

        return [output_file]

    def filter_fn(self, example):
        return example["text"].strip() != ""

    def dataset_chop_fn(self):
        def process_document(example):
            split_doc = self.chop_document([example["text"].strip()], delimiter="")
            example["split_parts"] = split_doc

            return example

        return process_document


class CrawledMedProcessor(SloveneMedProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, corpus="crawled")

    def filter_fn(self, example):
        return example["source"] == "ZV" and example["text"].strip() != ""


class VemoMedProcessor(SloveneMedProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, corpus="vemomed")


class OSSMedProcessor(SloveneMedProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards, corpus="ossMED")


class CLASSLAPocessor(CorpusProcessor):
    def process_document(self, content):
        metadata_keys = ["crawl_year", "domain", "id", "url"]
        new_document = {key: content[key] for key in metadata_keys}
        document_text = f"# {content['title']}\n{content['text'].strip()}"
        text = self.chop_document([document_text], delimiter="")
        new_document["text"] = text[0]

        return [new_document]


class CLASSLABaseProcessor(CLASSLAPocessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "slovene_corpora", "CLASSLA-web-2024_sl_base.jsonl")
        self.output_path = os.path.join(self.output_path, f"slovene_corpora_processed", "CLASSLA_base")
        self.n_shards = n_shards

        self.N_DOCS = 3286597
        self.shard_size = int(math.ceil(self.N_DOCS / n_shards))

        os.makedirs(self.output_path, exist_ok=True)

    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_index):
        f_in = open(self.input_path, "r")
        start_index = data_index * self.shard_size
        print("Start index:", start_index)
        print("Shard_size:", self.shard_size)
        end_index = min(start_index + self.shard_size, self.N_DOCS)
        self.data = []
        for i, line in enumerate(f_in):
            if i < start_index:
                continue
            if i == end_index:
                break

            self.data.append(json.loads(line))

        f_in.close()
        print("Data loaded!")
        f_in.close()

        output_file = os.path.join(self.output_path, f"classla_base_{data_index}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)


class CLASSLARewriteProcessor(CLASSLAPocessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards):
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "slovene_corpora", "CLASSLA-web-2024_rewrite")
        self.output_path = os.path.join(self.output_path, f"slovene_corpora_processed", "CLASSLA_rewrite")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, shard_idx):
        jsonl_files = [file for file in os.listdir(self.input_path) if file.endswith(".jsonl")]
        jsonl_files.sort()
        jsonl_files = jsonl_files[shard_idx::self.n_shards]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content


class KASExtensionProcessor(CorpusProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "KAS_Tomaz", "OCR")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "KAS_Extension")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, shard_idx):
        jsonl_files = []
        subdirs = os.listdir(self.input_path)
        for subdir in subdirs:
            for file in os.listdir(os.path.join(self.input_path, subdir)):
                if file.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(subdir, file))
                    os.makedirs(os.path.join(self.output_path, subdir), exist_ok=True)
        jsonl_files.sort()
        file_idcs = list(range(shard_idx, len(jsonl_files), self.n_shards))
        jsonl_files = [jsonl_files[idx] for idx in file_idcs]
        input_files = [os.path.join(self.input_path, file) for file in jsonl_files]
        output_files = [os.path.join(self.output_path, file) for file in jsonl_files]
        self.data = []
        for input_path in input_files:
            f_in = open(input_path, "r")
            content = f_in.readlines()
            self.data.append(content)
            self.size.append(len(content))
            f_in.close()

        return output_files

    def get_data_iter(self, file_idx):
        for line in self.data[file_idx]:
            content = json.loads(line)

            yield content

    def process_document(self, content):
        new_documents = []

        meta_info = ["id", "avtor", "celotno_besedilo_link", "subcorpus"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n\n") if unit.strip() != ""]
        texts = self.chop_document(document_units, delimiter="\n\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class MathJournalProcessor(CorpusProcessor):
    def get_data_iter(self, file_idx):
        for document in self.data:
            yield document

    def load_data(self, data_idx, corpus_name):
        f_in = open(self.input_path, "r")
        self.data = [json.loads(line) for line in f_in.readlines()]
        f_in.close()

        output_file = os.path.join(self.output_path, f"{corpus_name}.jsonl")

        return [output_file]

    def get_size(self, i):
        return len(self.data)

    def process_document(self, content):
        new_documents = []

        meta_info = ["filename"]
        metadata = {key: content[key] for key in meta_info}

        text = content["text"]
        document_units = [unit for unit in text.split("\n\n") if unit.strip() != ""]
        texts = self.chop_document(document_units, delimiter="\n\n", prefix="")

        for i, text in enumerate(texts):
            new_document = metadata.copy()
            new_document["part"] = i + 1
            new_document["text"] = text
            new_documents.append(new_document)

        return new_documents


class OMFProcessor(MathJournalProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "omf", "Markdown", "omf.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "omf")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_idx):
        return super().load_data(data_idx, corpus_name="omf")


class PresekProcessor(MathJournalProcessor):
    def __init__(self, data_dir, tokenizer_name, tokenizer, max_seq_len, n_shards) -> None:
        super().__init__(data_dir, tokenizer_name, tokenizer, max_seq_len)
        self.input_path = os.path.join(data_dir, "Presek", "Markdown", "presek.jsonl")
        self.output_path = os.path.join(self.output_path, "slovene_corpora_processed", "presek")
        self.n_shards = n_shards

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, data_idx):
        return super().load_data(data_idx, corpus_name="presek")
