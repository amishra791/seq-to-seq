from torch.utils.data import Dataset, DataLoader
import unicodedata
import re
import heapq
import timeit
from functools import partial
from collections.abc import Callable
import numpy as np
import jax

SRC_FILE = "./data/en_5000.txt"
TGT_FILE = "./data/fr_5000.txt"


# Special tokens added to raw text
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def process_line(line: str) -> list[str]:
    normalized_line = unicodedata.normalize("NFKD", line)
    cleaned_line = re.sub("([.,!?'\"])", r" \1 ", normalized_line)
    cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line)
    cleaned_line_list = cleaned_line.split()

    final_line = cleaned_line_list + [EOS_TOKEN]

    return final_line


class TranslationDataset(Dataset):
    @staticmethod
    def _convert_to_tokens(word_seq: list[str], vocab: dict[str, int]) -> list[int]:
        assert UNK_TOKEN in vocab
        token_list = [
            vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in word_seq
        ]
        return token_list

    def __init__(
        self,
        src_file_path: str,
        tgt_file_path: str,
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
    ):
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path

        self.num_lines_src = 0
        with open(self.src_file_path) as src_file:
            self.num_lines_src = sum(1 for _ in src_file)
        self.num_lines_tgt = 0
        with open(self.tgt_file_path) as tgt_file:
            self.num_lines_tgt = sum(1 for _ in tgt_file)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return min(self.num_lines_src, self.num_lines_tgt)

    def __getitem__(self, idx):
        src = None
        with open(self.src_file_path) as src_file:
            for i, line in enumerate(src_file):
                if i == idx:
                    src = line
                    break
        tgt = None
        with open(self.tgt_file_path) as tgt_file:
            for i, line in enumerate(tgt_file):
                if i == idx:
                    tgt = line
                    break

        src, tgt = process_line(src), process_line(tgt)
        src = TranslationDataset._convert_to_tokens(src, self.src_vocab)
        tgt = TranslationDataset._convert_to_tokens(tgt, self.tgt_vocab)

        return np.array(src), np.array(tgt)


def construct_vocab(file_path: str, non_punctuation_vocab_size: int):
    assert non_punctuation_vocab_size > 0, "vocab_size needs to be greater than 0"

    word_counts = {}
    max_len = -1
    with open(file_path) as f:
        for line in f:
            max_len = max(max_len, len(line))
            processed_line = process_line(line)
            for word in processed_line:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1

    kFrequent = heapq.nlargest(
        non_punctuation_vocab_size, word_counts.items(), key=lambda x: x[1]
    )
    elements = set([word for word, freq in kFrequent])
    for punc in [".", ",", "!", "?", "'", '"', "[", "]"]:
        if punc not in elements:
            elements.add(punc)
    for special_token in [EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]:
        elements.add(special_token)
    word_idx = {}
    for idx, word in enumerate(elements):
        word_idx[word] = idx

    return word_idx, max_len


def get_dataset(num_vocab: int) -> TranslationDataset:
    src_vocab, max_len_src = construct_vocab(SRC_FILE, num_vocab)
    tgt_vocab, max_len_tgt = construct_vocab(TGT_FILE, num_vocab)
    dataset = TranslationDataset(SRC_FILE, TGT_FILE, src_vocab, tgt_vocab)

    return dataset, max(max_len_src, max_len_tgt)


def translation_dataset_collate_fn(
    data: list[tuple[jax.Array, jax.Array]],
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    max_len: int,
) -> tuple[jax.Array, jax.Array]:
    assert PAD_TOKEN in src_vocab and PAD_TOKEN in tgt_vocab

    def pad_seq(sequences: list[jax.Array], vocab: dict[str, int]) -> list[jax.Array]:
        padded_sequences = []
        for sequence in sequences:
            padded_sequence = sequence.tolist() + (max_len - sequence.shape[0]) * [
                vocab[PAD_TOKEN]
            ]
            padded_sequences.append(padded_sequence)
        return padded_sequences

    src_data, tgt_data = zip(*data)
    src_data, tgt_data = pad_seq(src_data, src_vocab), pad_seq(tgt_data, tgt_vocab)

    return np.array(src_data), np.array(tgt_data)


def get_training_dataloader(
    dataset: TranslationDataset,
    max_len: int,
    batch_size: int,
    collate_fn: Callable[
        ..., tuple[jax.Array, jax.Array]
    ] = translation_dataset_collate_fn,
) -> DataLoader:
    collate_fn_vocab = partial(
        translation_dataset_collate_fn,
        src_vocab=dataset.src_vocab,
        tgt_vocab=dataset.tgt_vocab,
        max_len=max_len,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_vocab
    )

    return dataloader
