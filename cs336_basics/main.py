
from cs336_basics.bpe_tokenizer.bpe_train import BPETrainer
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
import logging
import time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
def main():
    bpe_trainer = BPETrainer(vocab_size=500,
                         special_tokens=["<|endoftext|>"])

    vocab, merges = bpe_trainer.train(input_path="data/tinystories_mini_train.txt")
    logging.info(merges)

    # merges = [[merges[idx][0].hex(), merges[idx][1].hex()] 
    #           for idx in range(len(merges))]
    # vocab = {token.hex(): idx for idx, token in enumerate(vocab) if idx >= 256}
    # import json

    with open("cs336_basics/merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{pair[0].decode("utf-8")}\t{pair[1].decode("utf-8")}\n")

    # with open("cs336_basics/vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab, f)

# main()

# trainer = BPETrainer(vocab_size=262, special_tokens=["<|endoftext|>"])
# vocab, merges = trainer.train(input_path="cs336_basics/test.txt")
# print(vocab)
# print(vocab[256:])

# vocab_path = "tests/fixtures/train-bpe-reference-vocab.json"
# merges_path = "tests/fixtures/train-bpe-reference-merges.txt"
# tok = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)

# text = """Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!
# He said, “Wow, that is a really amazing vase! Can I buy it?”
# The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
# So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was.
# And that's how Ben found an amazing vase in the store!"""
# tok.encode(text)
# logging.info(tok.vocab)



# vocab = {
#     # 基础字符
#     "a":  0,  "b":  1,  "c":  2,  "d":  3,  "e":  4,
#     "h":  5,  "i":  6,  "l":  7,  "n":  8,  "o":  9,
#     "r": 10,  "s": 11,  "t": 12,  "u": 13,  "w": 14,
#     # merge 产生的子词
#     "he":     15,   # merge 1
#     "lo":     16,   # merge 2
#     "low":    17,   # merge 3
#     "her":    18,   # merge 4
#     "ne":     19,   # merge 5
#     "new":    20,   # merge 6
#     "es":     21,   # merge 7
#     "est":    22,   # merge 8
#     "newest": 23,   # merge 9
#     "lowe":   24,   # merge 10
#     "lower":  25,   # merge 11
# }
# # vocab = {k.encode("utf-8"):v for k, v in vocab.items()}
# vocab = {v:k.encode("utf-8") for k, v in vocab.items()}
# merges = [
#     ("h",    "e"),    # 1 → he
#     ("l",    "o"),    # 2 → lo
#     ("lo",   "w"),    # 3 → low
#     ("he",   "r"),    # 4 → her
#     ("n",    "e"),    # 5 → ne
#     ("ne",   "w"),    # 6 → new
#     ("e",    "s"),    # 7 → es
#     ("es",   "t"),    # 8 → est
#     ("new",  "est"),  # 9 → newest
#     ("low",  "e"),    # 10 → lowe
#     ("lowe", "r"),    # 11 → lower
# ]
# merges = [(b1.encode("utf-8"), b2.encode("utf-8")) for (b1, b2) in merges]
# text = "hello"
# tok = Tokenizer(vocab, merges)
# result = tok.encode(text)
# logging.info(result)
# raw_str = tok.decode(result)
# logging.info(raw_str)



import unittest
from typing import Iterator
import tracemalloc
def huge_text_stream(num_lines: int, line_size: int) -> Iterator[str]:
    base = ("hello " * ((line_size - 1) // 6))[: line_size - 1]
    line = base + "\n"
    for _ in range(num_lines):
        yield line


def _run_collect_ids(tok: Tokenizer, num_lines: int, line_size: int) -> int:
    ids = list(tok.encode_iterable(huge_text_stream(num_lines, line_size)))
    return len(ids)


def _run_stream_only(tok: Tokenizer, num_lines: int, line_size: int) -> int:
    n = 0
    for _ in tok.encode_iterable(huge_text_stream(num_lines, line_size)):
        n += 1
    return n


def test_encode_iterable_reduces_memory():
    vocab_bytes_to_id = {
        b"h": 0, b"e": 1, b"l": 2, b"o": 3, b" ": 4, b"\n": 5,
        b"he": 6, b"lo": 7,
    }
    vocab = {v:k for k, v in vocab_bytes_to_id.items()}
    merges = [(b"h", b"e"), (b"l", b"o")]
    tok = Tokenizer(vocab, merges)

    num_lines = 200_00
    line_size = 200



if __name__ == "__main__":
    test_encode_iterable_reduces_memory()