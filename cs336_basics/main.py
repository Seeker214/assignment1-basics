
from .bpe_tokenizer.bpe_train import BPETrainer
import logging
import time

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    start_time = time.time()
    logging.info(f"start time: {start_time}")
    bpe_trainer = BPETrainer(vocab_size=1000,
                         special_tokens=["<|endoftext|>"])

    vocab, merges = bpe_trainer.train(input_path="data/tinystories_mini_train.txt")

    end_time = time.time()
    logging.info(f"end time: {end_time}")
    logging.info(f"total time: {end_time - start_time}")
    merges = [[merges[idx][0].hex(), merges[idx][1].hex()] 
              for idx in range(len(merges))]
    vocab = {token.hex(): idx for idx, token in enumerate(vocab) if idx >= 256}
    import json

    with open("cs336_basics/merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f)

    with open("cs336_basics/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

# main()

trainer = BPETrainer(vocab_size=262, special_tokens=["<|endoftext|>"])
vocab, merges = trainer.train(input_path="cs336_basics/test.txt")
print(vocab)
print(vocab[256:])

