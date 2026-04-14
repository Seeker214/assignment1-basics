import logging
import time

from .bpe_tokenizer.bpe_train import BPETrainer
from .utils.profile import profile_fun

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    trainer = BPETrainer(vocab_size=10000,
                        special_tokens=["<|endoftext|>"])
    profile_fun(func=trainer.train_multi,
                input_path="data/TinyStoriesV2-GPT4-train.txt",
                top_n=30,
                sort_by="tottime")


    # start_time = time.time()
    # logging.info(f"start time: {start_time}")
    # vocab, merges = trainer.train_multi(input_path="data/tinystories_mini_train.txt")
    # # vocab, merges = trainer.train(input_path="data/tinystories_mini_train.txt")
    # end_time = time.time()
    # logging.info(f"end time: {end_time}")
    # logging.info(f"total time: {end_time - start_time}")
    # merges = [[merges[idx][0].hex(), merges[idx][1].hex()] 
    #           for idx in range(len(merges))]
    # vocab = {token.hex(): idx for idx, token in enumerate(vocab) if idx >= 256}
    # import json

    # with open("cs336_basics/merges.json", "w", encoding="utf-8") as f:
    #     json.dump(merges, f)

    # with open("cs336_basics/vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab, f)


if __name__ == "__main__":
    main()

   


    