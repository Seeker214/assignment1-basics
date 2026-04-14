import re
import os
import logging

from .pre_tokenizer import pre_tokenizer
from .vocab import build_initial_vocab
from .state_manager import StateManager

class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = build_initial_vocab(self.special_tokens)
        self.merges = []

    def train(self, input_path: str):
        with open(input_path, "r") as f:
            text = f.read()
        
        special_tokens_escape = [re.escape(t) for t in self.special_tokens]
        pattern = "|".join(special_tokens_escape)
        text_list = re.split(pattern, text)
        # 清洗空白字符串
        text_list = [s for s in text_list if s]

        process = pre_tokenizer(text_list)

        manager = StateManager(process)

        merge_num = self.vocab_size - len(self.vocab)
        while merge_num:
            best_pair = manager.get_best_pair()
            self.merges.append(best_pair)
            manager.perform_merge(best_pair)
            self.vocab.append(best_pair[0] + best_pair[1])

            merge_num -= 1
        self.vocab = {token: idx for idx, token in enumerate(self.vocab)}
        return self.vocab, self.merges
    
    def train_multi(self, input_path: str, 
                    process_num: int | None = None):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        from multiprocessing import Pool, cpu_count
        process_num = process_num or max(1, cpu_count() - 1)
        # logging.info(f"process_num: {process_num}\n")
        boundaries = self._find_chunk_boundary(input_path=input_path, chunk_num=process_num)
        # logging.info(f"boundaries: {boundaries}\n")
        process_parameter = [(input_path, boundaries[i], boundaries[i + 1]) 
                             for i in range(len(boundaries) - 1)]

        with Pool(processes=process_num) as pool:
            all_process = pool.map(self._process_chunk, process_parameter)
        # logging.info(f"all_process: {all_process} \n")
        
        process = self._merge_process_other(all_process)
        # logging.info(f"process: {process} \n")
        manager = StateManager(process)

        merge_num = self.vocab_size - len(self.vocab)
        while merge_num:
            best_pair = manager.get_best_pair()
            self.merges.append(best_pair)
            manager.perform_merge(best_pair)
            self.vocab.append(best_pair[0] + best_pair[1])

            merge_num -= 1
        self.vocab = {token: idx for idx, token in enumerate(self.vocab)}
        return self.vocab, self.merges

    def _find_chunk_boundary(self, input_path: str, chunk_num: int) -> list[int]:
        with open(input_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            # logging.info(file_size)
            chunk_size = file_size // chunk_num
            boundaries = [i * chunk_size for i in range(chunk_num + 1)]
            # logging.info(boundaries)
            boundaries[-1] = file_size
            mini_chunk_size = 4096

            for i in range(1, len(boundaries) - 1):
                initial_position = boundaries[i]
                f.seek(initial_position)
                while True:
                    mini_chunk = f.read(mini_chunk_size)

                    if mini_chunk == b"":
                        boundaries[i] = file_size
                        break

                    found = float('inf')
                    for special_token in self.special_tokens:
                        found_at = mini_chunk.find(special_token.encode("utf-8"))
                        if found_at != -1:
                            found = min(found, found_at)
                    if found != float('inf'):
                        boundaries[i] = initial_position + found
                        break
                    initial_position += mini_chunk_size
        return sorted(set(boundaries))
    
    def _process_chunk(self, args):
        input_path, start, end = args
        with open(input_path, "rb") as f:
            f.seek(start)
            buffer = f.read(end - start)

            special_tokens_escape = [re.escape(t) for t in self.special_tokens]
            pattern = "|".join(special_tokens_escape)
            text_list = re.split(pattern, buffer.decode("utf-8"))
            # 清洗空白字符串
            text_list = [s for s in text_list if s]

            process = pre_tokenizer(text_list)
        return process

    def _merge_process_other(self, all_process: list[dict[int]]):
        from collections import defaultdict
        total_process = defaultdict(int)
        for process in all_process:
            for k, v in process.items():
                total_process[k] += v
        return total_process