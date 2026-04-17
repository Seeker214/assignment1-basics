
import logging

from typing import Iterable

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]

class Tokenizer:
    def __init__(self, vocab: Vocab, merges: Merges, 
                 special_tokens: list[str] | None = None):
        self.vocab_id_bytes = vocab
        self.vocab_bytes_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        self.special_tokens = []
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=lambda t: len(t), reverse=True)
            
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        import json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {v: k.encode("utf-8") for k, v in vocab.items()}

        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                pair = line.split()
                merges.append((pair[0].encode("utf-8"), pair[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        text_list = []
        if self.special_tokens:
            text_list = split_with_special_tokens(text, self.special_tokens)
        else:
            text_list.append(text)
        chunks = self._pre_tokenize(text_list)
        chunks_count = []
        for chunk in chunks:
            chunk_merged = self._token_merge(chunk, self.merges_rank)

            for ck in chunk_merged:
                chunks_count.append(self.vocab_bytes_id[ck])

        return chunks_count


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
        
    def encode_stream(
        self,
        iterable: Iterable[str],
        bpe_tail_bytes: int = 64,   # BPE 边界保留窗口（可调）
    ) -> Iterable[int]:
        buffer = ""
        bytes_tail = ""

        max_special_token_len = -1
        if self.special_tokens:
            max_special_token_len = max(self.special_tokens, key=lambda t: len(t))
        else:
            max_special_token_len = 0

        for str_iter in iterable:
            buffer += str_iter

            if len(buffer) <= max_special_token_len:
                continue
            elif max_special_token_len == 0:
                safe_buffer = buffer
                bytes_tail = ""
            else:
                safe_buffer = buffer[:-max_special_token_len]
                bytes_tail = buffer[-max_special_token_len:]
            
            if len(safe_buffer) < bpe_tail_bytes:
                safe_buffer += bytes_tail
                bytes_tail = ""
                continue
            for tid in self.encode(safe_buffer):
                yield tid

        final_tokens = safe_buffer + bytes_tail
        for tid in self.encode(final_tokens):
            yield tid






    def decode(self, ids: list[int]) -> str:
        raw_bytes = b''
        for idx in range(len(ids)):
            raw_bytes += self.vocab_id_bytes[ids[idx]]
        raw_str = raw_bytes.decode("utf-8", errors="replace")
        return raw_str

    def _pre_tokenize(self, text: list[str]) -> list[list[bytes]]:
        import regex
        GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        all_chunks = []
        for text_s in text:
            if self.special_tokens and text_s in self.special_tokens:
                all_chunks.append([text_s.encode("utf-8")])
                continue
            mini_chunk = regex.findall(GPT2_PATTERN, text_s)

            for idx in range(len(mini_chunk)):
                next_bytes = [bytes([b]) for b in mini_chunk[idx].encode("utf-8", errors="replace")]
                all_chunks.append(next_bytes)

        return all_chunks
    
    def _token_merge(self, chunks: list[bytes], merges_dict: dict[tuple[bytes, bytes], int]) -> list[bytes]:
        while True:
            best_idx = -1
            best_rank = float('inf')

            for idx in range(len(chunks) - 1):
                pair = (chunks[idx], chunks[idx + 1])
                rank = merges_dict.get(pair)

                if rank is not None and rank < best_rank:
                    best_idx = idx
                    best_rank = rank
            
            if best_idx == -1:
                break

            chunks[best_idx] = chunks[best_idx] + chunks[best_idx + 1]
            del chunks[best_idx + 1]
        return chunks

import re
def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    special_tokens_escape = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escape)
    special_at = []
    for match in re.finditer(pattern, text):
        special_at.append((match.start(), match.end()))
    text_list = []
    pre_idx = 0
    for start, end in special_at:
        text_list.append(text[pre_idx:start])
        text_list.append(text[start:end])
        pre_idx = end
    text_list.append(text[pre_idx:])
    # # 清洗空白字符串
    text_list = [s for s in text_list if s]

    return text_list