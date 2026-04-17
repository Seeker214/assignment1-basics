from collections import defaultdict
import re

def pre_tokenizer(text: list[str]) -> dict[tuple[bytes,...]: int]:
    # 构建 ("l", "o", "w") -> num 的映射
    process = defaultdict(int)
    # print(text_list)
    GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
    for text_s in text:
        # print(re.findall(GPT2_PATTERN, text_s))
        for chunk in re.findall(GPT2_PATTERN, text_s):
            text_chunk = tuple([bytes([b]) for b in chunk.encode("utf-8")])
            process[text_chunk] += 1
    return process
