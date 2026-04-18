今天开始回归古法编程提升代码水平。

# BPE Tokenization

## 为什么要有Unicode？

> 像我们日常交流的中文、英文，都是人类能够理解的文字，但是计算机无法理解，况且世界上的语言那么多，一个个输入工作量巨大不说，还不好统一测试。Unicode能够将文字转为编码形式，给复杂的文字以一种统一的表现形式。

## Problems

``` python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

这里的问题是，很多字符不是单个byte，一个字符可以对应多个bytes，如`你`这个字符encode后为`b'\xe4\xbd\xa0'`，但是`decode`时如果单个decode就会报错。

## BPE Train

先将输入的字符`encode`，分割成单个`byte`形式，然后根据其出现的频率进行`merge`，得到vocab和merges。
步骤分为以下几点：

1. Vocab 初始化
2. 预分词
3. 合并

### 1. Vocab 初始化

```Python
def build_initial_vocab(special_tokens: list[str]) -> list[bytes]:
    vocab = []
    for i in range(256):
        vocab.append(bytes([i]))
    for token in special_tokens:
        vocab.append(token.encode("utf-8"))
    return vocab
```
这里`bytes(i)`是生成$i$位的0 byte串，`bytes([i])`才是生成对应的`1-255`bytes。

### 2. 预分词
这里刚开始的理解有误，先是写成了`word tokenize`。可以分为两个步骤`分隔special tokens`和`分词`。

#### 2.1 分割special tokens

```Python
def split_with_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
    special_tokens_escape = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escape)
    text_list = re.split(pattern, text)
    # 清洗空白字符串
    text_list = [s for s in text_list if s]
    return text_list
```

#### 2.2 分词

```Python
def pre_tokenizer(text: list[str]) -> dict[tuple[bytes,...]: int]:
    # 构建 ("l", "o", "w") -> num 的映射
    process = defaultdict(int)
    GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
    for text_s in text:
        # print(re.findall(GPT2_PATTERN, text_s))
        for chunk in re.findall(GPT2_PATTERN, text_s):
            text_chunk = tuple([bytes([b]) for b in chunk.encode("utf-8")])
            process[text_chunk] += 1
    return process
```
chunk会生成如`['hi', ' i', ' am', ' bob', '!']`，构建的process便于后续每次统计相邻字节的数量

### 3. 合并
这里是最重要的部分。开始采用的是每次都全部遍历一遍字节方法，效率极其低下。在AI的帮助下，发现每次只有受影响的要更新，那么我们只需要维护**每个相邻字节串影响的字节序列映射**，每次只要更新这些被影响的字节序列即可。具体流程为：
1. 删除旧序列的贡献
2. 计算合并后的新序列
3. 添加新序列的贡献
4. 更新process

一次合并的大体框架如下：
```Python
def perform_merge(self, best_pair: Pair):
    affected_seq = self.token_affected_seq[best_pair].copy()

    for old_key in affected_seq:
        count = self.process[old_key]
        self._remove_old_key(old_key, count)
        new_key = self._calcucate_new_key(best_pair, old_key)
        self._add_new_key(new_key, count)
        self._update_process(old_key, new_key, count)
    
    del self.token_neibo_num[best_pair]
    del self.token_affected_seq[best_pair]
    self.process = {k: v for k, v in self.process.items() if v > 0}

```

#### 1. 删除旧序列贡献
传入的`key`就是待处理旧序列，`token_neibo_num`存相邻字符的数量，`token_affected_seq`存受影响的序列。
```python
def _remove_old_key(self, key: TokenSeq, count: int):
    for idx in range(len(key) - 1):
        pair = (key[idx], key[idx + 1])
        self.token_neibo_num[pair] -= count
        self.token_affected_seq[pair].discard(key)
```

#### 2. 计算合并后新序列
挨个遍历，合并后返回
```python
def _calcucate_new_key(self, pair: Pair, key: TokenSeq) -> TokenSeq:
    new_key = []
    idx = 0
    while idx < len(key) - 1:
        if key[idx] == pair[0] and key[idx + 1] == pair[1]:
            new_key.append(pair[0] + pair[1])
            idx += 2
        else:
            new_key.append(key[idx])
            idx += 1
    if idx != len(key):
        new_key.append(key[-1])
    return tuple(new_key)
```

#### 3.添加新序列贡献
跟删除是相反的处理
```python
def _add_new_key(self, key: TokenSeq, count: int):
    for idx in range(len(key) - 1):
        pair = (key[idx], key[idx + 1])
        self.token_neibo_num[pair] += count
        self.token_affected_seq[pair].add(key)
```

#### 4.更新process
```python
def _update_process(self, old_key: TokenSeq, new_key: TokenSeq, count: int):
    self.process[old_key] -= count
    self.process[new_key] = self.process.get(new_key, 0) + count
```

#### 分chunk处理
上述是一次性处理的代码，当需要切分chunk处理时，首先需要考虑切分chunk的边界问题。`[doc1][special_token][doc2]`我们不希望切分在`special_token`上或者将`[doc1]`给切分。需要找到一个安全边界。这里实现了一个根据`special_token`来切分的算法

```python
def _find_chunk_boundary(self, input_path: str, chunk_num: int) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // chunk_num
        boundaries = [i * chunk_size for i in range(chunk_num + 1)]

        boundaries[-1] = file_size
        mini_chunk_size = 4096      # 每次读4K

        for i in range(1, len(boundaries) - 1):
            initial_position = boundaries[i]
            f.seek(initial_position)
            while True:
                mini_chunk = f.read(mini_chunk_size)

                # 当查询到末尾后，字节串为空
                if mini_chunk == b"":
                    boundaries[i] = file_size
                    break
                
                # 查找最近的special token
                found = float('inf')
                for special_token in self.special_tokens:
                    found_at = mini_chunk.find(special_token.encode("utf-8"))
                    if found_at != -1:
                        found = min(found, found_at)
                if found != float('inf'):
                    boundaries[i] = initial_position + found
                    break

                # 未找到，继续扫描
                initial_position += mini_chunk_size
    return sorted(set(boundaries))
```
这个算法当special token集中在前面，是否会导致chunk的大小极差特别大的情况？还有待提升。
当待训练数据太大时，单个chunk大小按照预期数量切分时无法装入内存时，该算法失效。也许需要流式处理？

主体实现如下:
```python
def train_multi(self, input_path: str, 
                process_num: int | None = None):
    from multiprocessing import Pool, cpu_count
    process_num = process_num or max(1, cpu_count() - 1)

    boundaries = self._find_chunk_boundary(input_path=input_path, chunk_num=process_num)

    process_parameter = [(input_path, boundaries[i], boundaries[i + 1]) 
                            for i in range(len(boundaries) - 1)]

    with Pool(processes=process_num) as pool:
        all_process = pool.map(self._process_chunk, process_parameter)

    process = self._merge_process_other(all_process)

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
```
处理的逻辑和串行的一致，只是在pre tokenizer进行了chunk，后续需要再加入将子线程的结果合并的操作。

#### 一些思考
当数据集及其大的时候，无法全部进行训练，有什么方向进行改进呢？
从软件层面来看
1. 丢弃大量的出现数量为1的字节对
2. 卸载到磁盘，不过速度会变慢，这个思路跟DeepSpeed的offload思路一致
3. 进行数据采样，merge规律到一定的步数就会收敛

## Tokenizer
Tokenizer分为分词和合并操作。跟训练不同，训练时将`special tokens`用于分隔，这里的`special tokens`需要保存相应的位置且不进行任何操作，在实现的时候必须按照GPT2分词的词边界分词，不能。如：
```python
s = "i am bob"
s_list = ["i", " am", " bob"]
# 分词后为
[(b"i"), (b" ", b"a", b"m"), (b" ", b"b", b"o", b"b")]
# 只能每个序列中进行合并操作
```

以下为分词的结构
```python
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
```

合并操作：每次取字节对中，最先出现的字节对合并。`merges_rank`构造了`pair -> int`的映射，保证每次查询只要`O(1)`的时间复杂度

```python
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
```
