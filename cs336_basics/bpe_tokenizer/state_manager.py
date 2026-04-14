from collections import defaultdict

Pair = tuple[bytes, bytes]
TokenSeq = tuple[bytes, ...]

class StateManager:
    def __init__(self, process: dict[TokenSeq: int]):
        self.process = process.copy()
        self.token_neibo_num : dict[Pair: int] = defaultdict(int)
        self.token_affected_seq : dict[Pair: set] = defaultdict(set)
        self._initial_state()

    def _initial_state(self) -> None:
        for k, v in self.process.items():
            for idx in range(len(k) - 1):
                pair = (k[idx], k[idx + 1])
                self.token_neibo_num[pair] += v
                self.token_affected_seq[pair].add(k)
    
    def get_best_pair(self) -> Pair:
        return max(self.token_neibo_num, key=lambda p: (self.token_neibo_num[p], p))
    
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


    def _update_process(self, old_key: TokenSeq, new_key: TokenSeq, count: int):
        self.process[old_key] -= count
        self.process[new_key] = self.process.get(new_key, 0) + count

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

    def _remove_old_key(self, key: TokenSeq, count: int):
        for idx in range(len(key) - 1):
            pair = (key[idx], key[idx + 1])
            self.token_neibo_num[pair] -= count
            self.token_affected_seq[pair].discard(key)
        
    def _add_new_key(self, key: TokenSeq, count: int):
        for idx in range(len(key) - 1):
            pair = (key[idx], key[idx + 1])
            self.token_neibo_num[pair] += count
            self.token_affected_seq[pair].add(key)