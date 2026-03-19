# 优化点：新增了multiprocessing处理预分词；扔掉了cppyy
# 已被JIT气晕
from __future__ import annotations
from .pretokenization import find_chunk_boundaries
import regex as re
import json
from typing import Iterable
import multiprocessing
import collections
import psutil

# 牛马
def _pre_tokenize_worker(file_path: str, start: int, end: int, pat_str: str, special_tokens: list[str]) -> collections.Counter[bytes]:
    import regex # 子进程需要重新import（？）
    counts: collections.Counter[bytes] = collections.Counter()
    pat: regex.Pattern = regex.compile(pat_str)
    with open(file_path, "rb") as f:
        f.seek(start)
        data: str = f.read(end - start).decode("utf-8", errors='replace')
    
    if special_tokens:
        st_pattern: str = "|".join(regex.escape(st) for st in special_tokens)
        chunks: list[str] = regex.split(st_pattern, data)
    else:
        chunks = [data]

    for chunk in chunks:
        if not chunk:
            continue
        for match in pat.finditer(chunk):
            counts[match.group().encode('utf-8')] += 1
    print("one worker done")
    return counts


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes] = None, merges: dict[tuple[int, int], int] = None, special_tokens: list[str] = None):
        # 初始化vocab为 0-255 的 ASCII/Byte 字符
        # 用于decode
        if vocab is None:
            self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        else:
            self.vocab = vocab
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        # 合并规则，key 是由两个 int 组成的 tuple (id1, id2), value 是新合并后的 int
        # 用于encode
        if merges is None:
            self.merges: dict[tuple[int, int], int] = {}
        else:
            self.merges = merges
        # 不可被拆的特殊字符
        self.special_tokens: list[str] = []
        if not special_tokens is None:
            # 按长度降序排列，确保较长的特殊token优先匹配（如 "<|endoftext|><|endoftext|>" 优先于 "<|endoftext|>"）
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        for item in self.special_tokens:
            # 下面这句很容易写错
            if item.encode('utf-8') in self.vocab.values():
                continue
            self.vocab[len(self.vocab)] = item.encode('utf-8')
        # 一坨屎啊这个字典生成器
        self.st_map: dict[str, int] = {token: next(i for i, v in self.vocab.items() if v == token.encode('utf-8')) for token in self.special_tokens}
        # 洗正则表达式
        PAT_base = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PAT_base: re.Pattern = re.compile(PAT_base)
        if self.special_tokens:
            # encode 时用含 special token 的正则，确保 special token 被整体匹配
            self.PAT: re.Pattern = re.compile("|".join([f"{re.escape(token)}" for token in self.special_tokens]) + "|" + PAT_base)
        else:
            self.PAT = self.PAT_base

        
    def train(self, file_path: str, vocab_size: int, max_bytes: int = None):
        if not self.special_tokens:
            raise ValueError("Training requires at least one special token.")
        word_freq: list[int] = []
        word_bytes: list[bytes] = []
        word_ids: list[list[int]] = []
        # 获取边界
        split_bytes: bytes = self.special_tokens[0].encode('utf-8')
        boundaries: list[int] = find_chunk_boundaries(file_path, split_bytes)
        
        # 处理 max_bytes
        if max_bytes:
            boundaries = [b for b in boundaries if b <= max_bytes]
            if not boundaries or boundaries[-1] < max_bytes:
                boundaries.append(max_bytes)

        # (文件路径, 起点, 终点, 正则, 标记)
        tasks = [(file_path, start, end, self.PAT_base.pattern, self.special_tokens) 
                 for start, end in zip(boundaries[:-1], boundaries[1:])]
        print(f"Starting pre-tokenization.")
        global_counts: collections.Counter[bytes] = collections.Counter()
        # 开进程池
        cpu_core_num: int = psutil.cpu_count()
        with multiprocessing.Pool(processes = cpu_core_num) as pool:
            # starmap 会自动解包 tasks 里的元组并传给 _pre_tokenize_worker
            results: list[collections.Counter[bytes]] = pool.starmap(_pre_tokenize_worker, tasks)
            for c in results:
                global_counts.update(c)
        for b_word, freq in global_counts.items():
            word_bytes.append(b_word)
            word_freq.append(freq)
            word_ids.append(list(b_word))
        print(f"Pre-tokenization finished. Unique words: {len(global_counts)}")
        # 建立pair到word的表格
        pair_freqs = collections.defaultdict(int)
        pair_to_words = collections.defaultdict(set)
        num_words = len(word_ids)
        for i in range(num_words):
            w_id = word_ids[i]
            w_freq = word_freq[i]
            len_word = len(w_id)
            for j in range(len_word - 1):
                pair = (w_id[j], w_id[j+1])
                pair_freqs[pair] += w_freq
                pair_to_words[pair].add(i)

        current_size = len(self.vocab)
        while (current_size < vocab_size):
            if not pair_freqs:
                break
            print("Processing:" + str(current_size))
            # 找pair
            # 频率相同时，按 (first_token_bytes, second_token_bytes) 元组字典序降序选择
            best_pair: tuple[int, int] = max(pair_freqs, key=lambda p: (pair_freqs[p], self.vocab[p[0]], self.vocab[p[1]]))
            if pair_freqs[best_pair] == 0:
                break
            self.merges[best_pair] = current_size

            words_affected: set[int] = pair_to_words[best_pair]
            for i in words_affected:
                w_id = word_ids[i]
                w_freq = word_freq[i]
                len_word = len(w_id)
                new_w_id: list[int] = []
                j: int = 0
                original_pairs: list[tuple[int, int]] = []
                new_pairs: list[tuple[int, int]] = []
                while j < len_word - 1:
                    original_pairs.append((w_id[j], w_id[j+1]))
                    j += 1
                j = 0
                while j < len_word:
                    if j < len_word - 1 and (w_id[j], w_id[j+1]) == best_pair:
                        new_w_id.append(current_size)
                        if j > 0:
                            pair_to_words[(w_id[j-1], current_size)].add(i)
                            pair_freqs[(w_id[j-1], current_size)] += w_freq
                            pair_freqs[(w_id[j-1], w_id[j])] -= w_freq

                        if j < len_word - 2:
                            pair_to_words[(current_size, w_id[j+2])].add(i)
                            pair_freqs[(current_size, w_id[j+2])] += w_freq
                            pair_freqs[(w_id[j+1], w_id[j+2])] -= w_freq
                        j += 2
                    else:
                        new_w_id.append(w_id[j])
                        j += 1
                word_ids[i] = new_w_id
                len_word = len(new_w_id)
                j = 0
                while j < len_word - 1:
                    new_pairs.append((new_w_id[j], new_w_id[j+1]))
                    j += 1
                # 对pair_to_words的一些对进行删除i操作
                new_pairs_set = set(new_pairs)
                original_pairs_set = set(original_pairs)
                original_pairs_set.discard(best_pair)
                for pair in original_pairs_set:
                    if pair not in new_pairs_set:
                        pair_to_words[pair].discard(i) # 用discard防止删多次报错
            # 删除
            del pair_to_words[best_pair]
            del pair_freqs[best_pair]


            # 记得更新vocab
            self.vocab[current_size] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # 拼字节
            self.reversed_vocab[self.vocab[current_size]] = current_size
            current_size += 1

    def encode(self, text: str) -> list[int]:
        res: list[int] = []
        if not self.special_tokens:
            words: re.Match = re.finditer(self.PAT, text)
            for match in words:
                res.extend(self._encode_word(match.group().encode('utf-8')))
            return res
        else:
            parts = re.split(f"({'|'.join(re.escape(st) for st in self.special_tokens)})", text)
            for part in parts:
                if not part:
                    continue
                if part in self.st_map:
                    res.append(self.st_map[part])
                else:
                    words: re.Match = re.finditer(self.PAT, part)
                    for match in words:
                        res.extend(self._encode_word(match.group().encode('utf-8')))
            return res
        
    def decode(self, ids: list[int]) -> str:
        raw = b''
        for id in ids:
            raw += self.vocab[id]
        return raw.decode('utf-8', errors='replace')

    def save(self, save_dir: str):
        import json
        # 先对vocab做处理 把int变成str 把bytes解码为latin-1
        vocab_str: dict[str, str] = {str(k): v.decode('latin-1') for k, v in self.vocab.items()}
        # 再对merges做处理 把tuple变成list
        merges_list: list[int] = [list(k) + [v] for k, v in self.merges.items()]
        # 把vocab_str和merges_str保存到json文件中
        with open(save_dir, 'w') as f:
            json.dump({'vocab': vocab_str, 'merges': merges_list, 'special_tokens': self.special_tokens}, f)

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        ids: list[int] = [self.reversed_vocab[bytes([b])] for b in word_bytes]
        while len(ids) > 1:
            best_pair: tuple[int, int] | None = None
            best_rank: int = float('inf')
            for i in range(len(ids) - 1):
                pair: tuple[int, int] = (ids[i], ids[i + 1])
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.merges[pair]
            if best_pair is None:
                break
            new_ids: list[int] = []
            i: int = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                    new_ids.append(self.merges[best_pair])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def encode_iterable(self, text_iter: Iterable[str]):
        # 流式编码说是
        for text in text_iter:
            yield from self.encode(text)

    @classmethod
    def load(cls: type[Tokenizer], save_dir: str) -> Tokenizer:
        with open(save_dir, 'r') as f:
            data = json.load(f)
        vocab = {int(k): v.encode('latin-1') for k, v in data['vocab'].items()}
        # 注意一下上面的items 不写的话只能拿到key
        merges = {tuple(k[:2]): k[2] for k in data['merges']} # for k in data['merges'] 指的是小列表
        # k[:2] 取前两个元素，k[2] 取第三个元素
        special_tokens = data.get('special_tokens', []) # 考虑没有special token的情况
        return cls(vocab = vocab, merges = merges, special_tokens = special_tokens)