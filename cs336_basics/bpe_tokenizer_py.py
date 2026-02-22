import regex as re
import json

class Tokenizer:
    def __init__(self):
        # 初始化vocab为 0-255 的 ASCII/Byte 字符
        # 用于decode
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # 合并规则，key 是由两个 int 组成的 tuple (id1, id2), value 是新合并后的 int
        # 用于encode
        self.merges: dict[tuple[int, int], int] = {}
        self.freq: dict[tuple[int, int], int] = {}

    def _merge_pair(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        res: list[int] = []
        i: int = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                res.append(new_id)
                i += 2
            else:
                res.append(ids[i])
                i += 1
        return res
        
    def train(self, file_path, vocab_size):
        # with open()保证离开缩进块后自动 close，非常RAII
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Read {len(text)} characters.")

        word_bytes = self._pre_tokenize(text)
        current_size = 256
        while (current_size < vocab_size):
            self.freq.clear()
            for ids in word_bytes:
                for i in range(0, len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    if pair not in self.freq:
                        self.freq[pair] = 1
                    else:
                        self.freq[pair] += 1
            # 这里其实是在找最大的val对应的key
            # best_pair = max(self.freq, key=self.freq.get)
            # 但是我们还要考虑val相同的情况 此时我们要取大的那个 这个没有实际意义 只是为了保证结果的确定性
            # 此处“大”并不指字典序 而是ID
            best_pair = max(self.freq, key = lambda p: (self.freq[p], p))
            self.merges[best_pair] = current_size

            new_word_bytes: list[list[int]] = []
            for ids in word_bytes:
                # ids = self._merge_pair(ids, best_pair, current_size) 这种写法是错误的，它不会修改word_bytes
                new_ids = self._merge_pair(ids, best_pair, current_size)
                new_word_bytes.append(new_ids)
            word_bytes = new_word_bytes
            # 记得更新vocab
            self.vocab[current_size] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # 拼字节
            current_size += 1

    def _pre_tokenize(self, text: str) -> list[list[int]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words: re.Match = re.finditer(PAT, text)
        word_bytes: list[list[int]] = []
        for match in words:
            ids: list[int] = list(match.group().encode('utf-8'))
            word_bytes.append(ids)
        return word_bytes

    def encode(self, text: str) -> list[int]:
        word_bytes: list[list[int]] = self._pre_tokenize(text)
        # Dict保证保持插入顺序，因此不需要额外排序
        for pair, new_id in self.merges.items(): 
            new_word_bytes: list[list[int]] = []
            for ids in word_bytes:
                new_ids = self._merge_pair(ids, pair, new_id)
                new_word_bytes.append(new_ids)
            word_bytes = new_word_bytes
        # 接下来要把这一坨套娃list展开成一维
        res: list[int] = []
        for ids in word_bytes:
            res.extend(ids)
        return res
        
    def decode(self, ids: list[int]) -> str:
        bytes = b''
        for id in ids:
            bytes += self.vocab[id]
        return bytes.decode('utf-8')

    def save(self, save_dir: str):
        import json
        # 先对vocab做处理 把int变成str 把bytes解码为latin-1
        vocab_str: dict[str, str] = {str(k): v.decode('latin-1') for k, v in self.vocab.items()}
        # 再对merges做处理 把tuple变成list
        merges_list: list[int] = [list(k) + [v] for k, v in self.merges.items()]
        # 把vocab_str和merges_str保存到json文件中
        with open(save_dir, 'w') as f:
            json.dump({'vocab': vocab_str, 'merges': merges_list}, f)

    @classmethod
    def load(cls: type[Tokenizer], save_dir: str) -> Tokenizer:
        res: Tokenizer = cls()
        with open(save_dir, 'r') as f:
            data = json.load(f)
        res.vocab = {int(k): v.encode('latin-1') for k, v in data['vocab'].items()}
        # 注意一下上面的items 不写的话只能拿到key
        res.merges = {tuple(k[:2]): k[2] for k in data['merges']} # for k in data['merges'] 指的是小列表
        # k[:2] 取前两个元素，k[2] 取第三个元素
        return res
