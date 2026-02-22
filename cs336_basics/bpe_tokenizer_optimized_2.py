from __future__ import annotations
import regex as re
import json
import cppyy
from typing import Iterable

if not hasattr(cppyy.gbl, 'Train'):
    cppyy.cppdef('''
#include <vector>
#include <unordered_map>
#include <utility>
#include <thread>
#include <mutex>
#include <algorithm>
#include <atomic>

class Train{
public:
    Train(){
    }

    std::unordered_map<uint64_t, uint32_t> freq;
    std::vector<std::vector<uint32_t>> word_bytes;
    std::vector<std::string> vocab_cpp; // 这里我们用std:string存放Python中的bytes
                 
    void sync_vocab_entry(uint32_t id, const std::string& bytes) {
        if (id >= vocab_cpp.size()) {
            vocab_cpp.resize(id + 1);
        }
        vocab_cpp[id] = bytes;
    }

    void ejalucation(const std::string& bytes) {
        std::vector<uint32_t> ids;
        // 预分配内存
        ids.reserve(bytes.size());
        for (uint32_t i = 0; i < bytes.size(); i++) {
            // ids.push_back(bytes[i]);这是不正确的
            // 注意一下，考虑到非ASCII字符，要把bytes[i]转换成unsigned char
            ids.push_back(static_cast<unsigned char>(bytes[i]));
        }
        word_bytes.push_back(ids);
    }

    void merge_pair(std::pair<uint32_t, uint32_t> pair, uint32_t new_id){
        // 获取核心数
        uint16_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        // Atomic计数器
        std::atomic<uint64_t> index(0);
        uint64_t total_words = word_bytes.size();

        std::vector<std::thread> threads;
        for (uint16_t i = 0; i < num_threads; i++) {
            // 创建并启动一个新线程
            // std::ref() 是为了引用传递
            threads.emplace_back(&Train::merge_worker, this, total_words, std::ref(index), pair, new_id);
        }
        // 主线程等待子线程结束
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }

    void counting_freq(){
        freq.clear();
        // 获取核心数
        uint16_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        // Atomic计数器
        std::atomic<uint64_t> index(0);
        uint64_t total_words = word_bytes.size();

        std::vector<std::thread> threads;
        std::vector<std::unordered_map<uint64_t, uint32_t>> thread_freqs(num_threads);
        for (uint16_t i = 0; i < num_threads; i++) {
            // 创建并启动一个新线程
            // std::ref() 是为了引用传递
            threads.emplace_back(&Train::freq_worker, this, total_words, std::ref(index), std::ref(thread_freqs[i]));
        }
        // 主线程等待子线程结束
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
        // 合并所有线程的计数结果
        for (auto& thread_freq : thread_freqs) {
            for (auto& [pair, count] : thread_freq) {
                freq[pair] += count;
            }
        }
    }

    void clear_word_bytes(){
        word_bytes.clear();
    }

    std::vector<int> get_linear_word_bytes(){ // 已作废 不再使用
        std::vector<int> res;
        // 听说预分配内存可以提高性能
        uint32_t total_size = 0;
        for (uint32_t i = 0; i < word_bytes.size(); i++) {
            total_size += word_bytes[i].size();
        }
        res.reserve(total_size);
        for (uint32_t i = 0; i < word_bytes.size(); i++) {
            res.insert(res.end(), word_bytes[i].begin(), word_bytes[i].end());
        }
        return res;
    }

    // 牛马函数
    void freq_worker(uint64_t total_words, std::atomic<uint64_t>& index, std::unordered_map<uint64_t, uint32_t>& thread_freq){
        uint16_t batch_size = 1024;
        while (true){
            uint64_t start = index.fetch_add(batch_size);
            if (start >= total_words) break;
            uint64_t end = std::min(start + batch_size, total_words);
            for (uint64_t i = start; i < end; i++) {
                if (word_bytes[i].size() < 2) continue;
                for (uint64_t j = 0; j < word_bytes[i].size() - 1; j++) {
                    thread_freq[static_cast<uint64_t>(word_bytes[i][j]) << 32 | word_bytes[i][j+1]] += 1; //C++中，我们不需要检查pair是否在freq中，直接加就行
                }
            }
        }
    }

    void merge_worker(uint64_t total_words, std::atomic<uint64_t>& index, std::pair<uint32_t, uint32_t> pair, uint32_t new_id){
        uint16_t batch_size = 1024;
        while (true){
            uint64_t start = index.fetch_add(batch_size);
            if (start >= total_words) break;
            uint64_t end = std::min(start + batch_size, total_words);
            for (uint64_t i = start; i < end; i++) {
                // 这里我们使用极其先进的双指针法 一个读指针 一个写指针
                std::vector<uint32_t>& ids = word_bytes[i];
                if (ids.size() < 2) continue;
                uint32_t read_ptr = 0;
                uint32_t write_ptr = 0;
                while (read_ptr < ids.size()) {
                    if (read_ptr < ids.size() - 1 && ids[read_ptr] == pair.first && ids[read_ptr + 1] == pair.second){
                        ids[write_ptr] = new_id;
                        read_ptr += 2;
                    } else {
                        ids[write_ptr] = ids[read_ptr];
                        read_ptr += 1;
                    }
                    write_ptr += 1; 
                }
                ids.resize(write_ptr); //把末尾没用的部分夹断
            }
        }
    }
    
    uint64_t get_best_pair(){
        uint64_t best_pair = 0;
        uint32_t best_freq = 0;
        std::string best_p1, best_p2;
        for (auto& [pair, freq] : freq) {
            if (freq > best_freq) {
                best_freq = freq;
                best_pair = pair;
                best_p1 = vocab_cpp[pair >> 32];
                best_p2 = vocab_cpp[pair & 0xffffffff];
            } else if (freq == best_freq) {
                const std::string& p1 = vocab_cpp[pair >> 32];
                const std::string& p2 = vocab_cpp[pair & 0xffffffff];
                if (p1 > best_p1 || (p1 == best_p1 && p2 > best_p2)) {
                    best_pair = pair;
                    best_p1 = p1;
                    best_p2 = p2;
                }
            }
        }
        return best_pair;
    }
                 
};

''')

Train = cppyy.gbl.Train

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
        # 神人C++
        self.t = Train()
        # 洗正则表达式
        PAT_base = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PAT_base = re.compile(PAT_base)
        if self.special_tokens:
            # encode 时用含 special token 的正则，确保 special token 被整体匹配
            self.PAT = re.compile("|".join([f"{re.escape(token)}" for token in self.special_tokens]) + "|" + PAT_base)
        else:
            self.PAT = self.PAT_base
        # 同步vocab到C++
        for id, bval in self.vocab.items():
            self.t.sync_vocab_entry(id, bval)
        
    def train(self, file_path: str, vocab_size: int, max_bytes: int = None):
        # with open()保证离开缩进块后自动 close，非常RAII
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_bytes is None:
                text = f.read()
            else:
                text = f.read(max_bytes)
        print(f"Read {len(text)} characters.")

        # 训练时先按 special token 分割语料，去掉 special token 本身
        # 然后对每个片段用基础正则做 pre-tokenization
        if self.special_tokens:
            st_pattern = "|".join(re.escape(st) for st in self.special_tokens)
            chunks = re.split(st_pattern, text)
            for chunk in chunks:
                if chunk:
                    self._pre_tokenize_base(chunk)
        else:
            self._pre_tokenize_base(text)
        current_size = len(self.vocab)
        while (current_size < vocab_size):
            self.t.counting_freq()
            if self.t.freq.size() == 0:
                break
            # C++ map 迭代返回的是键值元组，不是 Key，行为和Python不一致
            # 下面的代码中p[1]是count, p[0]是uint64_t形式的pair
            # 频率相同时，按 (first_token_bytes, second_token_bytes) 元组字典序降序选择
            # 注意一下元组比较和拼接比较不一样
            # best_item = max(self.t.freq, key=lambda p: (p[1], (self.vocab[p[0] >> 32], self.vocab[p[0] & 0xffffffff])))
            # best_pair = (best_item[0] >> 32, best_item[0] & 0xffffffff) # 转成 Python tuple 方便后续使用
            best_pair_uint64_t = self.t.get_best_pair()
            best_pair = (best_pair_uint64_t >> 32, best_pair_uint64_t & 0xffffffff)
            self.merges[best_pair] = current_size

            self.t.merge_pair(best_pair, current_size)

            # 记得更新vocab
            self.vocab[current_size] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # 拼字节
            self.reversed_vocab[self.vocab[current_size]] = current_size
            # 同步新 vocab entry 到 C++ 侧，供 get_best_pair 的 tie-breaking 使用
            self.t.sync_vocab_entry(current_size, self.vocab[current_size])
            current_size += 1

    def _pre_tokenize_base(self, text: str):
        """训练时用基础正则（不含 special token）做 pre-tokenization"""
        for match in re.finditer(self.PAT_base, text):
            self.t.ejalucation(match.group().encode('utf-8'))

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