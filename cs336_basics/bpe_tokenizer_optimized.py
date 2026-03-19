# 版本 1：初步引入 C++ (cppyy)
# 特点：将核心逻辑移至 C++，但此时的 C++ 实现仍为单线程，且使用了较慢的 std::map，未实现多并行。
import regex as re
import json
import cppyy

cppyy.cppdef('''
#include <vector>
#include <map>
#include <utility>

class Train{
public:
    Train(){
        
    }
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> freq;
    std::vector<std::vector<uint32_t>> word_bytes;

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
        // 这里我们使用极其先进的双指针法 一个读指针 一个写指针
        for (uint32_t i = 0; i < word_bytes.size(); i++) {
            std::vector<uint32_t>& ids = word_bytes[i];
            if (ids.size() == 0 || ids.size() == 1) continue;
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

    void counting_freq(){
        freq.clear();
        for (uint32_t i = 0; i < word_bytes.size(); i++) {
            for (uint32_t j = 0; j < word_bytes[i].size() - 1; j++) {
                std::pair<uint32_t, uint32_t> pair = {word_bytes[i][j], word_bytes[i][j+1]};
                freq[pair] += word_freq[i]; //C++中，我们不需要检查pair是否在freq中，直接加就行
            }
        }
    }

    void clear_word_bytes(){
        word_bytes.clear();
    }

    std::vector<int> get_linear_word_bytes(){
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
};


''')

class Tokenizer:
    def __init__(self):
        # 初始化vocab为 0-255 的 ASCII/Byte 字符
        # 用于decode
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # 合并规则，key 是由两个 int 组成的 tuple (id1, id2), value 是新合并后的 int
        # 用于encode
        self.merges: dict[tuple[int, int], int] = {}

        # 神人C++
        self.t = Train()
        
    def train(self, file_path, vocab_size):
        # with open()保证离开缩进块后自动 close，非常RAII
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Read {len(text)} characters.")
        self._pre_tokenize(text)
        current_size = 256
        while (current_size < vocab_size):
            self.t.counting_freq()
            # C++ map 迭代返回的是键值元组，不是 Key，行为和Python不一致
            # 下面的代码中p[1]是count, p[0]是pair
            best_item = max(self.t.freq, key=lambda p: (p[1], p[0]))
            best_pair = tuple(best_item[0]) # 转成 Python tuple 方便后续使用
            self.merges[best_pair] = current_size

            self.t.merge_pair(best_pair, current_size)

            # 记得更新vocab
            self.vocab[current_size] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # 拼字节
            current_size += 1

    def _pre_tokenize(self, text: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words: re.Match = re.finditer(PAT, text)
        for match in words:
            self.t.ejalucation(match.group().encode('utf-8'))

    def encode(self, text: str) -> List[int]:
        self.t.clear_word_bytes()
        self._pre_tokenize(text)
        for pair, new_id in self.merges.items():
            self.t.merge_pair(pair, new_id)
        return self.t.get_linear_word_bytes()
        
    def decode(self, ids: List[int]) -> str:
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