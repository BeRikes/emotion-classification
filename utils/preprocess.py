import jieba
import collections
import joblib


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def save(self, path):
        with open(path, mode='wb') as f:
            joblib.dump(self, path, True)

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenAndLabel(data_path, num_example=None):
    """将path文件中的所有行拆分成单词词元"""
    line_tokens, label, sum = [], [], 0
    with open(data_path, mode='r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    lines = lines[1:]
    if num_example is None:
        num_example = len(lines)
    for i, line in enumerate(lines):
        line = line.strip('\n')
        line_tokens.append(jieba.lcut(line[2:]))
        sum += len(line_tokens[i])
        label.append(int(line[0]))
        if i >= num_example:
            break
    return line_tokens, label, sum / len(lines)


def token_to_same_len(line: list, lenth, padding: int, end: int):
    """把一行token索引序列，变换到相同长度，并在有效字符的结尾处加结束标志，结束标志占1位长度"""
    if len(line) > (lenth - 1):
        return line[:lenth-1] + [end]
    elif len(line) == 0:
        print('found zero words')
    else:
        return line + [end] + [padding] * (lenth - len(line) - 1)


if __name__ == '__main__':
    print(tokenAndLabel('../data/weibo_senti_100k.csv', 100))