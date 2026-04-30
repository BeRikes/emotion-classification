import torch
from torch.utils.data import Dataset, DataLoader
from torch import tensor, save, load
import utils.preprocess as pre


class webo_100kDataset(Dataset):
    def __init__(self, data_path=None, seq_len=None, min_freq=None, extra_tokens=None, preload=None):
        super(webo_100kDataset, self).__init__()
        if data_path:
            if extra_tokens is None:
                extra_tokens = ['<pad>', '<eos>']
            text, self.label, avg_seq_len = pre.tokenAndLabel(data_path)
            print(f'average sequence len: {avg_seq_len}')
            self.vocab = pre.Vocab(text, min_freq=min_freq if min_freq else 5, reserved_tokens=extra_tokens)
            self.data = torch.stack([tensor(pre.token_to_same_len(self.vocab[line], seq_len, self.vocab['<pad>'], self.vocab['<eos>'])) for line in text], dim=0)
            self.data_num = len(self.data)
            self.label = tensor(self.label)
        elif preload:
            all_data = load(preload)
            self.data, self.label = all_data['data'], all_data['label']
            self.data_num, self.vocab = all_data['data_num'], all_data['vocab']
        else:
            self.data, self.label, self.data_num, self.vocab = None, None, None, None

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def save_data(self, path):
        save({
            'data': self.data,
            'label': self.label,
            'data_num': self.data_num,
            'vocab': self.vocab,
        }, path)

    def split(self, rate=0.2):
        assert self.data is not None, 'cannot split a null dataset'
        import random
        train_dataset, test_dataset = webo_100kDataset(), webo_100kDataset()
        train_data, train_label = [], []
        test_data, test_label = [], []
        for i, item in enumerate(self.data):
            if random.random() < rate:
                test_data.append(item)
                test_label.append(self.label[i])
            else:
                train_data.append(item)
                train_label.append(self.label[i])
        train_dataset.data, train_dataset.label, train_dataset.data_num = train_data, train_label, len(train_label)
        train_dataset.vocab = self.vocab
        test_dataset.data, test_dataset.label, test_dataset.data_num = test_data, test_label, len(test_label)
        test_dataset.vocab = self.vocab
        return train_dataset, test_dataset


if __name__ == '__main__':
    path = '../data/weibo_senti_100k.csv'
    dataset = webo_100kDataset(path, 100)
    loader = DataLoader(dataset, batch_size=64)
    for x, y in loader:
        print(x.shape)   # (b_size, seq_len)
        print(y.shape)
        print(x)
        print(y)
        break


