from model.model import mini_bert
from utils.dataset import webo_100kDataset
from config.config_webo100k import setting_config
import torch
import jieba

def main(config, ckpt):
    test_dataset = webo_100kDataset(preload=config.test_preload)
    model = mini_bert(config.num_class, len(test_dataset.vocab), config.seq_len, test_dataset.vocab['<pad>'], N=config.num_blks, device='cpu')
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    model.cpu()
    # while True:
    #     idx = int(input('idx:'))
    #     if idx == '':
    #         break
    #     x, y = test_dataset[idx]
    #     x = x.cpu().unsqueeze(0)
    #     pred = torch.argmax(model(x), dim=-1)
    #     print('原句子:', ''.join(test_dataset.vocab.to_tokens(x.squeeze(0).tolist())))
    #     print('预测结果:', '积极情绪' if pred == 1 else '消极情绪')
    while True:
        words = input('输入:')
        if words == '':
            break
        tokens = jieba.lcut(words)
        tokens = torch.tensor(test_dataset.vocab[tokens])
        prob = torch.softmax(model(tokens.unsqueeze(0)).squeeze(0), dim=0)
        pred = torch.argmax(prob, dim=-1)
        print('原句子:', words)
        print('消极情绪概率:', prob[0].item())
        print('积极情绪概率:', prob[1].item())
        print('预测结果:', '积极情绪' if pred == 1 else '消极情绪')

main(setting_config(), 'result/exp1.pt')