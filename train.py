import sys
import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse

from config.config_webo100k import setting_config
from model.model import mini_bert
from utils.dataset import webo_100kDataset
from utils.early_stop import Early_stop


def print_red(text):
    print('\033[1;31m' + text + '\033[0m')


def train_one_epoch(train_loader, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader, file=sys.stdout, colour='blue')):
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / (i + 1)


def test_one_epoch(test_loader, model, criterion):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader, file=sys.stdout, colour='green')):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
    return test_loss / (i + 1)


def main(config):
    # dataset = webo_100kDataset(config.data_path, config.seq_len)
    # print('------ data preprocessed-------')
    # train_dataset, test_dataset = dataset.split()
    # del dataset.data
    # del dataset.label
    train_dataset = webo_100kDataset(preload=config.train_preload)
    test_dataset = webo_100kDataset(preload=config.test_preload)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    print('--------datasets loaded--------')
    model = mini_bert(config.num_class, len(train_dataset.vocab), config.seq_len, train_dataset.vocab['<pad>'], N=config.num_blks)
    print(f'model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print('----------model loaded---------')
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=config.eps, betas=(0.9, 0.98))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=10)
    max_epoch = config.epoch
    early_stop = Early_stop(15, 0.01, config.save_path)
    train_record, test_record = [], []
    criterion = CrossEntropyLoss()
    for epoch in range(1, max_epoch + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion)
        test_loss = test_one_epoch(test_loader, model, criterion)
        train_record.append(train_loss)
        test_record.append(test_loss)
        print_red(f'epoch {epoch}, train_loss {train_loss}, test_loss {test_loss}, lr {optimizer.state_dict()["param_groups"][0]["lr"]}')
        if epoch > config.warmup:
            scheduler.step(test_loss)
        early_stop(test_loss, epoch, model)
        if early_stop.stop:
            print(f'train early stop at epoch {epoch}')
            break
    print(f'the best test loss is {early_stop.best_loss}, best epoch is {early_stop.best_epoch}')
    fig, ax = plt.subplots(1, 2, sharey='all', sharex='all')
    ax[0].plot(train_record)
    ax[0].set_title('train loss')
    ax[1].plot(test_record)
    ax[1].set_title('test loss')
    plt.show()


if __name__ == '__main__':
    main(setting_config())