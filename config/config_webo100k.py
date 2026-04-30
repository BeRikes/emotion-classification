class setting_config:
    data_path = 'data/weibo_senti_100k.csv'
    save_path = './result/exp1.pt'
    train_preload = './data/webo100k_train.pth'
    test_preload = './data/webo100k_test.pth'
    # model_cfg
    seq_len = 60
    num_blks = 2
    num_class = 2
    # train_cfg
    batch_size = 1024
    epoch = 100
    lr = 1e-4
    weight_decay = 5e-4
    eps = 5e-9
    warmup = 5