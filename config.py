import logging
import codecs
import os

batch_size = 32
train_parameters = {
    "input_size": [3, 256, 512],
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得
    "train_data_dir": "sh/sh/part_B_final/train_data/images/",  # 训练数据存储地址

    "continue_train": False,        # 是否接着上一次保存的参数接着训练
    "continue_train_dir": '/home/aistudio/work/checkpoints1/csrnet/v3/CAN210',
    "mode": "train",
    "num_epochs": 2000,
    "method": 'CSR',
    "save_dir": "/home/aistudio/work/checkpoints1/csrnet/v3/",
    "train_batch_size": batch_size,

    "learning_strategy": {
        "name": "cosine_decay",
        "batch_size": batch_size,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },
    "lr": 0.00001
}


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量
    :return:
    """
    train_lists = os.listdir(train_parameters['train_data_dir'])
    train_parameters['image_count'] = len(train_lists)



#init_train_parameters()