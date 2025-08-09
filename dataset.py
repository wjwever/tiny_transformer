import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from tokenizer import Tokenizer


DATA_CACHE_DIR = 'data'
TOKENIZER_MODEL = "./data/tok5000.model"


class PretokDataset(torch.utils.data.IterableDataset):
    """从磁盘加载已预处理的分词数据，并将其以 PyTorch 张量的形式返回。"""

    def __init__(self, split, max_seq_len):
        """
        初始化数据集。

        参数:
        split: str, 数据集的分割方式（'train' 或 'test'）。
        max_seq_len: int, 最大序列长度，用于生成输入输出序列。
        """
        super().__init__()
        self.split = split  # 数据集划分（训练集或测试集）
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.sos = 1
        self.eos = 2

    def __iter__(self):
        """
        返回迭代器，按批次加载数据并生成模型输入/输出。
        """
        x_npz = np.load('feats.npz')
        y_npz = np.load('labels.npz')
        n_len = len(x_npz.files)

        sos_token = torch.tensor([self.sos], dtype=torch.int32)
        eos_token = torch.tensor([self.eos], dtype=torch.int32)

        while True:
            # 随机打乱分片文件
            idx = 0
            if self.split == "eval":
                idx = n_len - 1
            else:
                idx = random.randint(0, n_len - 2)

            x = torch.tensor(x_npz[f'arr_{idx}'].astype(np.int32))
            x =  torch.cat((sos_token, x, eos_token))

            y_head = torch.tensor(y_npz[f'arr_{idx}'].astype(np.int32))
            y_head = torch.cat((sos_token, y_head))

            y = torch.tensor(y_npz[f'arr_{idx}'].astype(np.int32))
            y = torch.cat((y, eos_token))

            yield x, y_head, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn
        )
        for x, y_head, y in dl:
            x = x.to(device, non_blocking=True).long()
            y_head = y_head.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()
            yield x, y_head, y

def collate_fn(batch):
    x_batch, y_head_batch, y_batch = zip(*batch)
    x = pad_sequence(x_batch, batch_first=True, padding_value=3)
    y_head = pad_sequence(y_head_batch, batch_first=True, padding_value=3)
    y = pad_sequence(y_batch, batch_first=True, padding_value=3)
    return x, y_head, y




if __name__ == "__main__":
    # data_loader
    batch_size = 2  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
    max_seq_len = 256  # 最大序列长度
    device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等

    iter_batches = partial(
        Task.iter_batches,  # 调用 Task 类中的 iter_batches 方法
        batch_size=batch_size,  # 每个批次的样本数量
        max_seq_len=max_seq_len,  # 每个序列的最大长度
        device=device,  # 运行模型的设备（如 GPU 或 CPU）
        num_workers=4,  # 用于数据加载的 worker 数量，0 表示在主线程中加载
    )

    train_batch_iter = iter_batches(split="train")
    X, Y_HEAD, Y = next(train_batch_iter)  # 获取第一个批次的数据
    print("X", X)
    print("Y_HEAD", Y_HEAD)
    print("Y", Y)

