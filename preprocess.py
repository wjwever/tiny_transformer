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
TOKENIZER_MODEL = "./data/tok8092.model"

def load_text_from_files(file_path):
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data.extend(file.readlines())
    return text_data


# 定义分片处理函数
def process_shard(shard_id, shard, vocab_size, tokenizer_model_path, add_bos, add_eos):
    """
    处理数据分片，将其中的文本进行分词并保存为二进制文件。
    
    参数:
    args: tuple, 包含分片ID和分片文件名
    vocab_size: int, 词汇表大小，用于决定输出文件存储路径
    """
    # 提取分片ID和文件名
    #shard_id, shard = args

    # 初始化分词器
    enc = Tokenizer(tokenizer_model_path)
    
    # 打开并读取当前分片的文件
    text_data = load_text_from_files(shard)
    
    # 用于保存所有的分词后的token
    all_tokens = []
    
    # 遍历每一个例子，tqdm显示进度条
    for text in tqdm(text_data, position=shard_id):
        # 提取故事文本，并去除首尾空白字符
        text = text.strip()  # 去掉首尾空白字符
        
        # 对文本进行编码，使用BOS（开始标志）但不使用EOS（结束标志）
        tokens = enc.encode(text, bos=add_bos, eos=add_eos)
        # 将当前文本的token添加到总token列表
        #all_tokens.extend(tokens)
        all_tokens.append(np.array(tokens, dtype=np.int16))

    
    # 将token以二进制形式保存
    np.savez(f'{os.path.basename(shard)}.npz', *all_tokens)
    
    # 计算平均序列长度（以BOS标记`1`分隔的序列）
    #avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    #print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


# 定义预处理函数，用于对多个数据分片进行批量处理
def pretokenize(vocab_size):
    """
    预处理所有的数据分片，并将分词后的数据保存为二进制文件。
    
    参数:
    vocab_size: int, 词汇表大小，用于决定输出文件存储路径
    """
    # 如果词汇表大小大于0，则创建对应的保存目录
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # 使用partial函数将vocab_size绑定到process_shard函数
    #fun = partial(process_shard, vocab_size=vocab_size, tokenizer_model_path=TOKENIZER_MODEL)
    
    # 使用进程池并行处理每个分片
    #with ProcessPoolExecutor() as executor:
    #    executor.map(fun, enumerate(shard_filenames))

    data_dir = "/home/ww/work/data/trans/"
    process_shard(0, data_dir + "tiny.zh",  vocab_size, TOKENIZER_MODEL, True, True)
    process_shard(1, data_dir + "tiny.en",  vocab_size, TOKENIZER_MODEL, False, False)
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """从磁盘加载已预处理的分词数据，并将其以 PyTorch 张量的形式返回。"""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        """
        初始化数据集。

        参数:
        split: str, 数据集的分割方式（'train' 或 'test'）。
        max_seq_len: int, 最大序列长度，用于生成输入输出序列。
        vocab_size: int, 词汇表的大小。
        vocab_source: str, 词汇表的来源（'llama2' 或 'custom'）。
        """
        super().__init__()
        self.split = split  # 数据集划分（训练集或测试集）
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.vocab_size = vocab_size  # 词汇表大小
        self.vocab_source = vocab_source  # 词汇表来源
        self.sos = 1
        self.eos = 2

    def __iter__(self):
        """
        返回迭代器，按批次加载数据并生成模型输入/输出。
        """
        x_npz = np.load('tiny.zh.npz')
        y_npz = np.load('tiny.en.npz')
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
            y_head = torch.tensor(y_npz[f'arr_{idx}'].astype(np.int32))
            y_head = torch.cat((sos_token, y_head))

            y = torch.tensor(y_npz[f'arr_{idx}'].astype(np.int32))
            y = torch.cat((y, eos_token))

            if x.shape[0] < 256 and y.shape[0] < 256:
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
    # tokenize
    # pretokenize(vocab_size=8092)

    # data_loader
    batch_size = 2  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
    max_seq_len = 256  # 最大序列长度
    vocab_size = 8192  # 自定义词汇表大小
    vocab_source= 'custom'
    device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等

    iter_batches = partial(
        Task.iter_batches,  # 调用 Task 类中的 iter_batches 方法
        batch_size=batch_size,  # 每个批次的样本数量
        max_seq_len=max_seq_len,  # 每个序列的最大长度
        vocab_size=vocab_size,  # 词汇表大小
        vocab_source=vocab_source,  # 词汇表来源（如 llama2 或 custom）
        device=device,  # 运行模型的设备（如 GPU 或 CPU）
        num_workers=0,  # 用于数据加载的 worker 数量，0 表示在主线程中加载
    )

    train_batch_iter = iter_batches(split="train")
    X, Y_HEAD, Y = next(train_batch_iter)  # 获取第一个批次的数据
    print("X", X)
    print("Y_HEAD", Y_HEAD)
    print("Y", Y)

