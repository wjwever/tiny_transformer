import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from functools import partial
from multiprocessing import Pool

from tokenizer import Tokenizer

TOKENIZER_MODEL = "./data/tok4096.model"

class EnZhTranslationDataset(Dataset):
    def __init__(self, en_file, zh_file, en_tokenizer, zh_tokenizer, max_len=100, sort_by_length=True):
        """
        Args:
            en_file: 英文文本文件路径
            zh_file: 中文文本文件路径
            en_tokenizer: 英文分词器（如sentencepiece）
            zh_tokenizer: 中文分词器
            max_len: 最大序列长度
            sort_by_length: 是否按长度排序（提升批处理效率）
        """
        # 读取原始数据
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_lines = [line.strip() for line in f]
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f]
        
        assert len(self.en_lines) == len(self.zh_lines), "英中文本行数不匹配"
        
        # 多进程分词（加速预处理）
        with Pool(processes=4) as pool:
            self.en_data = pool.map(partial(self._tokenize, tokenizer=en_tokenizer), self.en_lines)
            self.zh_data = pool.map(partial(self._tokenize, tokenizer=zh_tokenizer), self.zh_lines)
        
        # 过滤过长句子
        self.valid_indices = [
            i for i in range(len(self.en_data)) 
            if len(self.en_data[i]) <= max_len and len(self.zh_data[i]) <= max_len
        ]
        
        # 按英文长度排序（优化批处理）
        if sort_by_length:
            self.valid_indices.sort(key=lambda i: -len(self.en_data[i]))
        else:
            random.shuffle(self.valid_indices)

        
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self._enc = zh_tokenizer
        self.pad_id = 3  # 假设0是填充符
        self._sos_token = torch.LongTensor([1])
        self._eos_token = torch.LongTensor([2])

    def _tokenize(self, text, tokenizer):
        """多进程安全的分词函数"""
        ids = tokenizer.encode(text, False, False)
        return ids

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        en_ids = torch.LongTensor(self.en_data[real_idx])
        x = torch.cat((self._sos_token, en_ids, self._eos_token))

        zh_ids = torch.LongTensor(self.zh_data[real_idx])
        y_head = torch.cat((self._sos_token, zh_ids))

        y = torch.cat((zh_ids, self._eos_token))

        return x, y_head, y

    @staticmethod
    def collate_fn(batch):
        """动态填充批次数据"""
        x, y_head, y = zip(*batch)
        
        # 按长度降序排序（适配pack_padded_sequence）
        en_lengths = torch.tensor([len(d) for d in x])
        #zh_lengths = torch.tensor([len(x) for x in zh_batch])
        en_sorted_idx = torch.argsort(en_lengths, descending=True)
        
        # 排序并填充
        x = pad_sequence(
            [x[i] for i in en_sorted_idx], 
            batch_first=True, 
            padding_value=3
        )
        y_head = pad_sequence(
            [y_head[i] for i in en_sorted_idx],
            batch_first=True,
            padding_value=3
        )
        y = pad_sequence(
            [y[i] for i in en_sorted_idx],
            batch_first=True,
            padding_value=3
        )
        
        return x, y_head, y
# 使用示例
if __name__ == "__main__":
    # 假设已有分词器
    en_tokenizer = Tokenizer(TOKENIZER_MODEL)
    zh_tokenizer = Tokenizer(TOKENIZER_MODEL)
    
    dataset = EnZhTranslationDataset(
        "/home/ww/work/data/cmn/eng.txt", "/home/ww/work/data/cmn/zh.txt",
        en_tokenizer, zh_tokenizer,
        max_len=256,
        sort_by_length=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,  # 已按长度排序，无需shuffle
        num_workers=4,  # 多进程加载
        collate_fn=EnZhTranslationDataset.collate_fn,
        pin_memory=True  # 加速GPU传输
    )
    
    for x, y_head, y in dataloader:
        print("x", x)
        print("y_head", y_head)
        print("y", y)
        pass
        #break
