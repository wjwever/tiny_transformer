import glob
import json
import os
from tqdm import tqdm
import requests
import sentencepiece as spm
import argparse
from tokenizer import Tokenizer
import numpy as np

DATA_CACHE_DIR = 'data'
# 定义分片处理函数
#
def load_text_from_file(file_path):
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data.extend(file.readlines())
    return text_data

def process_token(args):
    """
    处理数据分片，将其中的文本进行分词并保存为二进制文件。
    
    参数:
    args: tuple, 包含分片ID和分片文件名
    vocab_size: int, 词汇表大小，用于决定输出文件存储路径
    """
    # 初始化分词器
    TOKENIZER_MODEL = f"data/tok{args.vocab_size}.model"
    enc = Tokenizer(TOKENIZER_MODEL)
    
    # 打开并读取当前分片的文件
    feats = load_text_from_file(args.feat_txt)
    labels = load_text_from_file(args.label_text)
    
    # 用于保存所有的分词后的token
    all_feat_tokens = []
    all_label_tokens = []
    
    # 遍历每一个例子，tqdm显示进度条
    for feat, label in tqdm(zip(feats, labels)):
        # 提取故事文本，并去除首尾空白字符
        feat = feat.strip()
        label = label.strip()
        
        feat_tokens = enc.encode(feat, bos=False, eos=False)
        label_tokens = enc.encode(label, bos=False, eos=False)
        # 将当前文本的token添加到总token列表
        #all_tokens.extend(tokens)
        if len(feat_tokens) < 256 and len(label_tokens) < 256:
            all_feat_tokens.append(np.array(feat_tokens, dtype=np.int16))
            all_label_tokens.append(np.array(label_tokens, dtype=np.int16))

    
    # 将token以二进制形式保存
    np.savez(f'feats.npz', *all_feat_tokens)
    np.savez(f'labels.npz', *all_label_tokens)
    print(f"feats: {len(all_feat_tokens)}, labels: {len(all_label_tokens)}")


def train_vocab(args, num_shards: int=20):
    """
    vocab_size: int, 词汇表的大小，决定分词器的词汇量。
    num_shards: int, 用于加快词汇表训练的效率，指定要处理的分片数量。
    """
    # 确保词汇表大小为正数
    vocab_size = args.vocab_size
    data_dir = args.corpus_dir
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece 模型的前缀路径，将用于保存分词器
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 1) 将多个分片中的文本导出为单个文本文件 tiny.txt
    #tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    #data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    #shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # 创建 tiny.txt 文件并写入指定数量的分片中的文本
    #print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    #with open(tiny_file, "w", encoding="utf-8") as of:
    #    # 遍历前 num_shards 个分片
    #    for shard in tqdm(shard_filenames[:num_shards]):
    #        with open(shard, "r") as f:
    #            data = json.load(f)  # 读取分片中的JSON数据
    #        # 遍历每个例子，将其中的故事文本写入 tiny.txt 文件
    #        for example in data:
    #            text = example["story"]
    #            text = text.strip()  # 去除文本首尾的空白字符
    #            of.write(text + "\n")  # 每个文本写入一行

    ## 输出生成的 tiny.txt 文件的大小
    #print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 使用 SentencePiece 训练分词器
    inputs = [args.feat_txt, args.label_text]
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=inputs,         # 输入文件为之前生成的 tiny.txt
        model_prefix=prefix,     # 模型前缀路径
        model_type="bpe",        # 使用 Byte-Pair Encoding (BPE) 训练分词器
        vocab_size=vocab_size,   # 词汇表大小
        self_test_sample_size=0, # 自测样本大小设置为 0
        input_format="text",     # 输入文件格式为纯文本
        character_coverage=1.0,  # 覆盖所有字符（包括非常见字符）
        num_threads=os.cpu_count(),  # 使用 CPU 的线程数
        split_digits=True,       # 拆分数字
        allow_whitespace_only_pieces=True,  # 允许仅由空格组成的词元
        byte_fallback=True,      # 启用字节级回退
        unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
        normalization_rule_name="identity",  # 使用“identity”归一化规则
        user_defined_symbols=["<pad>"],
        pad_id=3,  # 显式指定 pad_id（需与后续 PyTorch 对齐）
    )

    # 输出模型保存的路径
    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--download", type=bool, default=True, help="download the dataset")
    parser.add_argument("--vocab_size", type=int, default=5000, help="vocab size")
    parser.add_argument("--corpus_dir", type=str, default="/home/ww/work/data/trans_ted/", help="corpus dir")
    parser.add_argument("--feat_txt", type=str, default="/home/ww/work/data/trans_ted/TED2013.en-zh.en", help="")
    parser.add_argument("--label_text", type=str, default="/home/ww/work/data/trans_ted/TED2013.en-zh.zh", help="")
    args = parser.parse_args()
    #train_vocab(args)
    process_token(args) 

