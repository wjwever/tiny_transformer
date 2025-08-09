import tqdm
import torch
import torch.optim as optim
from transformer import Transformer, TransformerConfig
from torch.utils.data import Dataset, DataLoader
from dataset import Task
from functools import partial
from tokenizer import Tokenizer

# constants

enc = Tokenizer('./data/tok5000.model') # 加载分词器
args = TransformerConfig()

# helpers
max_seq_len = 256  # 最大序列长度
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等

iter_batches = partial(
    Task.iter_batches,  # 调用 Task 类中的 iter_batches 方法
    batch_size=args.batch_size,  # 每个批次的样本数量
    max_seq_len=max_seq_len,  # 每个序列的最大长度
    device=device,  # 运行模型的设备（如 GPU 或 CPU）
    num_workers=2,  # 用于数据加载的 worker 数量，0 表示在主线程中加载
)

train_batch_iter = iter_batches(split="train")
model = Transformer(args).cuda()
model.train()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

# training
iter = 0
while iter < args.max_iter:
    x, y_head, y = next(train_batch_iter)  # 获取第一个批次的数据
    _, loss = model(x, y_head, y)
    loss.backward()
    print(f'loss {iter}: {loss.item()}')
    optim.step()
    optim.zero_grad()
    iter += 1

    if iter % args.gen_interval == 0:
        model.eval()
        x, y_head, y = next(train_batch_iter)  # 获取第一个批次的数据
        x, y_head, y = x[:1], y_head[:1], y[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()
        infer_tokens = model.generate(x, start_tokens, 256)

        x_list = x.tolist()[0]
        y_list = y.tolist()[0]
        preds = infer_tokens.tolist()[0]

        print(f"x: {enc.decode(x_list)}")
        print(f"y: {enc.decode(y_list)}")
        print(f"preds: {enc.decode(preds)}")
        model.train()

    if iter % args.save_interval == 0:
        model.eval()
        ckp = f'{args.save_dir}/translate.pth'
        state_dict = model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, ckp)
        model.train()

