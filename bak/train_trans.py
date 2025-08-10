import tqdm
import torch
import torch.optim as optim
from transformer import Transformer, TransformerConfig
from torch.utils.data import Dataset, DataLoader
from dataset import Task
from functools import partial
from tokenizer import Tokenizer
from trans_dataset import EnZhTranslationDataset

# constants

args = TransformerConfig()
enc = Tokenizer(f'./data/tok{args.vocab_size}.model') # 加载分词器

# helpers
max_seq_len = 256  # 最大序列长度
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0'等




model = Transformer(args).cuda()
model.train()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

TOKENIZER_MODEL = "./data/tok4096.model"
en_tokenizer = Tokenizer(TOKENIZER_MODEL)
zh_tokenizer = Tokenizer(TOKENIZER_MODEL)

dataset = EnZhTranslationDataset(
    "/home/ww/work/data/cmn/eng.txt", "/home/ww/work/data/cmn/zh.txt",
    en_tokenizer, 
    zh_tokenizer,
    max_len=256,
    sort_by_length=False
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,  # 已按长度排序，无需shuffle
    num_workers=4,  # 多进程加载
    collate_fn=EnZhTranslationDataset.collate_fn,
    pin_memory=True  # 加速GPU传输
)

# training
for epoch in range(10000):
    iter = 0;
    for x, y_head, y in dataloader:
        iter += 1
        x = x.to("cuda")
        y_head = y_head.to("cuda")
        y = y.to("cuda")
        if iter % args.gen_interval == 0:
            model.eval()
            x, y_head, y = x[:1], y_head[:1], y[:1]
            start_tokens = (torch.ones((1, 1)) * 1).long().cuda()
            infer_tokens = model.generate(x, start_tokens, 100)

            x_list = x.tolist()[0]
            y_list = y.tolist()[0]
            preds = infer_tokens.tolist()[0]

            print(f"x: {x_list} \n{enc.decode(x_list)}")
            print(f"y: {y_list } \n {enc.decode(y_list)}")
            print(f"preds: {preds} \n {enc.decode(preds)}")
            model.train()
        else:
            _, loss = model(x, y_head, y)
            loss.backward()
            print(f'loss {epoch} {iter}: {loss.item()}')
            optim.step()
            optim.zero_grad()


    model.eval()
    ckp = f'{args.save_dir}/translate.pth'
    state_dict = model.state_dict()
    state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
    torch.save(state_dict, ckp)
    model.train()

