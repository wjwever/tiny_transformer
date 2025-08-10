import tqdm
import torch
import torch.optim as optim
from transformer import Transformer, TransformerConfig
from torch.nn.utils.rnn import pad_sequence

# constants

ENC_SEQ_LEN = 32

# helpers
args = TransformerConfig()

#def cycle():
#    while True:
#        prefix = torch.ones((args.batch_size, 1)).long().cuda()
#        suffix = 2 * torch.ones((args.batch_size, 1)).long().cuda()
#        data = torch.randint(4, args.vocab_size - 1, (args.batch_size, ENC_SEQ_LEN)).long().cuda()
#        src = torch.cat((prefix, data), 1)
#        tgt = torch.cat((torch.flip(data, dims=[1]), suffix), 1)
#        src_mask = torch.ones(args.batch_size, src.shape[1]).bool().cuda()
#        yield src, tgt, src_mask


def cycle():
    """生成随机批次数据，每样本有效长度不等，用3填充"""
    data_len = torch.randint(1, ENC_SEQ_LEN + 1, (args.batch_size,))  # [1, ENC_SEQ_LEN]
    
    # 生成数据 + 填充 (优化版，避免循环)
    data = [torch.randint(4, args.vocab_size, (L,))  for _, L in enumerate(data_len)]

    sos_token = torch.ones((1,), dtype=torch.long)  # prefix=1
    eos_token = torch.full((1,), 2, dtype=torch.long)  # suffix=2


    y_head = [torch.cat((sos_token, d), dim=0) for d in data]
    y = [torch.cat((d, eos_token), dim=0) for d in data]
    x = [torch.cat((sos_token, d, eos_token), dim = 0) for d in data]


    x = pad_sequence(x, batch_first=True, padding_value=3).cuda()
    y_head = pad_sequence(y_head, batch_first=True, padding_value=3).cuda()
    y = pad_sequence(y, batch_first=True, padding_value=3).cuda()

    yield x, y_head, y


model = Transformer(args).cuda()


# optimizer
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

# training
for i in tqdm.tqdm(range(args.max_iter), mininterval=10., desc='training'):
    model.train()

    x, src, tgt = next(cycle())
    _, loss = model(x, src, targets=tgt)
    loss.backward()
    print(f'loss {i}: {loss.item()}')

    optim.step()
    optim.zero_grad()

    if i != 0 and i % args.gen_every == 0:
        x, src, tgt = next(cycle())
        x, src, tgt = x[:1], src[:1], tgt[:1]

        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        sample = model.generate(x, start_tokens, ENC_SEQ_LEN + 1)

        print(f"x:  ", x)
        print("start", start_tokens)
        print(f"target:  ", tgt)
        print(f"predicted:  ", sample)
