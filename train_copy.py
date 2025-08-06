import tqdm
import torch
import torch.optim as optim
from transformer import Transformer, TransformerConfig

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 100
ENC_SEQ_LEN = 16
DEC_SEQ_LEN = 16

# helpers

def cycle():
    while True:
        prefix = torch.zeros((BATCH_SIZE, 1)).long().cuda()
        suffix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        data = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        src = torch.cat((prefix, data), 1)
        tgt = torch.cat((data, suffix), 1)
        src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().cuda()
        yield (src, tgt, src_mask)

args = TransformerConfig()
args.vocab_size = NUM_TOKENS
model = Transformer(args).cuda()


# optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask = next(cycle())

    _, loss = model(src, targets=tgt)
    loss.backward()
    print(f'loss {i}: {loss.item()}')

    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        src, tgt, src_mask = next(cycle())
        src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]

        start_tokens = (torch.zeros((1, 1)) * 1).long().cuda()
        print("start_tokens", start_tokens)

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN + 1)

        print(f"input:  ", src)
        print(f"tgt:  ", tgt)
        print(f"predicted output:  ", sample)
