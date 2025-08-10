import torch
import math
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    n_embd: int = 256              # 嵌入维度
    n_heads: int =  8              # 头数
    n_hidden_dim: int  =  256
    dropout: float = 0.1
    #max_seq_len: int = 512
    vocab_size: int = 4096
    block_size: int = 300
    n_layer: int = 6

    #tokenizer
    sos_idx : int = 1
    eos_idx : int = 2
    pad_idx : int = 3

    # data
    batch_size : int = 32
    lr : float = 1e-4
    max_iter : int = 100000000
    gen_interval: int = 100
    save_interval = 1000
    save_dir = "."

    #


class MultiHeadAttention(nn.Module):

    def __init__(self, args: TransformerConfig, is_causal=False, is_encoder = True):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.n_embd % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        #model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.n_embd // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.wk = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.wv = nn.Linear(args.n_embd, args.n_embd, bias=False)
        # 输出权重矩阵，维度为 dim x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(args.n_embd, args.n_embd, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal
        self.is_encoder = is_encoder

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.block_size, args.block_size), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, q.shape[1], self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, k.shape[1], self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, v.shape[1], self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)  # batch, head, tk, head_dim

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, Tq, hs) x (B, nh, hs, Tk) -> (B, nh, Tq, Tk)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        #if self.is_causal:
        #    assert hasattr(self, 'mask')
        #    # 这里截取到序列长度，因为有些序列可能比 block_size 短
        #    scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # mask (batch, t) -> (batch, 1, 1, t)
       
        #print("encoder" if self.is_encoder else "decoder", self.is_causal, scores.shape,  mask.shape)
        scores = scores + mask
        # 计算 softmax，维度为 (B, nh, T, T)
        #scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = F.softmax(scores.float(), dim=-1)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #print("encoder" if self.is_encoder else "decoder", scores.shape, xv.shape)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 线性矩阵做映射
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, args):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(args.n_embd, args.n_hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(args.n_hidden_dim, args.n_embd, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args)

    def forward(self, x, x_mask):
        # Layer Norm
        x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(x, x, x, x_mask)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out

class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, x_mask):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    '''Decoder 层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True, is_encoder=False)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False, is_encoder = False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args)

    def forward(self, x, x_mask, enc_out, enc_mask):
        # Layer Norm
        x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(x, x, x, x_mask)
        # 多头注意力
        x = self.attention_norm_2(x)
        h = x + self.attention.forward(x, enc_out, enc_out, enc_mask)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, x_mask, enc_out, enc_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, x_mask, enc_out, enc_mask)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    '''整体模型'''

    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            emb=nn.Embedding(args.vocab_size, args.n_embd),
            pe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    '''统计所有参数的数量'''

    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''

    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, x, tokens, max_len):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        _, t = x.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"
        x_mask = torch.where(x == self.args.pad_idx, torch.tensor(float('-inf')), torch.tensor(0.0))
        x_mask = x_mask[:, None, None, :].cuda()  # 使用 None 或 np.newaxis
        y_mask = torch.full((1, 1, self.args.block_size, self.args.block_size), float("-inf"))
        y_mask = torch.triu(y_mask, diagonal=1).cuda()


        tok_emb = self.transformer.emb(x)
        #print("tok_emb", tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.pe(tok_emb)
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        #print("x after wpe:", x.size())
        enc_out = self.transformer.encoder(x, x_mask)
        #print("enc_out:", enc_out.size())

        # 再通过 Decoder
        for i in range(max_len):
            b, y_len = tokens.size()
            yy_mask = y_mask[:, :, :y_len, :y_len]

            y = self.transformer.emb(tokens)
            y_pe = self.transformer.pe(y)

            y = self.transformer.decoder(y_pe, yy_mask, enc_out, x_mask)
            logits = self.lm_head(y[:, [-1], :])  # note: using list [-1] to preserve the time dim
            predicted_ids = torch.argmax(logits, dim=-1) 
            tokens = torch.cat((tokens, predicted_ids), dim=1)
            if predicted_ids.item() == self.args.eos_idx:
                print("hit eos break")
                break
        return tokens




    '''前向计算函数'''

    def forward(self, x, y_head, targets):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = x.device
        _, t = x.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # make x mask
        x_mask = torch.where(x == self.args.pad_idx, torch.tensor(float('-inf')), torch.tensor(0.0))
        x_mask = x_mask[:, None, None, :].cuda()  # 使用 None 或 np.newaxis
        y_mask = torch.full((1, 1, self.args.block_size, self.args.block_size), float("-inf"))
        y_mask = torch.triu(y_mask, diagonal=1).cuda()
        _, y_len = y_head.size()
        y_mask = y_mask[:, :, :y_len, :y_len]

        x = self.transformer.emb(x)
        x = self.transformer.pe(x)
        x = self.transformer.drop(x)
        enc_out = self.transformer.encoder(x, x_mask)

        y_head = self.transformer.emb(y_head)
        y_head = self.transformer.pe(y_head)
        y_head = self.transformer.drop(y_head)

        out = self.transformer.decoder(y_head, y_mask, enc_out, x_mask)

        # 训练阶段，如果我们给了 targets，就计算 loss
        # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
        logits = self.lm_head(out)
        # 再跟 targets 计算交叉熵
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.args.pad_idx)

        return logits, loss

