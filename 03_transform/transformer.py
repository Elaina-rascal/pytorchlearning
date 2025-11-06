import math
import torch
from torch import nn, functional as F
class MultiAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, key_size,query_size,value_size,num_hiddens,num_heads,dropout=0.1,bias=False):
        self.num_heads = num_heads
        self.attention = nn.functional.scaled_dot_product_attention
        super().__init__()  
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def transpose_qkv(X:torch.Tensor, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
        # num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def create_mask(self, valid_lens: torch.Tensor, seq_len: int):
        """
        valid_lens: 形状 (batch_size,)，每个元素是该样本的有效长度
        seq_len: 实际序列长度（键-值对个数）
        返回: 形状 (batch_size, 1, seq_len) 的掩码矩阵，无效位置为True（被掩码）
        """
        batch_size = valid_lens.shape[0]
        # 生成 (batch_size, seq_len) 的矩阵，每个位置表示是否超过有效长度
        mask = torch.arange(seq_len, device=valid_lens.device).expand(batch_size, seq_len) >= valid_lens.unsqueeze(1)
        return mask.unsqueeze(1)  # 扩展维度以匹配注意力权重的形状
    
    #@save
    def transpose_output(X:torch.Tensor, num_heads):
        """逆转transpose_qkv函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    def forward(self, queries:torch.Tensor, keys:torch.Tensor, values:torch.Tensor, valid_lens:torch.Tensor=None,attn_mask:torch.Tensor=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = MultiAttention.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = MultiAttention.transpose_qkv(self.W_k(keys), self.num_heads)
        values = MultiAttention.transpose_qkv(self.W_v(values), self.num_heads)
        if(attn_mask is not None):
            if valid_lens is not None:
                # 原始valid_lens形状 (batch_size,)，复制到多头
                valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)  # 形状 (batch×heads,)
                # 生成掩码矩阵：形状 (batch×heads, 1, seq_k)
                attn_mask = self.create_mask(valid_lens, keys.shape[1])  # keys.shape[1] 是 seq_k
                attn_mask = attn_mask.to(dtype=queries.dtype)  # 匹配查询的dtype

        # output的形状:(batch_size*num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, attn_mask, dropout_p=0.1, is_causal=False)

        # output_concat的形状:(batch_size，查询或者“键－值”对的个数，
        # num_hiddens)
        output_concat = MultiAttention.transpose_output(output, self.num_heads)

        return self.W_o(output_concat)
class PositionFFN(nn.Module):
    """位置前馈网络,一个多层感知机"""
    def __init__(self, num_hiddens, ffn_num_hiddens, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X:torch.Tensor):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))
class AddNorm(nn.Module):
    """残差连接后进行层归一化"""
    def __init__(self, normalized_shape, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X:torch.Tensor, Y:torch.Tensor):
        return self.ln(X + self.dropout(Y))
#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
class EncoderBlock(nn.Module):
    def __init__(self,key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout=0.1, bias=False):
        super().__init__()
        self.attention = MultiAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionFFN(ffn_num_input, ffn_num_hiddens, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X:torch.Tensor, valid_lens:torch.Tensor):
        # 多头自注意力模块
        Y = self.attention(X, X, X, valid_lens)
        X = self.addnorm1(X, Y)
        # 前馈神经网络模块
        Y = self.ffn(X)
        return self.addnorm2(X, Y)
class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout=0.1, bias=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("encoder_blk_%d",EncoderBlock(key_size,query_size,
                                                               value_size,num_hiddens,
                                                               norm_shape,ffn_num_input,
                                                               ffn_num_hiddens,num_heads,
                                                               dropout,bias))
    def forward(self, X:torch.Tensor, valid_lens:torch.Tensor):
        # 在以下代码段中，X的形状保持不变:(batch_size，序列长度，num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(
            self.embedding.embedding_dim))
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionFFN(ffn_num_input, ffn_num_hiddens,
                                   dropout)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X:torch.Tensor, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
class TransformDecorder(nn.Module):
    def __init__(self,vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout=0.1, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens, dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("decoder_blk_%d",DecoderBlock(
                key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def forward(self, X:torch.Tensor, state):
        # 在以下代码段中，X的形状保持不变:(batch_size，序列长度，num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(
            self.embedding.embedding_dim))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i,blk in enumerate(self.blks):
            X, state = blk(X, state)
            if isinstance(blk, DecoderBlock):
                # self._attention_weights[0][
                #     i] = blk.attention1.attention.attention_weights
                # # “编码器－解码器”自注意力权重
                # self._attention_weights[1][
                #     i] = blk.attention2.attention.attention_weights
                X
        return self.dense(X)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_X, enc_valid_lens)->torch.Tensor:
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        # state=[enc_outputs,enc_valid_lens,[None] * self.num_layers]
        state=self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, state)
        # return self.decoder(dec_X, dec_state)
if __name__== "__main__":
    num_hiddens, num_heads = 100, 5
    attention = MultiAttention(
        key_size=num_hiddens, query_size=num_hiddens,
        value_size=num_hiddens, num_hiddens=num_hiddens,
        num_heads=num_heads, dropout=0.1)
    print(attention.eval())
    
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y,valid_lens).shape)