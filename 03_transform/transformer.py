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
    def forward(self, queries:torch.Tensor, keys:torch.Tensor, values:torch.Tensor, valid_lens:torch.Tensor=None):
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

        attn_mask = None
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