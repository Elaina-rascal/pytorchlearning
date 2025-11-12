import torch.nn as nn
from torch.nn import functional as F
import torch
class LightweightAttention(nn.Module):
    """轻量级注意力模块 先行人自查询,再与车辆特征交叉查询"""
    def __init__(self, feature_dim=2, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.sq=nn.Sequential()
        self.sq.add_module('linear1',nn.Linear(feature_dim,hidden_dim*2))
        self.sq.add_module('relu',nn.ReLU())
        self.sq.add_module('linear2',nn.Linear(hidden_dim*2,hidden_dim))
        self.sq2=nn.Sequential()
        self.sq2.add_module('linear1',nn.Linear(feature_dim,hidden_dim*2))
        self.sq2.add_module('relu',nn.ReLU())
        self.sq2.add_module('linear2',nn.Linear(hidden_dim*2,hidden_dim))
        # 特征映射层（将2维特征映射到隐藏维度）
        # self.proj_p = nn.Linear(feature_dim, hidden_dim)  # 行人特征映射
        # self.proj_v = nn.Linear(feature_dim, hidden_dim)  # 车辆特征映射
        
        # 自注意力参数（查询/键/值线性变换）
        self.q_self = nn.Linear(hidden_dim, hidden_dim)
        self.k_self = nn.Linear(hidden_dim, hidden_dim)
        self.v_self = nn.Linear(hidden_dim, hidden_dim)
        
        # 交叉注意力参数（行人-车辆）
        self.q_cross = nn.Linear(hidden_dim, hidden_dim)
        self.k_cross = nn.Linear(hidden_dim, hidden_dim)
        self.v_cross = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层与残差连接
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, pedestrian_feat, vehicle_feats):
        """
        Args:
            pedestrian_feat: 行人特征 (batch_size, seq_len_p, 2)
            vehicle_feats: 车辆特征 (batch_size, seq_len_v, 2)
        Returns:
            增强后的行人特征 (batch_size, seq_len_p, 2)
        """
        residual = pedestrian_feat  # 残差连接起点
        batch_size = pedestrian_feat.shape[0]
        
        # pedestrian_feat=self.sq(pedestrian_feat)
        # vehicle_feats=self.sq(vehicle_feats)
        # 1. 特征映射到高维
        # p_hidden = self.proj_p(pedestrian_feat)  # (B, P, H)
        # v_hidden = self.proj_v(vehicle_feats)    # (B, V, H)
        p_hidden = self.sq(pedestrian_feat)
        v_hidden = self.sq2(vehicle_feats)
        # 2. 行人自查询（使用PyTorch内置scaled_dot_product_attention）
        q_self = self.q_self(p_hidden)  # (B, P, H)
        k_self = self.k_self(p_hidden)  # (B, P, H)
        v_self = self.v_self(p_hidden)  # (B, P, H)
        # 自注意力计算（无掩码）
        self_attn_out = F.scaled_dot_product_attention(
            q_self, k_self, v_self, 
            attn_mask=None, 
            is_causal=True if self.training else False,
            dropout_p=self.dropout.p if self.training else 0.0
        )  # (B, P, H)
        
        # 3. 与车辆特征交叉查询（行人查询，车辆提供键值）
        q_cross = self.q_cross(self_attn_out)  # (B, P, H)
        k_cross = self.k_cross(v_hidden)       # (B, V, H)
        v_cross = self.v_cross(v_hidden)       # (B, V, H)
        # 交叉注意力计算（无掩码）
        cross_attn_out = F.scaled_dot_product_attention(
            q_cross, k_cross, v_cross, 
            attn_mask=None, 
            is_causal=True if self.training else False,
            dropout_p=self.dropout.p if self.training else 0.0
        )  # (B, P, H)
        
        # 4. 输出映射与残差连接
        output = self.output_proj(cross_attn_out)  # (B, P, 2)
        output = residual +output  # 残差+层归一化
        
        return output
import torch
import torch.nn as nn

class AmplifiedResidualLoss(nn.Module):
    def __init__(self, threshold=0.05, scale=10.0, penalty_weight=100.0):
        super().__init__()
        self.threshold = threshold  # 预期最大误差
        self.scale = scale  # 残差放大系数
        self.penalty_weight = penalty_weight  # 超阈值惩罚权重

    def forward(self, pred, target):
        # 计算残差（L1残差，更适合误差约束）
        residual = torch.abs(pred - target)
        
        # 1. 放大残差（增强对小误差的敏感性）
        scaled_residual = residual * self.scale
        
        # 2. 对超过阈值的残差添加额外惩罚
        penalty = torch.where(
            residual > self.threshold,
            (residual - self.threshold) * self.penalty_weight,
            torch.zeros_like(residual)
        )
        
        # 总损失 = 放大后的残差均值 + 超阈值惩罚均值
        total_loss = torch.mean(scaled_residual) + torch.mean(penalty)
        return total_loss
if __name__== "__main__":
    # 测试轻量级注意力模块
    batch_size = 4
    seq_len_p = 5  # 行人序列长度
    seq_len_v = 8  # 车辆序列长度
    feature_dim = 2

    pedestrian_feat = torch.randn(batch_size, seq_len_p, feature_dim)
    vehicle_feats = torch.randn(batch_size, seq_len_v, feature_dim)

    model = LightweightAttention(feature_dim=feature_dim, hidden_dim=16)
    enhanced_pedestrian_feat = model(pedestrian_feat, vehicle_feats)

    print("输入行人特征形状:", pedestrian_feat.shape)
    print("输入车辆特征形状:", vehicle_feats.shape)
    print("输出增强后行人特征形状:", enhanced_pedestrian_feat.shape)