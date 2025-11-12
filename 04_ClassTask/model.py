import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch

class LightweightAttention(nn.Module):
    """轻量级注意力模块（仅保留行人自查询）"""
    def __init__(self, feature_dim=2, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 行人特征映射序列
        self.sq = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.attn=nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1,dropout=dropout,batch_first=True)
        
        # # 自注意力参数（查询/键/值线性变换）
        # self.q_self = nn.Linear(hidden_dim, hidden_dim)
        # self.k_self = nn.Linear(hidden_dim, hidden_dim)
        # self.v_self = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层与残差连接
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, pedestrian_feat, vehicle_feats=None):  # 车辆特征变为可选参数
        """
        Args:
            pedestrian_feat: 行人特征 (batch_size, seq_len_p, 2)
            vehicle_feats: 车辆特征（仅保留参数位置，实际不使用）
        Returns:
            增强后的行人特征 (batch_size, seq_len_p, 2)
        """
        residual = pedestrian_feat  # 残差连接起点
        
        # 1. 特征映射到高维
        p_hidden = self.sq(pedestrian_feat)  # (B, P, H)
        
        # 2. 行人自注意力计算
        X,weights=self.attn(p_hidden,p_hidden,p_hidden, need_weights=True)
        #残差连接与输出映射
        output = self.output_proj(self.dropout(X)) + residual

        return output,weights
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
class SequenceXYNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, seq_len, 2)，其中最后一维[0]是x，[1]是y
        # 计算每个序列的x和y的均值（按seq_len维度求平均）
        mean = x.mean(dim=1, keepdim=True)  # 形状: (batch_size, 1, 2)，分别对应x和y的均值
        # 计算每个序列的x和y的方差
        var = x.var(dim=1, keepdim=True, unbiased=False)  # 形状: (batch_size, 1, 2)，分别对应x和y的方差
        scale_factor = 0.5  # 可根据需要调整（越小放大越明显）
        # 对x和y分别归一化：(x - mean_x)/sqrt(var_x + eps)，(y - mean_y)/sqrt(var_y + eps)
        x_normed = (x - mean) / torch.sqrt(var*scale_factor + self.eps)
        return x_normed
class GRUEnhancer(nn.Module):
    """基于GRU的特征增强模块，适合输入输出变化较小的场景"""
    def __init__(self, feature_dim=2, hidden_dim=16, dropout=0.0, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.ln = SequenceXYNorm(eps=eps)
        # 输入特征映射
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # GRU层 - 更适合处理序列变化较小的情况
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # 单向GRU更适合预测任务
        )
        
        # 输出映射与残差连接
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        # 增加跳跃连接以保留原始特征
        self.skip_weight = nn.Parameter(torch.tensor(0.5))  # 控制跳跃连接权重

    def forward(self, pedestrian_feat, vehicle_feats=None):
        """
        Args:
            pedestrian_feat: 行人特征 (batch_size, seq_len_p, 2)
            vehicle_feats: 车辆特征（可选，未使用）
        Returns:
            增强后的行人特征 (batch_size, seq_len_p, 2)
        """
        residual = pedestrian_feat  # 残差连接
        #增加归一化
        pedestrian_feat = self.ln(pedestrian_feat)
        # 1. 特征映射到高维
        p_hidden = self.input_proj(pedestrian_feat)  # (B, P, H)
        
        # 2. GRU处理序列
        gru_out, _ = self.gru(p_hidden)  # (B, P, H)
        
        # 3. 输出映射与残差融合
        output = self.output_proj(self.dropout(gru_out))
        output=output+residual
        # 4. 加权残差连接（对于变化小的特征更友好）
        
        return output, None  # 无注意力权重，返回None
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