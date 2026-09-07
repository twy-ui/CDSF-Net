import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv

class ChannelAttention(nn.Module):#通道注意力
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)# 全局平均池化: 将每个通道的 H×W 压缩为 1×1
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 全局最大池化: 同上，但保留最大响应位置信息
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) # 共享MLP (使用1×1卷积实现)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x # 保存残差连接，用于最终相乘
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))# 平均池化分支: 捕获全局统计信息（背景/整体趋势）
        # 步骤分解: x → (B,C,1,1) → (B,C/16,1,1) → ReLU → (B,C,1,1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))# 最大池化分支: 捕获最显著特征（目标/边缘响应）
        out = avg_out + max_out# 融合两种统计信息
        return self.sigmoid(out) * res# 生成通道注意力权重，并与原始特征相乘

class SpatialAttention(nn.Module):#空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'# 只允许3或7的卷积核，确保padding后尺寸不变
        padding = 3 if kernel_size == 7 else 1# 计算padding: 7→3, 3→1，保持(H,W)不变
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # 关键：输入通道=2（avg+max），输出通道=1（空间注意力图）
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x# 保存原始特征，用于最终相乘 (B×C×H×W)
        avg_out = torch.mean(x, dim=1, keepdim=True)# 沿通道维度(dim=1)计算平均值，保留通道维度(keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)# 沿通道维度取最大值，保留通道维度
        x = torch.cat([avg_out, max_out], dim=1) # 在通道维度拼接: [avg, max] → (B, 2, H, W)
        x = self.conv1(x) # 卷积融合: 2通道 → 1通道，生成空间注意力图
        return self.sigmoid(x) * x_source# Sigmoid生成(0,1)权重，与原始特征相乘

class h_sigmoid(nn.Module):#计算高效的Sigmoid近似
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttiton(nn.Module):#坐标注意力，用于编码空间位置信息
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        # 水平方向池化: 对每个通道，沿宽度压缩为1，保留高度
        # 输出: (B, C, H, 1) - "每行平均特征"
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向池化: 对每个通道，沿高度压缩为1，保留宽度
        # 输出: (B, C, 1, W) - "每列平均特征"
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction) # 中间通道数: 压缩率控制，最少8个通道（防止过小）

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)# 1×1卷积降维: 将拼接后的特征压缩
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()# 使用之前讲解的 h_swish 激活

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # 两个分支: 分别生成高度和宽度的注意力

    def forward(self, x):
        identity = x # 保存原始输入用于最终相乘 (B,C,H,W)

        n, c, h, w = x.size()# 获取维度信息
        # ========== 步骤1: 坐标信息编码 ==========
        # 水平池化: 沿W方向平均，得到每行的全局特征
        # x_h: (B, C, H, 1) - "高方向的特征分布"
        x_h = self.pool_h(x)
        # 垂直池化: 沿H方向平均，然后转置使W变H
        # x_w: (B, C, 1, W) → permute → (B, C, W, 1) - "宽方向的特征分布"
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # ========== 步骤2: 拼接并处理 ==========
        # 在高度维度拼接: (B,C,H,1) + (B,C,W,1) = (B,C,H+W,1)
        # 关键: 两个方向的信息在同一特征图中交互！
        y = torch.cat([x_h, x_w], dim=2)
        # 降维 + 批归一化 + 非线性激活
        y = self.conv1(y)# (B,C,H+W,1) → (B,mip,H+W,1)
        y = self.bn1(y)
        y = self.act(y)# h_swish激活
        # ========== 步骤3: 分离并生成注意力 ==========
        # 沿高度维度切分: (B,mip,H+W,1) → (B,mip,H,1) 和 (B,mip,W,1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 转置回去: (B,mip,W,1) → (B,mip,1,W)
        x_w = x_w.permute(0, 1, 3, 2)
        # 分别生成两个方向的注意力图 (使用标准sigmoid)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # ========== 步骤4: 应用注意力 ==========
        # 原始特征 × 宽度注意力 × 高度注意力
        # 广播机制: (B,C,H,W) × (B,C,1,W) × (B,C,H,1) = (B,C,H,W)
        out = identity * a_w * a_h

        return out

class HFFE(nn.Module):#层级特征融合编码器
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(HFFE, self).__init__()
        # in_channel 是元组 (C_low, C_high)，分别表示低层和高层通道数
        feature_low_channel, feature_high_channel = in_channel
        # ========== 空间权重生成器（SWM生成） ==========
        # 低层分支: C1 → C1/16 → 1 → Sigmoid
        # 生成单通道的空间权重图，用于校准高层特征
        self.conv_block_low = nn.Sequential(
            Conv(feature_low_channel, feature_low_channel // 16, kernel_size),
            nn.Conv2d(feature_low_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        # 高层分支: C2 → C2/16 → 1 → Sigmoid
        # 生成单通道的空间权重图，用于校准低层特征
        self.conv_block_high = nn.Sequential(
            Conv(feature_high_channel, feature_high_channel // 16, kernel_size),
            nn.Conv2d(feature_high_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        # ========== 特征变换卷积 ==========
        # 用于残差分支的特征处理
        self.conv1 = Conv(feature_low_channel, out_channel, 1)# 处理低层
        self.conv2 = Conv(feature_high_channel, out_channel, 1)# 处理高层
        self.conv3 = Conv(feature_low_channel + feature_high_channel, out_channel, 1) # 处理融合特征
        # ========== 上采样模块 ==========
        # 将高层特征分辨率翻倍，匹配低层
        self.Up_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # ========== 空间注意力模块 ==========
        # 论文中的SAM，用于预处理，突出目标区域
        self.feature_low_sa = SpatialAttention()
        self.feature_high_sa = SpatialAttention()
        # ========== 坐标注意力模块 ==========
        # 论文中的CoordAtt，生成SWM_fuse
        self.ca = CoordAttiton(out_channel,out_channel)
        # ========== 最终融合卷积 ==========
        # 将两个分支的输出融合为最终F_HFFE
        self.conv_final = Conv(out_channel * 2, out_channel, 1)

    def forward(self, x):
        x_low, x_high = x# x 是元组 (x_low, x_high)
        # 获取维度信息
        b1, c1, w1, h1 = x_low.size()# 低层: (B, C1, H, W)
        b2, c2, w2, h2 = x_high.size()# 高层: (B, C2, H/2, W/2)
        # ========== 步骤1: 分辨率对齐 ==========
        # 如果高层分辨率不等于低层，进行上采样
        if (w1, h1) != (w2, h2):
            # 双线性插值上采样，align_corners=False避免边界 artifact
            x_high = F.interpolate(x_high, (w1, h1), mode='bilinear', align_corners=False)
        # 保存原始特征，用于后续残差连接
        source_low = x_low
        source_high = x_high
        # ========== 步骤2: 空间注意力预处理（SAM） ==========
        # 论文公式(11): F'_low = CBR₃(S_att(F_low))
        x_low = self.feature_low_sa(x_low)# 低层经过SAM精炼
        x_high = self.feature_high_sa(x_high)# 高层经过SAM精炼
        # ========== 步骤3: 生成空间权重矩阵（SWM） ==========
        # 论文公式(12): SWM_low = σ(CBR₁(F'_low))
        # 这里用conv_block生成单通道权重图
        x_low_map = self.conv_block_low(x_low)# (B, 1, H, W) - 用于校准高层
        x_high_map = self.conv_block_high(x_high)# (B, 1, H, W) - 用于校准低层
        # ========== 步骤4: 交叉校准与融合 ==========
        # 论文公式(13)(14):
        # F''_low = F_low ⊗ SWM_high
        # F''_high = F_high ⊗ SWM_low

        # 但代码实现有所不同，采用更复杂的融合策略:
        # source_low * x_high_map: 低层特征用高层权重校准
        # source_high * x_low_map: 高层特征用低层权重校准
        # 然后在通道维度拼接
        x_mix = torch.cat([source_low * x_high_map, source_high * x_low_map], 1)
        # x_mix: (B, C1+C2, H, W)

        # ========== 步骤5: 坐标注意力生成SWM_fuse ==========
        # 论文公式(15): SWM_fuse = σ(CA(Conv₁×₁[F'_low, F'_high]))

        # 先用conv3降维，再通过CoordAtt
        x_ca = torch.sigmoid(self.ca(self.conv3(x_mix)))# (B, C1+C2, H, W) → (B, out_channel, H, W)
        # x_ca 就是 SWM_fuse，用于后续加权

        # ========== 步骤6: 残差增强与最终加权 ==========
        # 论文公式(16):
        # F_HFFE = Conv₁×₁[SWM_fuse⊗CBR₁(F'_low+F_low), SWM_fuse⊗CBR₁(F'_high+F_high)]

        # 残差连接: F' + F，保留原始信息
        x_low_att = x_ca * self.conv1((source_low + x_low))# 低层分支
        # source_low + x_low: 原始低层 + SAM精炼后的低层
        # conv1: 降维到out_channel
        # x_ca * : 用SWM_fuse加权
        x_high_att = x_ca * self.conv2((source_high + x_high))# 高层分支，同理

        out = self.conv_final(torch.cat([x_low_att, x_high_att], 1))
        # ========== 步骤7: 最终融合输出 ==========
        # 拼接两个分支，再通过1×1卷积融合
        # out: (B, out_channel, H, W) = F_HFFE
        return out