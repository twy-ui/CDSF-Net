import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

class DWM_MSA(nn.Module):
    def __init__(
            self,
            dim,    # 输入通道数 C
            window_size1=(4, 4),  # 小窗口尺寸 (论文中8×8，这里是4×4)
            window_size2=(10, 10), # 大窗口尺寸 (论文中16×16，这里是10×10)
            dim_head=32,  # 每个头的维度
            heads=2, # 注意力头数
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子: 1/√d，防止softmax梯度消失
        self.window_size1 = window_size1
        self.window_size2 = window_size2

        # position embedding
        # seq_l1 = window_size1[0] * window_size1[1]
        # self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l1, seq_l1))
        # h, w = 128 // self.heads, 128 // self.heads
        # seq_l2 = h * w * 4 // seq_l1
        # self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l2, seq_l2))
        # seq_l3 = window_size2[0] * window_size2[1]
        # self.pos_emb3 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l3, seq_l3))
        # h, w = 128 // self.heads, 128 // self.heads
        # seq_l4 = h * w * 4 // seq_l3
        # self.pos_emb4 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l4, seq_l4))

        # trunc_normal_(self.pos_emb1)
        # trunc_normal_(self.pos_emb2)
        # trunc_normal_(self.pos_emb3)
        # trunc_normal_(self.pos_emb4)

        inner_dim = dim_head * heads   # 总投影维度 = 32 × 2 = 64
        self.to_q = nn.Linear(dim, inner_dim, bias=False) # C → 64
        self.to_k = nn.Linear(dim, inner_dim, bias=False) # C → 64
        self.to_v = nn.Linear(dim, inner_dim, bias=False) # C → 64
        self.to_out = nn.Linear(inner_dim, dim) # 64 → C，输出投影

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] → [B, H, W, C]
        b, h, w, _ = x.shape  # 提取维度
        w_size1 = self.window_size1
        w_size2 = self.window_size2
        assert h % w_size1[0] == 0 and w % w_size1[1] == 0, 'fmap dimensions must be divisible by the window size 1'
        assert h % w_size2[0] == 0 and w % w_size2[1] == 0, 'fmap dimensions must be divisible by the window size 2'

        q = self.to_q(x)  # [B, H, W, inner_dim] = [B, H, W, 64]
        k = self.to_k(x)
        v = self.to_v(x)
        _, _, _, c = q.shape  # c = 64 = inner_dim
        # 通道四等分: 每份 64/4 = 16 维
        q1, q2, q3, q4 = q[:, :, :, :c // 4], q[:, :, :, c // 4:c // 2], \
                         q[:, :, :, c // 2:c // 4 * 3], q[:, :, :, c // 4 * 3:] # [B, H, W, 16]
        k1, k2, k3, k4 = k[:, :, :, :c // 4], k[:, :, :, c // 4:c // 2], \
                         k[:, :, :, c // 2:c // 4 * 3], k[:, :, :, c // 4 * 3:]
        v1, v2, v3, v4 = v[:, :, :, :c // 4], v[:, :, :, c // 4:c // 2], \
                         v[:, :, :, c // 2:c // 4 * 3], v[:, :, :, c // 4 * 3:]
        # local branch of window size 1
        # 1. 窗口划分: [B, H, W, 16] → [B, num_windows, window_pixels, 16]
        q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q1, k1, v1))
        # 输出: [B, (H/4)×(W/4), 4×4, 16] = [B, num_windows, 16, 16]
        # 2. 多头分割: [B, n, mm, 16] → [B, n, heads, mm, dim_head]
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q1, k1, v1))
        # 输出: [B, num_windows, 2, 16, 8]  (heads=2, dim_head=8? 不对，应该是16/2=8)
        q1 *= self.scale  # 缩放: q / √d
        # 3. 计算注意力矩阵: Q×K^T
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        # [B, n, heads, mm, d] × [B, n, heads, mm, d] → [B, n, heads, mm, mm]
        # mm = 16 (4×4窗口像素数)，sim1: [B, n, 2, 16, 16]
        # 4. 加位置编码（已注释掉）
        # sim1 = sim1 + self.pos_emb1
        # 5. Softmax归一化
        attn1 = sim1.softmax(dim=-1) # 对最后一个维度（key维度）归一化
        # 6. 加权求和: Attention × V
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        # [B, n, heads, mm, mm] × [B, n, heads, mm, d] → [B, n, heads, mm, d]
        # 7. 合并多头
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')
        # [B, n, heads, mm, d] → [B, n, mm, heads×d] = [B, n, 16, 16]

        # non-local branch of window size 1
        # 2. 🔥关键: Shuffle操作 - 交换维度实现跨窗口交互
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q2, k2, v2))
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        # [B, n, mm, c] → [B, mm, n, c]
        # 将窗口维度(n)和像素维度(mm)交换，实现不同窗口间同一位置像素的交互
        # 3. 多头分割
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q2, k2, v2))
        q2 *= self.scale # 4-7. 同样的注意力计算
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        # sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = out2.permute(0, 2, 1, 3) # 8. 还原Shuffle

        out_1 = torch.cat([out1, out2], dim=-1).contiguous() # 拼接两个分支: [B, n, 16, 16] + [B, n, 16, 16] → [B, n, 16, 32]
        # 还原空间布局: [B, n, 16, 32] → [B, H, W, 32]
        out_1 = rearrange(out_1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size1[0], w=w // w_size1[1],
                          b0=w_size1[0])# 输出: [B, H, W, 32] (16+16=32维，对应c//2)

        # local branch of window size 2
        # 与分支1完全相同的逻辑，只是窗口尺寸改为window_size2 (10×10)
        q3, k3, v3 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),
                         (q3, k3, v3))# [B, (H/10)×(W/10), 100, 16]
        q3, k3, v3 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q3, k3, v3))
        q3 *= self.scale
        sim3 = einsum('b n h i d, b n h j d -> b n h i j', q3, k3)
        # sim3 = sim3 + self.pos_emb3
        attn3 = sim3.softmax(dim=-1)
        out3 = einsum('b n h i j, b n h j d -> b n h i d', attn3, v3)
        out3 = rearrange(out3, 'b n h mm d -> b n mm (h d)')

        # non-local of window size 2
        # 与分支2完全相同的逻辑，窗口尺寸为window_size2 (10×10)
        q4, k4, v4 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),
                         (q4, k4, v4))
        q4, k4, v4 = map(lambda t: t.permute(0, 2, 1, 3), (q4.clone(), k4.clone(), v4.clone()))
        q4, k4, v4 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q4, k4, v4))
        q4 *= self.scale
        sim4 = einsum('b n h i d, b n h j d -> b n h i j', q4, k4)
        # sim4 = sim4 + self.pos_emb4
        attn4 = sim4.softmax(dim=-1)
        out4 = einsum('b n h i j, b n h j d -> b n h i d', attn4, v4)
        out4 = rearrange(out4, 'b n h mm d -> b n mm (h d)')
        out4 = out4.permute(0, 2, 1, 3)# 还原Shuffle
        # 拼接: [B, n, 100, 16] + [B, n, 100, 16] → [B, n, 100, 32]
        out_2 = torch.cat([out3, out4], dim=-1).contiguous()
        # 还原空间布局: [B, n, 100, 32] → [B, H, W, 32]
        out_2 = rearrange(out_2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size2[0], w=w // w_size2[1],
                          b0=w_size2[0]) # 输出: [B, H, W, 32]
        # 拼接两个窗口尺度的结果: [B, H, W, 32] + [B, H, W, 32] → [B, H, W, 64]
        out = torch.cat([out_1, out_2], dim=-1).contiguous()
        # 输出投影: 64 → dim (原始通道数C)
        out = self.to_out(out)
        # 还原为图像格式: [B, H, W, C] → [B, C, H, W]
        return out.permute(0, 3, 1, 2)