import torch
import torch.nn as nn
import torch.nn.functional as F


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).view(b, n, -1).permute(0, 2, 1)
        return x


class Point_Transformer_Last(nn.Module):
    """pos_xyz 仅接受 3 通道坐标 (x,y,z)"""

    def __init__(self, args, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, 1, bias=False)
        self.pos_xyz = nn.Conv1d(3, 256, 1)

        # 分离 BN：避免 feat 与 pos 共享统计
        self.bn_feat = nn.BatchNorm1d(256)
        self.bn_pos = nn.BatchNorm1d(256)

        self.sa_layers = nn.ModuleList([SA_Layer(256) for _ in range(4)])

    def forward(self, feat, xyz):
        """
        feat: [B,C,N]
        xyz:  [B,N,3]
        """
        xyz = xyz.permute(0, 2, 1).contiguous()  # [B,3,N]
        pos_feat = self.bn_pos(self.pos_xyz(xyz))
        feat = F.relu(self.bn_feat(self.conv1(feat))) + pos_feat

        outputs = []
        x = feat
        for sa in self.sa_layers:
            x = sa(x, pos_feat)
            outputs.append(x)
        return torch.cat(outputs, dim=1)  # [B,1024,N]


class Node_Point_Transformer_Last(nn.Module):
    """pos_xyz 仅接受 3 通道坐标 (x,y,z)"""

    def __init__(self, args, in_channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, 1, bias=False)
        self.pos_xyz = nn.Conv1d(3, 256, 1)

        self.bn_feat = nn.BatchNorm1d(256)
        self.bn_pos = nn.BatchNorm1d(256)

        self.sa_layers = nn.ModuleList([SA_Layer(256) for _ in range(4)])

    def forward(self, feat, xyz):
        """
        feat: [B,C,M]
        xyz:  [B,3,M]
        """
        pos_feat = self.bn_pos(self.pos_xyz(xyz))
        feat = F.relu(self.bn_feat(self.conv1(feat))) + pos_feat

        outputs = []
        x = feat
        for sa in self.sa_layers:
            x = sa(x, pos_feat)
            outputs.append(x)
        return torch.cat(outputs, dim=1)  # [B,1024,M]


class SA_Layer(nn.Module):
    """修正注意力加权方向为 A @ V"""

    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x, xyz_feat):
        identity = x
        x = x + xyz_feat

        q = self.q_conv(x).permute(0, 2, 1)  # [B,N,C/4]
        k = self.k_conv(x)                   # [B,C/4,N]
        v = self.v_conv(x)                   # [B,C,N]

        d_k = q.size(-1)
        energy = torch.bmm(q, k) / (d_k ** 0.5)  # [B,N,N]
        attention = F.softmax(energy, dim=-1)

        x_r = torch.bmm(attention, v.permute(0, 2, 1)).permute(0, 2, 1)  # [B,C,N]
        x_r = self.act(self.after_norm(self.trans_conv(identity - x_r)))
        return identity + x_r


class CrossAttentionLayer(nn.Module):
    """
    query=point_feat, key/value=node_feat
    加入可选的相对位置 bias：bias(Δx,Δy,Δz,||Δ||^2) -> [B,heads,N,M]
    """

    def __init__(self, query_dim, key_dim, num_heads=4, use_rpe=True, rpe_hidden=32):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.use_rpe = bool(use_rpe)

        self.point_pos_conv = nn.Conv1d(3, query_dim, 1)
        self.node_pos_conv = nn.Conv1d(3, key_dim, 1)

        self.q_conv = nn.Conv1d(query_dim, query_dim, 1)
        self.k_conv = nn.Conv1d(key_dim, query_dim, 1)
        self.v_conv = nn.Conv1d(key_dim, query_dim, 1)

        # 相对位置 bias 网络：输入 4 通道 (dx,dy,dz,dist2)
        if self.use_rpe:
            self.rpe = nn.Sequential(
                nn.Conv2d(4, rpe_hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(rpe_hidden, num_heads, 1, bias=True),
            )

        self.out_conv = nn.Conv1d(query_dim, query_dim, 1)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, point_feat, node_feat, point_xyz, node_xyz, return_attention=False):
        """
        point_feat: [B,query_dim,N]
        node_feat:  [B,key_dim,M]
        point_xyz:  [B,3,N]
        node_xyz:   [B,3,M]
        """
        B, Cq, N = point_feat.shape

        point_pos = self.point_pos_conv(point_xyz)
        q_feat = point_feat + point_pos

        node_pos = self.node_pos_conv(node_xyz)
        k_feat = node_feat + node_pos
        v_feat = k_feat

        Q = self.q_conv(q_feat).view(B, self.num_heads, self.head_dim, N)
        K = self.k_conv(k_feat).view(B, self.num_heads, self.head_dim, -1)
        V = self.v_conv(v_feat).view(B, self.num_heads, self.head_dim, -1)

        attn = torch.einsum('bhcn,bhcm->bhnm', Q, K) / (self.head_dim ** 0.5)  # [B,h,N,M]

        # 相对位置 bias（加在 softmax 前）
        if self.use_rpe:
            # delta: [B,3,N,M]
            delta = point_xyz.unsqueeze(-1) - node_xyz.unsqueeze(-2)
            dist2 = (delta * delta).sum(dim=1, keepdim=True)  # [B,1,N,M]
            rpe_in = torch.cat([delta, dist2], dim=1)          # [B,4,N,M]
            bias = self.rpe(rpe_in)                            # [B,h,N,M]
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, V).contiguous().view(B, Cq, N)
        out = self.out_conv(out)
        out = self.norm((out + point_feat).permute(0, 2, 1)).permute(0, 2, 1)
        if return_attention:
            return out, attn
        return out
