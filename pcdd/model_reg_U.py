import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group
from model_new import Local_op, Point_Transformer_Last, CrossAttentionLayer, Node_Point_Transformer_Last


class NodeRegressionHead(nn.Module):
    def __init__(self, node_dim: int, global_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(node_dim + global_dim, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 3, 1),
        )

    def forward(
        self,
        bottleneck_feat: torch.Tensor,
        skip_ca1: torch.Tensor,
        skip_sa: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, num_nodes = bottleneck_feat.shape
        global_expanded = global_feat.unsqueeze(-1).expand(bsz, global_feat.size(1), num_nodes)

        del skip_ca1, skip_sa
        disp = self.mlp(torch.cat([bottleneck_feat, global_expanded], dim=1))
        return disp.permute(0, 2, 1).contiguous()


class PctRegressor(nn.Module):
    def __init__(self, args, disp_mean=(0.0, 0.0, 0.0), disp_std=(1.0, 1.0, 1.0), num_nodes: int = 1118):
        super().__init__()
        self.args = args
        self.num_nodes = int(num_nodes)
        self.num_point_tokens = int(getattr(args, "num_point_tokens", 512))

        dm = torch.as_tensor(disp_mean, dtype=torch.float32).view(-1)
        ds = torch.as_tensor(disp_std, dtype=torch.float32).view(-1)
        if dm.numel() == 1:
            dm = dm.repeat(3)
        if ds.numel() == 1:
            ds = ds.repeat(3)
        if dm.numel() != 3 or ds.numel() != 3:
            raise ValueError(f"disp_mean/std must be scalar or length-3, got dm={dm.shape}, ds={ds.shape}")
        self.register_buffer("disp_mean", dm.view(1, 1, 3))
        self.register_buffer("disp_std", ds.view(1, 1, 3))

        self.cloud_conv1 = nn.Conv1d(4, 64, kernel_size=1, bias=False)
        self.cloud_bn1 = nn.BatchNorm1d(64)
        self.cloud_conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.cloud_bn2 = nn.BatchNorm1d(128)

        self.node_conv1 = nn.Conv1d(4, 64, kernel_size=1, bias=False)
        self.node_bn1 = nn.BatchNorm1d(64)
        self.node_conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.node_bn2 = nn.BatchNorm1d(128)

        self.gather_local_0 = Local_op(256, 256)
        self.gather_local_1 = Local_op(512, 512)

        self.pt_last = Point_Transformer_Last(args, in_channels=512)
        self.npt_last = Node_Point_Transformer_Last(args, in_channels=128)

        self.fuse_conv = nn.Sequential(
            nn.Conv1d(1536, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.node_fuse_conv = nn.Sequential(
            nn.Conv1d(1024, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.point_from_node_attn = CrossAttentionLayer(query_dim=512, key_dim=128, use_rpe=False)
        self.node_from_point_attn = nn.ModuleList(
            [
                CrossAttentionLayer(query_dim=256, key_dim=1024, use_rpe=True),
                CrossAttentionLayer(query_dim=256, key_dim=1024, use_rpe=True),
            ]
        )

        self.global_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

        self.reg_head = NodeRegressionHead(
            node_dim=256,
            global_dim=256,
            dropout=getattr(args, "dropout", 0.1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(
        self,
        x: torch.Tensor,
        node_xyz: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        x:        [B,4,N]  (x,y,z,min_dist)
        node_xyz: [B,4,M]  (x,y,z,min_dist)
        return:   [B,M,3]
        """
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        node_xyz_pos = node_xyz[:, :3, :].contiguous()

        x = F.relu(self.cloud_bn1(self.cloud_conv1(x)))
        x = F.relu(self.cloud_bn2(self.cloud_conv2(x)))

        node_feat = F.relu(self.node_bn1(self.node_conv1(node_xyz)))
        node_feat = F.relu(self.node_bn2(self.node_conv2(node_feat)))

        feat = x.permute(0, 2, 1).contiguous()

        new_xyz, new_feat = sample_and_group(
            npoint=self.num_nodes * 4,
            radius=0.15,
            nsample=16,
            xyz=xyz,
            points=feat,
        )
        feat_0 = self.gather_local_0(new_feat)

        feat_0_perm = feat_0.permute(0, 2, 1).contiguous()
        new_xyz, new_feat = sample_and_group(
            npoint=self.num_point_tokens,
            radius=0.2,
            nsample=32,
            xyz=new_xyz,
            points=feat_0_perm,
        )
        feat_1 = self.gather_local_1(new_feat)

        cross_feat_1 = self.point_from_node_attn(
            feat_1,
            node_feat,
            new_xyz.permute(0, 2, 1),
            node_xyz_pos,
        )

        trans_feat = self.pt_last(cross_feat_1, new_xyz)
        fused_feat = torch.cat([trans_feat, feat_1], dim=1)
        fused_feat = self.fuse_conv(fused_feat)

        node_feat = self.npt_last(node_feat, node_xyz_pos)
        skip_sa = self.node_fuse_conv(node_feat)

        pxyz = new_xyz.permute(0, 2, 1).contiguous()
        if return_attention:
            skip_ca1, attn_stage_1 = self.node_from_point_attn[0](
                skip_sa,
                fused_feat,
                node_xyz_pos,
                pxyz,
                return_attention=True,
            )
            bottleneck, attn_stage_2 = self.node_from_point_attn[1](
                skip_ca1,
                fused_feat,
                node_xyz_pos,
                pxyz,
                return_attention=True,
            )
        else:
            skip_ca1 = self.node_from_point_attn[0](
                skip_sa,
                fused_feat,
                node_xyz_pos,
                pxyz,
            )
            bottleneck = self.node_from_point_attn[1](
                skip_ca1,
                fused_feat,
                node_xyz_pos,
                pxyz,
            )

        g_max = F.adaptive_max_pool1d(fused_feat, 1).squeeze(-1)
        g_avg = F.adaptive_avg_pool1d(fused_feat, 1).squeeze(-1)
        global_feat = self.global_proj(torch.cat([g_max, g_avg], dim=1))

        disp = self.reg_head(bottleneck, skip_ca1, skip_sa, global_feat)
        disp = disp * self.disp_std + self.disp_mean
        if return_attention:
            return disp, {
                "node_from_point_stage_1": attn_stage_1,
                "node_from_point_stage_2": attn_stage_2,
                "point_token_xyz": pxyz,
            }
        return disp
