# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision.models.convnext import convnext_small, ConvNeXt_Small_Weights
# import math
# from torchvision.ops import misc as misc_ops  # LayerNorm2d 在這裡
# ---------- Encoder + FPN ----------
# class EncoderFPN(nn.Module):
#     """
#     輸入:  [B, in_ch, H, W]
#     輸出:  P4, P8, P16  分別是 [B, out_c, H/4, W/4], [B, out_c, H/8, W/8], [B, out_c, H/16, W/16]
#
#     backbone 選項:
#       - 'resnet18'         (預設)
#       - 'convnext_small'   (新增)
#     """
#     def __init__(self, in_ch=1, out_c=128, backbone: str = "convnext_small", pretrained: bool = True):
#         super().__init__()
#         self.backbone_name = backbone.lower()
#
#         if self.backbone_name == "resnet18":
#             weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#             m = resnet18(weights=weights)
#             # 改第一層以支援非 RGB 輸入
#             if in_ch != 3:
#                 m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#             # stages (stride): stem(/4), layer1(/4), layer2(/8), layer3(/16)
#             self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # /4
#             self.layer1 = m.layer1   # /4,  C=64
#             self.layer2 = m.layer2   # /8,  C=128
#             self.layer3 = m.layer3   # /16, C=256
#
#             C4, C8, C16 = 64, 128, 256
#
#         elif self.backbone_name == "convnext_small":
#             weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
#             m = convnext_small(weights=weights)
#
#             # torchvision ConvNeXt 結構：m.features = [stem, stage1, stage2, stage3, stage4]
#             # strides: stem(/4) -> s1(/4) -> s2(/8) -> s3(/16) -> s4(/32)
#             # channels (small): stem out=96, s1=96, s2=192, s3=384, s4=768
#             # 若 in_ch != 3，需要同時替換 stem 的 LayerNorm2d(3) 與 Conv2d(3,96,4,4)
#             if in_ch != 3:
#                 # conv
#                 old_conv = m.features[0][0]
#                 m.features[0][0] = nn.Conv2d(
#                     in_ch, old_conv.out_channels,
#                     kernel_size=old_conv.kernel_size,
#                     stride=old_conv.stride,
#                     padding=old_conv.padding,
#                     bias=old_conv.bias is not None
#                 )
#                 # norm（Conv2dNormActivation 預設用 LayerNorm2d）
#                 from torchvision.ops import misc as misc_ops
#                 m.features[0][1] = misc_ops.LayerNorm2d(in_ch, eps=1e-6)
#
#                 # （可選）把 RGB 預訓練權重平均到單通道
#                 with torch.no_grad():
#                     if hasattr(old_conv, "weight") and old_conv.weight.shape[1] == 3:
#                         new_w = old_conv.weight.data.mean(dim=1, keepdim=True)  # [96,1,4,4]
#                         m.features[0][0].weight.copy_(new_w)
#
#             # 取到 /4, /8, /16 的特徵
#             self.convnext = m
#             # /4 輸出位置：經 stem 與 stage1
#             # /8：stage2；/16：stage3
#             C4, C8, C16 = 96, 192, 384
#
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone}")
#
#         # FPN lateral + smooth（統一輸出通道 out_c）
#         self.lat4   = nn.Conv2d(C4,  out_c, 1)
#         self.lat8   = nn.Conv2d(C8,  out_c, 1)
#         self.lat16  = nn.Conv2d(C16, out_c, 1)
#
#         self.smooth4  = nn.Conv2d(out_c, out_c, 3, padding=1)
#         self.smooth8  = nn.Conv2d(out_c, out_c, 3, padding=1)
#         self.smooth16 = nn.Conv2d(out_c, out_c, 3, padding=1)
#
#     def forward(self, x):
#         if self.backbone_name == "resnet18":
#             x  = self.stem(x)      # /4
#             c4 = self.layer1(x)    # /4,  C=64
#             c8 = self.layer2(c4)   # /8,  C=128
#             c16= self.layer3(c8)   # /16, C=256
#
#         elif self.backbone_name == "convnext_small":
#             # ConvNeXt features flow:
#             # stem -> stage1 -> stage2 -> stage3 -> stage4
#             f = self.convnext.features
#             x = f[0](x)        # stem, /4
#             c4 = f[1](x)       # stage1, /4, C=96
#             s2 = f[2](c4)
#             c8 = f[3](s2)      # stage2, /8, C=192
#             s3 = f[4](c8)
#             c16= f[5](s3)      # stage3, /16, C=384
#             # （stage4 是 /32，本模組不需要）
#             assert c4.shape[1] == 96 and c8.shape[1] == 192 and c16.shape[1] == 384, \
#                 f"unexpected channels: c4={c4.shape}, c8={c8.shape}, c16={c16.shape}"
#         # 形狀護欄
#         assert c4.shape[1] in (64, 96), f"c4 channels={c4.shape[1]} unexpected"
#         assert c8.shape[1] in (128, 192), f"c8 channels={c8.shape[1]} unexpected"
#         assert c16.shape[1] in (256, 384), f"c16 channels={c16.shape[1]} unexpected"
#         # top-down FPN
#         l16 = self.lat16(c16)
#         l8  = self.lat8(c8)  + F.interpolate(l16, size=c8.shape[-2:], mode='nearest')
#         l4  = self.lat4(c4)  + F.interpolate(l8,  size=c4.shape[-2:], mode='nearest')
#
#         P16 = self.smooth16(l16)
#         P8  = self.smooth8(l8)
#         P4  = self.smooth4(l4)
#         return P4, P8, P16
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import (
    convnext_small, convnext_base,
    ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
)
from torchvision.ops import misc as misc_ops  # LayerNorm2d 在這裡
import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

class EncoderFPN(nn.Module):
    """
    輸入:  [B, in_ch, H, W]
    輸出:  P4, P8, P16  分別是 [B, out_c, H/4, W/4], [B, out_c, H/8, W/8], [B, out_c, H/16, W/16]

    backbone 選項:
      - 'resnet18'         (預設)
      - 'convnext_small'
      - 'convnext_base'
    """
    def __init__(self, in_ch=1, out_c=128, backbone: str = "convnext_base", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        # ------------------------------
        # 1) Backbone
        # ------------------------------
        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = resnet18(weights=weights)

            # 改第一層以支援非 RGB 輸入
            if in_ch != 3:
                m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # stages (stride): stem(/4), layer1(/4), layer2(/8), layer3(/16)
            self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # /4
            self.layer1 = m.layer1   # /4,  C=64
            self.layer2 = m.layer2   # /8,  C=128
            self.layer3 = m.layer3   # /16, C=256

            C4, C8, C16 = 64, 128, 256

        elif self.backbone_name in ["convnext_small", "convnext_base"]:
            is_base = (self.backbone_name == "convnext_base")

            weights = None
            if pretrained:
                weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if is_base else ConvNeXt_Small_Weights.IMAGENET1K_V1

            m = convnext_base(weights=weights) if is_base else convnext_small(weights=weights)

            # torchvision ConvNeXt：m.features = [stem, stage1, stage2, stage3, stage4] (實際上是可 index 的 Sequential)
            # strides: stem(/4) -> s1(/4) -> s2(/8) -> s3(/16) -> s4(/32)
            # channels:
            #   small: stem out=96,  s1=96,  s2=192, s3=384, s4=768
            #   base : stem out=128, s1=128, s2=256, s3=512, s4=1024
            stem_out = 128 if is_base else 96

            # 若 in_ch != 3，需要替換 stem 的 Conv2d 與 LayerNorm2d（注意 LN 的 channel = stem_out）
            if in_ch != 3:
                old_conv = m.features[0][0]  # Conv2d(3, stem_out, 4,4,stride=4)

                m.features[0][0] = nn.Conv2d(
                    in_ch, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                # ✅ LN 應該對 stem_out 做 norm，不是 in_ch
                m.features[0][1] = misc_ops.LayerNorm2d(stem_out, eps=1e-6)

                # （可選）把 RGB 預訓練權重平均到單通道（只在 in_ch=1 時做）
                with torch.no_grad():
                    if hasattr(old_conv, "weight") and old_conv.weight.shape[1] == 3 and in_ch == 1:
                        new_w = old_conv.weight.data.mean(dim=1, keepdim=True)  # [stem_out,1,4,4]
                        m.features[0][0].weight.copy_(new_w)

            self.convnext = m
            C4, C8, C16 = (128, 256, 512) if is_base else (96, 192, 384)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ------------------------------
        # 2) FPN lateral + smooth（統一輸出通道 out_c）
        # ------------------------------
        self.lat4   = nn.Conv2d(C4,  out_c, 1)
        self.lat8   = nn.Conv2d(C8,  out_c, 1)
        self.lat16  = nn.Conv2d(C16, out_c, 1)

        self.smooth4  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth8  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth16 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x):
        # ------------------------------
        # Backbone forward
        # ------------------------------
        if self.backbone_name == "resnet18":
            x   = self.stem(x)      # /4
            c4  = self.layer1(x)    # /4,  C=64
            c8  = self.layer2(c4)   # /8,  C=128
            c16 = self.layer3(c8)   # /16, C=256

        elif self.backbone_name in ["convnext_small", "convnext_base"]:
            is_base = (self.backbone_name == "convnext_base")
            ch4, ch8, ch16 = (128, 256, 512) if is_base else (96, 192, 384)

            # ConvNeXt features flow: stem -> stage1 -> stage2 -> stage3 -> stage4
            f = self.convnext.features

            # 注意：torchvision 的 convnext.features 通常可以用 0..5 index（stem + 4 stages）
            x   = f[0](x)       # stem,  /4
            c4  = f[1](x)       # stage1 /4
            s2  = f[2](c4)      # downsample(/8 前置 block/transition)
            c8  = f[3](s2)      # stage2 /8
            s3  = f[4](c8)      # downsample(/16 前置)
            c16 = f[5](s3)      # stage3 /16
            # stage4(/32) 不用

            assert c4.shape[1] == ch4 and c8.shape[1] == ch8 and c16.shape[1] == ch16, \
                f"unexpected channels: c4={c4.shape}, c8={c8.shape}, c16={c16.shape}"

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # ------------------------------
        # Shape guard (保險，不影響訓練)
        # ------------------------------
        assert c4.shape[1] in (64, 96, 128), f"c4 channels={c4.shape[1]} unexpected"
        assert c8.shape[1] in (128, 192, 256), f"c8 channels={c8.shape[1]} unexpected"
        assert c16.shape[1] in (256, 384, 512), f"c16 channels={c16.shape[1]} unexpected"

        # ------------------------------
        # Top-down FPN
        # ------------------------------
        l16 = self.lat16(c16)
        l8  = self.lat8(c8) + F.interpolate(l16, size=c8.shape[-2:], mode="nearest")
        l4  = self.lat4(c4) + F.interpolate(l8,  size=c4.shape[-2:], mode="nearest")

        P16 = self.smooth16(l16)
        P8  = self.smooth8(l8)
        P4  = self.smooth4(l4)
        return P4, P8, P16

# ---------- ROI-free 點特徵取樣 ----------


def sample_point_feats(P, p_norm, patch=1):
    """
    Fast version.
    P: [B,C,H,W]
    p_norm: [B,N,2] in [-1,1] (x,y)
    patch: odd int. 1 means center only; >1 returns C*patch*patch by concatenation.
    """
    B, C, H, W = P.shape
    B2, N, _ = p_norm.shape
    assert B == B2, "Batch size mismatch"

    # center only
    if patch <= 1:
        grid = p_norm.view(B, N, 1, 2)
        feat = F.grid_sample(
            P, grid, mode='bilinear',
            align_corners=True, padding_mode='zeros'
        )  # [B,C,N,1]
        return feat.squeeze(-1).transpose(1, 2)  # [B,N,C]

    assert patch % 2 == 1, "patch should be odd (e.g., 3/5/7)."
    r = patch // 2

    dx = 2.0 / max(W - 1, 1)
    dy = 2.0 / max(H - 1, 1)

    offs = []
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            offs.append((i * dx, j * dy))
    offs = torch.tensor(offs, device=p_norm.device, dtype=p_norm.dtype)  # [K,2]
    K = offs.shape[0]  # patch*patch

    # p_norm: [B,N,2] -> [B,N,1,2]
    base = p_norm.unsqueeze(2)  # [B,N,1,2]
    grid = base + offs.view(1, 1, K, 2)  # [B,N,K,2] (may go out of [-1,1], zeros padding handles it)
    # reshape to grid_sample format: [B, H_out, W_out, 2]
    # we want H_out=N, W_out=K
    grid = grid.view(B, N, K, 2)

    feat = F.grid_sample(
        P, grid, mode='bilinear',
        align_corners=True, padding_mode='zeros'
    )  # [B,C,N,K]

    # rearrange to [B,N,C*K] with the SAME offset ordering
    feat = feat.permute(0, 2, 1, 3).contiguous()  # [B,N,C,K]
    feat = feat.view(B, N, C * K)                 # [B,N,C*patch*patch]
    return feat

class PointConditioner(nn.Module):
    def __init__(self, c_fpn=128, cond_c=64, patch=5, with_gate=False):
        super().__init__()
        self.c4  = nn.Conv2d(c_fpn, cond_c, 1)
        self.c8  = nn.Conv2d(c_fpn, cond_c, 1)
        self.c16 = nn.Conv2d(c_fpn, cond_c, 1)
        self.patch = patch
        if patch > 1:
            self.flat4  = nn.Linear(cond_c * patch * patch, cond_c)
            self.flat8  = nn.Linear(cond_c * patch * patch, cond_c)
            self.flat16 = nn.Linear(cond_c * patch * patch, cond_c)
        self.with_gate = with_gate
        if with_gate:
            self.gate = MSFusionGate(cond_c=cond_c)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(c_fpn, cond_c)
        self.out_dim = cond_c * 3 + cond_c  # local(3C)+global(C)

    @torch.no_grad()
    def precompute(self, P4, P8, P16):
        """
        每個 batch 只做一次的昂貴部分：
          1) 1x1 conv 降維後的 feature map
          2) global context (from P16)
        回傳 cache dict，之後 forward_cached() 可重用。
        """
        q4  = self.c4(P4)     # [B,cond_c,H4,W4]
        q8  = self.c8(P8)     # [B,cond_c,H8,W8]
        q16 = self.c16(P16)   # [B,cond_c,H16,W16]

        g = self.global_pool(P16).flatten(1)     # [B,c_fpn]
        g = self.global_proj(g)                  # [B,cond_c]
        return {"q4": q4, "q8": q8, "q16": q16, "g": g}

    def forward_cached(self, cache, p_norm):
        """
        cache: 由 precompute() 得到
        p_norm: [B,N,2] in [-1,1]
        """
        q4, q8, q16, g = cache["q4"], cache["q8"], cache["q16"], cache["g"]

        f4  = sample_point_feats(q4,  p_norm, patch=self.patch)   # [B,N,cond_c*k*k] or [B,N,cond_c]
        f8  = sample_point_feats(q8,  p_norm, patch=self.patch)
        f16 = sample_point_feats(q16, p_norm, patch=self.patch)

        if self.patch > 1:
            f4  = self.flat4(f4)
            f8  = self.flat8(f8)
            f16 = self.flat16(f16)

        if self.with_gate:
            local_feat = self.gate(f4, f8, f16)   # [B,N,3*cond_c]
        else:
            local_feat = torch.cat([f4, f8, f16], dim=-1)

        N = p_norm.shape[1]
        g_exp = g.unsqueeze(1).expand(-1, N, -1)  # [B,N,cond_c]
        return torch.cat([local_feat, g_exp], dim=-1)  # [B,N, 3C + C] = [B,N,4C]

    def forward(self, P4, P8, P16, p_norm):
        # 舊介面保留：不想改其他地方時仍可用
        cache = self.precompute(P4, P8, P16)
        return self.forward_cached(cache, p_norm)

class MSFusionGate(nn.Module):
    def __init__(self, cond_c=64, hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cond_c * 3, hidden), nn.ReLU(True),
            nn.Linear(hidden, 3)
        )
        # 最後輸出仍維持 cond_c*3，讓你現有 head 不用改
        self.out  = nn.Sequential(
            nn.Linear(cond_c * 4, cond_c * 3), nn.ReLU(True)
        )

    def forward(self, f4, f8, f16):
        cat = torch.cat([f4, f8, f16], dim=-1)   # [B,N,3C]
        w = self.gate(cat).softmax(dim=-1)       # [B,N,3]
        fused = w[..., 0:1]*f4 + w[..., 1:2]*f8 + w[..., 2:3]*f16  # [B,N,C]
        out = self.out(torch.cat([cat, fused], dim=-1))            # [B,N,3C]
        return out

# ---------- timestep embedding ----------
class TimestepEmbed(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):  # t: [B,1] or [B,N] (int/float)
        orig = t.shape
        t = t.float().unsqueeze(-1)          # [B,1,1] or [B,N,1]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device) * (-torch.log(torch.tensor(10000.0, device=device))/half))
        ang = t * freqs                      # broadcast
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [..., dim]
        pe = pe.view(*orig, self.dim)
        return self.proj(pe)                 # [B,1,dim] or [B,N,dim]

#----------------------------------Denoiser---------------------------------

# 最一般的MLP
# class DenoiserHead(nn.Module):
#     def __init__(self, in_dim, hidden=256, with_score=True):
#         super().__init__()
#         out_dim = 2 + (1 if with_score else 0)
#         self.with_score = with_score
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.SiLU(),
#             nn.Linear(hidden, hidden), nn.SiLU(),
#             nn.Linear(hidden, out_dim)
#         )
#
#     def forward(self, x):  # x: [B,N,in_dim]
#         y = self.mlp(x)
#         if self.with_score:
#             return y[..., :2], y[..., 2]  # eps_pred, score
#         else:
#             return y, None

# 最一般的MLP
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim*2)
        self.act  = nn.SiLU()
        self.fc2  = nn.Linear(dim*2, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.dropout(h)
        return x + h

class DenoiserHeadRes(nn.Module):
    def __init__(self, in_dim, hidden=384, depth=3, dropout=0.2, p_prior=0.07):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.eps_head   = nn.Linear(hidden, 2)
        self.exist_head = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.exist_head.bias.fill_(math.log(p_prior/(1-p_prior)))
    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks: h = blk(h)
        h = self.norm(h)
        eps = self.eps_head(h)
        exist_logit = self.exist_head(h).squeeze(-1)
        return eps

#
# class ResidualMLPBlock(nn.Module):
#     def __init__(self, dim, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.fc1   = nn.Linear(dim, dim * 4)
#         self.fc2   = nn.Linear(dim * 4, dim)
#         self.drop  = nn.Dropout(dropout)
#
#     def forward(self, x):
#         h = self.norm1(x)
#         h = F.silu(self.fc1(h))
#         h = self.drop(h)
#         h = self.fc2(h)
#         return x + h
#
# class EpsHeadFiLM(nn.Module):
#     def __init__(self, pf_dim, t_dim, hidden=512, depth=6, dropout=0.1):
#         super().__init__()
#         self.in_proj = nn.Linear(pf_dim + 2, hidden)    # pf + p_t
#         self.t_proj  = nn.Linear(t_dim, hidden * 2)     # 產生 scale/shift
#         self.blocks  = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(depth)])
#         self.norm    = nn.LayerNorm(hidden)
#         self.out     = nn.Linear(hidden, 2)
#
#     def forward(self, pf, te, p_t):
#         h = self.in_proj(torch.cat([pf, p_t], dim=-1))
#         ss = self.t_proj(te)                 # [B,N,2H]
#         scale, shift = ss.chunk(2, dim=-1)
#         h = h * (1 + scale) + shift          # FiLM 調制
#
#         for blk in self.blocks:
#             h = blk(h)
#         h = self.norm(h)
#         return self.out(h)

#----------------------------------ConfidenceHead---------------------------------

#沒加layer norm的
# class ConfidenceHead(nn.Module):
#     def __init__(self, in_dim, hidden=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden), nn.SiLU(),
#             nn.Linear(hidden, hidden), nn.SiLU(),
#             nn.Linear(hidden, 1)
#         )
#     def forward(self, f):  # f: [B,N,in_dim]
#         return self.mlp(f).squeeze(-1)  # [B,N]

#加layer norm的
class ConfidenceHead(nn.Module):
    def __init__(self, in_dim, hidden=256, p_prior=0.07):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

        # prior-bias init: sigmoid(bias) = p_prior
        prior_logit = math.log(p_prior / (1.0 - p_prior))
        with torch.no_grad():
            self.mlp[-1].bias.fill_(prior_logit)

    def forward(self, f):  # f: [B,N,in_dim]
        f = self.norm(f)
        return self.mlp(f).squeeze(-1)  # [B,N]


class ModelBuilder(nn.Module):
    """
    介面：
      encode(images) -> (P4,P8,P16)
      denoise(feats, p_t, t) -> eps_pred, score
    """
    def __init__(self, in_ch=1, fpn_c=128, cond_c=64, t_dim=256, with_score=True):
        super().__init__()
        self.backbone = EncoderFPN(in_ch=in_ch, out_c=fpn_c)
        self.temb = TimestepEmbed(dim=t_dim)
        self.cond = PointConditioner(c_fpn=fpn_c, cond_c=cond_c, patch=5, with_gate=True)
        #self.head_eps = EpsHeadFiLM(pf_dim=cond_c * 4, t_dim=t_dim, hidden=512, depth=6, dropout=0.1)
        self.head_eps = DenoiserHeadRes(in_dim=cond_c * 4 + t_dim + 2, hidden=384, depth=3, dropout=0.2)
        self.conf_head = ConfidenceHead(in_dim=cond_c*4, hidden=256)

    def encode(self, images):
        # images: [B,in_ch,H,W]
        return self.backbone(images)

    def denoise(self, feats, p_t, t, abar_t=None, clamp_eps=1e-6, cond_cache=None, need_exist=True):
        P4, P8, P16 = feats

        # pf at p_t
        if cond_cache is None:
            pf = self.cond(P4, P8, P16, p_t)
        else:
            pf = self.cond.forward_cached(cond_cache, p_t)

        te = self.temb(t)
        if te.dim() == 3 and te.size(1) == 1:
            te = te.expand(pf.size(0), pf.size(1), te.size(-1))

        x = torch.cat([pf, te, p_t], dim=-1)
        # eps_pred = self.head_eps(pf, te, p_t)
        eps_pred = self.head_eps(x)

        # x0_hat
        if abar_t is None:
            x0_hat = p_t
        else:
            abar = abar_t
            if abar.dim() == 3 and abar.size(1) == 1:
                abar = abar.expand(pf.size(0), pf.size(1), 1)
            sqrt_abar = (abar + clamp_eps).sqrt()
            sqrt_onem = (1.0 - abar).clamp_min(0).sqrt()
            x0_hat = (p_t - sqrt_onem * eps_pred) / (sqrt_abar + 1e-12)
            x0_hat = x0_hat.clamp(-1.0 + 1e-3, 1.0 - 1e-3)

        # exist head optional
        exist_logit = None
        if need_exist:
            if cond_cache is None:
                pf_hat = self.cond(P4, P8, P16, x0_hat.detach())
            else:
                pf_hat = self.cond.forward_cached(cond_cache, x0_hat.detach())
            exist_logit = self.conf_head(pf_hat)
        # if need_exist:
        #     print("[DEBUG] need_exist=True")
        #     print("[DEBUG] exist_logit type:", type(exist_logit))
        #     assert exist_logit is not None, "BUG: need_exist=True but exist_logit is None"
        return eps_pred, exist_logit



