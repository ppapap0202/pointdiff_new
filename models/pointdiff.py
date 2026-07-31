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
import torch.nn as nn
import time
from .proposal_prior import ProposalPriorHead



def merge_slots_same_pixel(
    x0_hat: torch.Tensor,   # [B,N,2] in [-1,1]
    pro: torch.Tensor,      # [B,N,C]
    H: int,
    W: int,
    max_slots: int = None,
    weight: torch.Tensor = None,
    weight_min: float = 0.0,
):
    device = x0_hat.device
    B, N, _ = x0_hat.shape
    C = pro.size(-1)
    if max_slots is None:
        max_slots = N

    def m11_to_pixels_batch(p_m11, H, W):
        x = (p_m11[..., 0] + 1.0) * 0.5 * (W - 1)
        y = (p_m11[..., 1] + 1.0) * 0.5 * (H - 1)
        return torch.stack([x, y], dim=-1)

    def pixels_to_m11_batch(p_pix, H, W):
        x = p_pix[..., 0] / max(W - 1, 1) * 2.0 - 1.0
        y = p_pix[..., 1] / max(H - 1, 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    x0_pix = m11_to_pixels_batch(x0_hat.detach(), H, W)  # [B,N,2]

    merged_xy_list = []
    merged_feat_list = []
    merged_mask_list = []

    for b in range(B):
        pts = x0_pix[b]          # [N,2]
        feat = pro[b]            # [N,C]

        # round 到整數 pixel
        pts_int = torch.round(pts).long()
        pts_int[:, 0].clamp_(0, W - 1)
        pts_int[:, 1].clamp_(0, H - 1)

        # 用 unique 做分群
        uniq_xy, inverse = torch.unique(pts_int, dim=0, return_inverse=True)  # uniq_xy:[M,2], inverse:[N]
        M = uniq_xy.size(0)

        # 群內平均原始浮點位置，得到 cluster center
        merged_xy_pix = torch.zeros((M, 2), device=device, dtype=pts.dtype)
        counts = torch.zeros((M,), device=device, dtype=pts.dtype)

        if weight is not None:
            w = weight[b].to(device=device, dtype=pts.dtype).clamp_min(float(weight_min))
            merged_xy_pix.index_add_(0, inverse, pts * w[:, None])
            counts.index_add_(0, inverse, w)
        else:
            merged_xy_pix.index_add_(0, inverse, pts)
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=pts.dtype))
        merged_xy_pix = merged_xy_pix / counts[:, None].clamp_min(1e-6)

        # 代表 feature：選群內最接近中心的點
        merged_feat = torch.zeros((M, C), device=device, dtype=feat.dtype)
        if weight is not None:
            w_feat = weight[b].to(device=device, dtype=feat.dtype).clamp_min(float(weight_min))
            merged_feat.index_add_(0, inverse, feat * w_feat[:, None])
        else:
            merged_feat.index_add_(0, inverse, feat)
        merged_feat = merged_feat / counts[:, None].to(dtype=feat.dtype).clamp_min(1e-6)

        merged_xy = pixels_to_m11_batch(merged_xy_pix, H, W)  # [M,2]

        pad_M = max_slots - M
        if pad_M > 0:
            merged_xy = torch.cat(
                [merged_xy, torch.zeros((pad_M, 2), device=device, dtype=x0_hat.dtype)],
                dim=0
            )
            merged_feat = torch.cat(
                [merged_feat, torch.zeros((pad_M, C), device=device, dtype=pro.dtype)],
                dim=0
            )
            merged_mask = torch.cat(
                [
                    torch.ones(M, device=device, dtype=torch.bool),
                    torch.zeros(pad_M, device=device, dtype=torch.bool),
                ],
                dim=0
            )
        else:
            merged_xy = merged_xy[:max_slots]
            merged_feat = merged_feat[:max_slots]
            merged_mask = torch.ones(max_slots, device=device, dtype=torch.bool)

        merged_xy_list.append(merged_xy)
        merged_feat_list.append(merged_feat)
        merged_mask_list.append(merged_mask)

    merged_xy = torch.stack(merged_xy_list, dim=0)      # [B,N,2]
    merged_feat = torch.stack(merged_feat_list, dim=0)  # [B,N,C]
    merged_mask = torch.stack(merged_mask_list, dim=0)  # [B,N]
    return merged_xy, merged_feat, merged_mask


def merge_slots_same_pixel(
    x0_hat: torch.Tensor,
    pro: torch.Tensor,
    H: int,
    W: int,
    max_slots: int = None,
    weight: torch.Tensor = None,
    weight_min: float = 0.0,
    merge_radius_px: float = 0.0,
):
    """Merge duplicate selector slots by same rounded pixel or by a small radius."""
    device = x0_hat.device
    B, N, _ = x0_hat.shape
    C = pro.size(-1)
    max_slots = int(max_slots or N)

    def m11_to_pixels_batch(p_m11, h, w):
        x = (p_m11[..., 0] + 1.0) * 0.5 * (w - 1)
        y = (p_m11[..., 1] + 1.0) * 0.5 * (h - 1)
        return torch.stack([x, y], dim=-1)

    def pixels_to_m11_batch(p_pix, h, w):
        x = p_pix[..., 0] / max(w - 1, 1) * 2.0 - 1.0
        y = p_pix[..., 1] / max(h - 1, 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    x0_pix = m11_to_pixels_batch(x0_hat.detach(), H, W)
    merged_xy_list = []
    merged_feat_list = []
    merged_mask_list = []
    radius_px = float(merge_radius_px)

    for b in range(B):
        pts = x0_pix[b]
        feat = pro[b]

        if radius_px > 0.0:
            if weight is not None:
                order_weight = weight[b].to(device=device, dtype=pts.dtype).clamp_min(float(weight_min))
            else:
                order_weight = torch.ones((N,), device=device, dtype=pts.dtype)
            order = torch.argsort(order_weight, descending=True)
            assigned = torch.zeros((N,), device=device, dtype=torch.bool)
            inverse = torch.empty((N,), device=device, dtype=torch.long)
            cluster_count = 0
            for idx in order:
                i = int(idx.item())
                if bool(assigned[i]):
                    continue
                dist = torch.norm(pts - pts[i], dim=1)
                members = (~assigned) & (dist < radius_px)
                inverse[members] = int(cluster_count)
                assigned[members] = True
                cluster_count += 1
            M = int(cluster_count)
        else:
            pts_int = torch.round(pts).long()
            pts_int[:, 0].clamp_(0, W - 1)
            pts_int[:, 1].clamp_(0, H - 1)
            _, inverse = torch.unique(pts_int, dim=0, return_inverse=True)
            M = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0

        merged_xy_pix = torch.zeros((M, 2), device=device, dtype=pts.dtype)
        counts = torch.zeros((M,), device=device, dtype=pts.dtype)
        if weight is not None:
            w_xy = weight[b].to(device=device, dtype=pts.dtype).clamp_min(float(weight_min))
            merged_xy_pix.index_add_(0, inverse, pts * w_xy[:, None])
            counts.index_add_(0, inverse, w_xy)
        else:
            merged_xy_pix.index_add_(0, inverse, pts)
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=pts.dtype))
        merged_xy_pix = merged_xy_pix / counts[:, None].clamp_min(1e-6)

        merged_feat = torch.zeros((M, C), device=device, dtype=feat.dtype)
        if weight is not None:
            w_feat = weight[b].to(device=device, dtype=feat.dtype).clamp_min(float(weight_min))
            merged_feat.index_add_(0, inverse, feat * w_feat[:, None])
        else:
            merged_feat.index_add_(0, inverse, feat)
        merged_feat = merged_feat / counts[:, None].to(dtype=feat.dtype).clamp_min(1e-6)

        merged_xy = pixels_to_m11_batch(merged_xy_pix, H, W)
        pad_M = max_slots - M
        if pad_M > 0:
            merged_xy = torch.cat(
                [merged_xy, torch.zeros((pad_M, 2), device=device, dtype=x0_hat.dtype)],
                dim=0,
            )
            merged_feat = torch.cat(
                [merged_feat, torch.zeros((pad_M, C), device=device, dtype=pro.dtype)],
                dim=0,
            )
            merged_mask = torch.cat(
                [
                    torch.ones(M, device=device, dtype=torch.bool),
                    torch.zeros(pad_M, device=device, dtype=torch.bool),
                ],
                dim=0,
            )
        else:
            merged_xy = merged_xy[:max_slots]
            merged_feat = merged_feat[:max_slots]
            merged_mask = torch.ones(max_slots, device=device, dtype=torch.bool)

        merged_xy_list.append(merged_xy)
        merged_feat_list.append(merged_feat)
        merged_mask_list.append(merged_mask)

    return (
        torch.stack(merged_xy_list, dim=0),
        torch.stack(merged_feat_list, dim=0),
        torch.stack(merged_mask_list, dim=0),
    )


def selector_object_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3 and logits.size(-1) == 2:
        return logits.softmax(-1)[..., 1]
    if logits.dim() == 3 and logits.size(-1) == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.sigmoid(logits)


def selector_object_logit(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3 and logits.size(-1) == 2:
        return logits[..., 1] - logits[..., 0]
    if logits.dim() == 3 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits


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
def sample_point_tokens(P, p_norm, patch=1):
    """
    ROI-free token sampling around points (no flatten).
    P:      [B,C,H,W]
    p_norm: [B,N,2] in [-1,1] (x,y)
    patch:  odd int (e.g., 5) -> K=patch*patch tokens

    return:
      tokens: [K, B*N, C]  (K=patch*patch)
    """
    assert patch >= 1 and (patch % 2 == 1), "patch must be odd and >=1"
    B, C, H, W = P.shape
    _, N, _ = p_norm.shape
    if patch == 1:
        # K=1 special case, keep shape [1, B*N, C]
        grid = p_norm.view(B, N, 1, 2)
        feat = F.grid_sample(P, grid, mode='bilinear', align_corners=True, padding_mode='zeros')  # [B,C,N,1]
        feat = feat.permute(3, 0, 2, 1).contiguous()  # [1,B,N,C]
        return feat.view(1, B * N, C)

    r = patch // 2
    dx = 2.0 / max(W - 1, 1)
    dy = 2.0 / max(H - 1, 1)

    offs = []
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            offs.append((i * dx, j * dy))
    offs = torch.tensor(offs, device=p_norm.device, dtype=p_norm.dtype)  # [K,2]
    K = offs.shape[0]
    base = p_norm.unsqueeze(2)                 # [B,N,1,2]
    grid = base + offs.view(1, 1, K, 2)        # [B,N,K,2]
    grid = grid.view(B, N, K, 2)

    feat = F.grid_sample(
        P, grid, mode='bilinear',
        align_corners=True, padding_mode='zeros'
    )  # [B,C,N,K]
    # SAME offset ordering as sample_point_feats, but keep token axis:
    # [B,C,N,K] -> [K,B*N,C]
    feat = feat.permute(0, 2, 1, 3).contiguous()  # [B,N,C,K]
    feat = feat.permute(3, 0, 1, 2).contiguous()  # [K,B,N,C]
    feat = feat.view(K, B * N, C)                 # [K,B*N,C]
    return feat

class PointDynamicConv(nn.Module):
    def __init__(self, hidden_dim=64, dim_dynamic=32, num_dynamic=2, token_count=75):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.token_count = token_count

        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        self.out_layer = nn.Linear(self.hidden_dim * token_count, self.hidden_dim)

    def forward(self, pro_features, roi_tokens):
        K, BN, C = roi_tokens.shape
        assert K == self.token_count, f"Expected K={self.token_count}, got {K}"
        assert C == self.hidden_dim, f"roi_tokens C={C} must match hidden_dim={self.hidden_dim}"

        features = roi_tokens.permute(1, 0, 2).contiguous()  # [BN,K,C]
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2).contiguous()  # [BN,1,2*num_params]
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.activation(self.norm1(features))

        features = torch.bmm(features, param2)
        features = self.activation(self.norm2(features))

        features = features.flatten(1)                # [BN, K*C]
        features = self.out_layer(features)           # [BN, C]
        features = self.activation(self.norm3(features))
        return features


class PointRCNNHead(nn.Module):
    """
    RCNNHead-style refinement for ROI-free point denoising:
      - self-attn across points
      - dynamic interaction between point state and local patch tokens
     - FFN + time FiLM
      - predict eps (dx, dy)
    """
    def __init__(self, d_model=64, nhead=4, dim_ff=256, dim_dynamic=32, t_dim=256, dropout=0.1, token_count=75):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = PointDynamicConv(hidden_dim=d_model, dim_dynamic=dim_dynamic, num_dynamic=2, token_count=token_count)


        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.ff_act = nn.SiLU()

        # time FiLM (per-image), then repeat to points
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, d_model * 2))

        self.eps_head = nn.Linear(d_model, 2)

    def forward(self, local_tokens, pro_features, time_img):
        """
        local_tokens: [S, B*N, d_model]
        pro_features: [B, N, d_model]
        time_img:     [B, t_dim] or [B,1,t_dim]
        return: eps_pred [B,N,2]
        return: pro_next: [B,N,C]
        """
        if time_img.dim() == 3 and time_img.size(1) == 1:
            time_img = time_img.squeeze(1)  # [B,t_dim]
        B, N, C = pro_features.shape
        assert C == self.d_model

        # 1) self-attn across points (N as seq_len, B as batch)
        x = pro_features.permute(1, 0, 2).contiguous()  # [N,B,C]
        x2 = self.self_attn(x, x, x)[0]
        x = self.norm1(x + self.drop1(x2))

        # 2) dynamic interaction with local tokens
        x_bn = x.permute(1, 0, 2).contiguous().view(1, B * N, C)  # [1,BN,C]
        dx = self.inst_interact(x_bn, local_tokens)               # [BN,C]
        x_bn = self.norm2(x_bn + self.drop2(dx.unsqueeze(0)))

        # 3) FFN
        h = self.linear2(self.drop3(self.ff_act(self.linear1(x_bn))))
        x_bn = self.norm3(x_bn + h)                               # [1,BN,C]
        feat = x_bn.squeeze(0)                                    # [BN,C]

        # 4) time FiLM
        ss = self.time_mlp(time_img)                              # [B,2C]
        ss = ss.repeat_interleave(N, dim=0)                       # [BN,2C]
        scale, shift = ss.chunk(2, dim=1)
        feat = feat * (1.0 + scale) + shift

        eps = self.eps_head(feat).view(B, N, 2)
        pro_next = feat.view(B, N, C)
        return eps, pro_next


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

def pool_local_tokens(local_tokens, B, N, use_max=True):
    """
    local_tokens: [S, BN, C]
    return pooled: [B, N, C*(1 or 2)]
    """
    # [S, BN, C] -> [BN, S, C]
    tok = local_tokens.permute(1, 0, 2).contiguous()

    mean_tok = tok.mean(dim=1)  # [BN, C]
    if use_max:
        max_tok = tok.max(dim=1).values  # [BN, C]
        pooled = torch.cat([mean_tok, max_tok], dim=-1)  # [BN, 2C]
    else:
        pooled = mean_tok  # [BN, C]

    return pooled.view(B, N, -1)


class ConfidenceHead(nn.Module):
    """2-class confidence head for DETR/P2PNet-style softmax classification.

    Output:
      logits: [B, N, 2] where class 0 = no-object, class 1 = object
    """
    def __init__(self, in_dim, hidden=256, p_prior=0.07):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )

        # prior-bias init: softmax(logits) gives p(object)=p_prior when logit0=0, logit1=log(p/(1-p))
        prior_logit = math.log(p_prior / (1.0 - p_prior))
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[1].fill_(prior_logit)

    def forward(self, f):  # f: [B,N,in_dim]
        f = self.norm(f)
        return self.mlp(f)  # [B,N,2]


class CandidateInteractionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.attn.out_proj.weight.zero_()
            self.attn.out_proj.bias.zero_()
            self.ff[3].weight.zero_()
            self.ff[3].bias.zero_()

    def forward(
            self,
            x: torch.Tensor,
            valid_mask: torch.Tensor,
            attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        key_padding_mask = ~valid_mask.bool()
        valid_f = valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x * valid_f


class CandidateInteractionSelector(nn.Module):
    """Let selector candidates compete before the final confidence head."""

    def __init__(
            self,
            d_model: int,
            num_layers: int = 1,
            nhead: int = 4,
            dim_ff: int = None,
            dropout: float = 0.1,
            radius: float = 12.0,
            H: int = 256,
            W: int = 256,
    ):
        super().__init__()
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
            raise ValueError(f"selector_interaction_heads={nhead} must divide cond_c={d_model}")
        self.nhead = nhead
        self.radius = float(radius)
        self.H = int(H)
        self.W = int(W)
        dim_ff = int(dim_ff or d_model * 4)
        self.layers = nn.ModuleList([
            CandidateInteractionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=float(dropout),
            )
            for _ in range(max(1, int(num_layers)))
        ])

    def _radius_attn_mask(
            self,
            points_m11: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.radius <= 0.0:
            return None

        points = points_m11.detach().float()
        x = (points[..., 0] + 1.0) * 0.5 * float(self.W - 1)
        y = (points[..., 1] + 1.0) * 0.5 * float(self.H - 1)
        pix = torch.stack([x, y], dim=-1)
        dist = torch.cdist(pix, pix, p=2)

        valid = valid_mask.bool()
        pair_valid = valid[:, :, None] & valid[:, None, :]
        allowed = (dist <= float(self.radius)) & pair_valid
        # Padded queries are ignored after attention; give them legal keys to
        # avoid all-masked attention rows.
        allowed = allowed | ((~valid[:, :, None]) & valid[:, None, :])
        return (~allowed).repeat_interleave(self.nhead, dim=0)

    def forward(
            self,
            feat: torch.Tensor,
            points_m11: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        original_valid = valid_mask.bool()
        safe_valid = original_valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any():
            safe_valid[empty, 0] = True

        attn_mask = self._radius_attn_mask(points_m11, safe_valid)
        x = feat * original_valid.unsqueeze(-1).to(dtype=feat.dtype)
        for layer in self.layers:
            x = layer(x, safe_valid, attn_mask=attn_mask)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x * original_valid.unsqueeze(-1).to(dtype=feat.dtype)


class LocalCompetitionSelector(nn.Module):
    """Differentiable local non-max competition on selector object logits."""

    def __init__(
            self,
            d_model: int,
            radius: float = 6.0,
            temperature: float = 0.7,
            max_strength: float = 0.35,
            init_strength: float = 0.05,
            residual_scale: float = 0.25,
            exclude_self: bool = True,
            H: int = 256,
            W: int = 256,
    ):
        super().__init__()
        d_model = int(d_model)
        self.radius = float(radius)
        self.temperature = max(float(temperature), 1e-4)
        self.max_strength = max(float(max_strength), 0.0)
        self.residual_scale = max(float(residual_scale), 0.0)
        self.exclude_self = bool(exclude_self)
        self.H = int(H)
        self.W = int(W)

        hidden = max(64, d_model * 2)
        self.score_delta = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.score_delta[-1].weight.zero_()
            self.score_delta[-1].bias.zero_()

        if self.max_strength > 0.0:
            init_strength = min(max(float(init_strength), 1e-4), self.max_strength * (1.0 - 1e-4))
            ratio = init_strength / self.max_strength
            self.strength_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        else:
            self.register_parameter("strength_logit", None)

    def _pixel_points(self, points_m11: torch.Tensor) -> torch.Tensor:
        points = points_m11.float()
        x = (points[..., 0] + 1.0) * 0.5 * float(self.W - 1)
        y = (points[..., 1] + 1.0) * 0.5 * float(self.H - 1)
        return torch.stack([x, y], dim=-1)

    def _effective_strength(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.strength_logit is None or self.max_strength <= 0.0:
            return torch.zeros((), dtype=dtype, device=device)
        return self.max_strength * torch.sigmoid(self.strength_logit.to(device=device, dtype=dtype))

    def forward(
            self,
            logits: torch.Tensor,
            feat: torch.Tensor,
            points_m11: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.radius <= 0.0:
            return logits
        if logits.numel() == 0:
            return logits

        logits_f = logits.float()
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        valid_f = valid.unsqueeze(-1).to(dtype=feat.dtype, device=feat.device)

        raw_score = selector_object_logit(logits_f)
        delta = self.score_delta(feat.float() * valid_f.float()).squeeze(-1)
        if self.residual_scale > 0.0:
            delta = torch.tanh(delta) * float(self.residual_scale)
        comp_score = raw_score + delta

        pix = self._pixel_points(points_m11.to(device=logits.device))
        dist = torch.cdist(pix, pix, p=2)
        allowed = (dist <= float(self.radius)) & valid[:, :, None] & valid[:, None, :]
        if self.exclude_self:
            eye = torch.eye(allowed.size(1), dtype=torch.bool, device=allowed.device).unsqueeze(0)
            allowed = allowed & ~eye

        has_neighbor = allowed.any(dim=-1)
        neighbor_scores = comp_score[:, None, :].masked_fill(~allowed, -1.0e4)
        neighbor_lse = torch.logsumexp(neighbor_scores / self.temperature, dim=-1) * self.temperature
        neighbor_lse = torch.where(has_neighbor, neighbor_lse, torch.zeros_like(comp_score))

        suppression = F.softplus((neighbor_lse - comp_score) / self.temperature) * self.temperature
        suppression = torch.where(has_neighbor, suppression, torch.zeros_like(suppression))
        strength = self._effective_strength(raw_score.dtype, raw_score.device)
        final_score = comp_score - strength * suppression
        final_score = torch.where(valid, final_score, raw_score)
        score_shift = (final_score - raw_score).clamp(-20.0, 20.0)

        if logits.dim() == 3 and logits.size(-1) == 2:
            out = logits_f.clone()
            out[..., 1] = out[..., 1] + score_shift
        elif logits.dim() == 3 and logits.size(-1) == 1:
            out = logits_f + score_shift.unsqueeze(-1)
        else:
            out = logits_f + score_shift
        return out.to(dtype=logits.dtype)


def fourier_encode_points(points_m11: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Fourier encode normalized point coordinates for position-aware confidence."""
    num_bands = int(num_bands)
    if num_bands <= 0:
        return points_m11.new_empty((*points_m11.shape[:-1], 0))

    freqs = (2.0 ** torch.arange(num_bands, device=points_m11.device, dtype=points_m11.dtype)) * math.pi
    angles = points_m11[..., None] * freqs  # [B,N,2,K]
    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,N,2,2K]
    return pe.flatten(start_dim=-2)  # [B,N,4K]


def selector_local_features(
        points_m11: torch.Tensor,
        valid_mask: torch.Tensor,
        radii=(3.0, 6.0),
        H: int = 256,
        W: int = 256,
) -> torch.Tensor:
    """Local duplicate/context features for selector scoring."""
    points = points_m11.detach().float()
    valid = valid_mask.bool() if valid_mask is not None else torch.ones(
        points.shape[:2], dtype=torch.bool, device=points.device
    )
    x = (points[..., 0] + 1.0) * 0.5 * float(W - 1)
    y = (points[..., 1] + 1.0) * 0.5 * float(H - 1)
    pix = torch.stack([x, y], dim=-1)

    dist = torch.cdist(pix, pix, p=2)
    B, N, _ = dist.shape
    eye = torch.eye(N, dtype=torch.bool, device=dist.device).unsqueeze(0)
    pair_valid = valid[:, :, None] & valid[:, None, :] & (~eye)
    max_radius = max([float(r) for r in radii] + [1.0])

    feats = []
    for radius in radii:
        count = ((dist <= float(radius)) & pair_valid).sum(dim=-1).to(points.dtype)
        feats.append(torch.log1p(count) / math.log1p(32.0))

    masked_dist = dist.masked_fill(~pair_valid, max_radius)
    nn_dist = masked_dist.min(dim=-1).values.clamp(max=max_radius) / max_radius
    feats.append(nn_dist.to(points.dtype))
    feats.append(valid.to(dtype=points.dtype))
    return torch.stack(feats, dim=-1)


def sample_selector_prior_features(
        prior_maps,
        points_m11: torch.Tensor,
        valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prior_maps is None:
        return None
    occ_logits, density = prior_maps
    if occ_logits is None or density is None:
        return None

    occ = torch.sigmoid(occ_logits.float())
    dens = torch.log1p(density.float().clamp_min(0.0))
    maps = torch.cat([occ, dens], dim=1)
    grid = points_m11.detach().float().clamp(-1.0, 1.0).unsqueeze(2)
    sampled = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2)
    if valid_mask is not None:
        sampled = sampled * valid_mask.unsqueeze(-1).to(dtype=sampled.dtype)
    return sampled


class ModelBuilder(nn.Module):
    """
    介面：
      encode(images) -> (P4,P8,P16)
      denoise(feats, p_t, t) -> eps_pred, score
    """
    def __init__(
            self,
            in_ch=1,
            fpn_c=128,
            cond_c=64,
            t_dim=256,
            with_score=True,
            conf_pos_bands=8,
            selector_local_features=False,
            selector_prior_features=False,
            num_refine=3,
            use_proposal_prior=False,
            proposal_prior_hidden=64,
            selector_refine_gate=False,
            selector_refine_gate_strength=0.0,
            selector_refine_gate_detach=True,
            selector_conf_weighted_merge=False,
            selector_merge_weight_min=0.05,
            selector_merge_radius_px=0.0,
            selector_merge_weight_detach=True,
            selector_refine_point_gate=False,
            selector_refine_point_gate_strength=0.0,
            selector_refine_point_gate_min=0.25,
            selector_refine_point_gate_detach=True,
            selector_interaction=False,
            selector_interaction_layers=1,
            selector_interaction_heads=4,
            selector_interaction_radius=12.0,
            selector_interaction_dropout=0.1,
            selector_local_competition=False,
            selector_local_competition_radius=6.0,
            selector_local_competition_temperature=0.7,
            selector_local_competition_strength=0.35,
            selector_local_competition_init_strength=0.05,
            selector_local_competition_residual_scale=0.25,
            selector_local_competition_exclude_self=True,
            selector_dim=None,
            selector_feature_fusion="add",
    ):
        super().__init__()
        self.backbone = EncoderFPN(in_ch=in_ch, out_c=fpn_c)
        self.temb = TimestepEmbed(dim=t_dim)
        self.cond = PointConditioner(c_fpn=fpn_c, cond_c=cond_c, patch=5, with_gate=True)
        # #self.head_eps = EpsHeadFiLM(pf_dim=cond_c * 4, t_dim=t_dim, hidden=512, depth=6, dropout=0.1)
        # self.head_eps = DenoiserHeadRes(in_dim=cond_c * 4 + t_dim + 2, hidden=384, depth=3, dropout=0.2)
        self.pf_proj = nn.Linear(cond_c * 4, cond_c)  # proposal feature dim = cond_c
        self.num_refine = max(1, int(num_refine))
        token_count = 3 * (self.cond.patch * self.cond.patch)
        self.head_eps = nn.ModuleList([
            PointRCNNHead(
                d_model=cond_c,
                nhead=4,
                dim_ff=cond_c * 4,
                dim_dynamic=max(8, cond_c // 2),
                t_dim=t_dim,
                dropout=0.1,
                token_count=token_count
            ) for _ in range(self.num_refine)
        ])
        #self.conf_head = ConfidenceHead(in_dim=cond_c*6, hidden=256, p_prior=0.07)
        # ModelBuilder.__init__()
        self.conf_pos_bands = int(conf_pos_bands)
        conf_pos_dim = 4 * self.conf_pos_bands
        if conf_pos_dim > 0:
            self.conf_pos_proj = nn.Sequential(
                nn.Linear(conf_pos_dim, cond_c),
                nn.SiLU(),
                nn.Linear(cond_c, cond_c),
            )
            # Preserve old checkpoint behavior at load time; fine-tuning can
            # learn to use this residual position branch from zero.
            with torch.no_grad():
                self.conf_pos_proj[-1].weight.zero_()
                self.conf_pos_proj[-1].bias.zero_()
        else:
            self.conf_pos_proj = None
        self.selector_local_features = bool(selector_local_features)
        if self.selector_local_features:
            self.selector_local_proj = nn.Sequential(
                nn.Linear(4, cond_c),
                nn.SiLU(),
                nn.Linear(cond_c, cond_c),
            )
            with torch.no_grad():
                self.selector_local_proj[-1].weight.zero_()
                self.selector_local_proj[-1].bias.zero_()
        else:
            self.selector_local_proj = None

        self.selector_prior_features = bool(selector_prior_features)
        if self.selector_prior_features:
            self.selector_prior_proj = nn.Sequential(
                nn.Linear(2, cond_c),
                nn.SiLU(),
                nn.Linear(cond_c, cond_c),
            )
            with torch.no_grad():
                self.selector_prior_proj[-1].weight.zero_()
                self.selector_prior_proj[-1].bias.zero_()
        else:
            self.selector_prior_proj = None
        self.selector_dim = int(selector_dim or cond_c)
        if self.selector_dim <= 0:
            raise ValueError(f"selector_dim must be positive, got {selector_dim}")
        self.selector_feature_fusion = str(selector_feature_fusion).strip().lower().replace("-", "_")
        if self.selector_feature_fusion not in {"add", "concat_proj"}:
            raise ValueError(f"unsupported selector_feature_fusion={selector_feature_fusion!r}")

        fusion_in_dim = cond_c
        if self.selector_feature_fusion == "concat_proj":
            if self.conf_pos_proj is not None:
                fusion_in_dim += cond_c
            if self.selector_local_proj is not None:
                fusion_in_dim += cond_c
            if self.selector_prior_proj is not None:
                fusion_in_dim += cond_c
            self.selector_fuse_proj = nn.Sequential(
                nn.LayerNorm(fusion_in_dim),
                nn.Linear(fusion_in_dim, self.selector_dim),
                nn.SiLU(),
                nn.Linear(self.selector_dim, self.selector_dim),
            )
        elif self.selector_dim != cond_c:
            self.selector_fuse_proj = nn.Sequential(
                nn.LayerNorm(cond_c),
                nn.Linear(cond_c, self.selector_dim),
            )
        else:
            self.selector_fuse_proj = None

        self.conf_from_pro = ConfidenceHead(in_dim=self.selector_dim, hidden=256, p_prior=0.07)
        self.selector_refine_gate = bool(selector_refine_gate)
        self.selector_refine_gate_strength = float(selector_refine_gate_strength)
        self.selector_refine_gate_detach = bool(selector_refine_gate_detach)
        self.selector_conf_weighted_merge = bool(selector_conf_weighted_merge)
        self.selector_merge_weight_min = float(selector_merge_weight_min)
        self.selector_merge_radius_px = float(selector_merge_radius_px)
        self.selector_merge_weight_detach = bool(selector_merge_weight_detach)
        self.selector_refine_point_gate = bool(selector_refine_point_gate)
        self.selector_refine_point_gate_strength = float(selector_refine_point_gate_strength)
        self.selector_refine_point_gate_min = float(selector_refine_point_gate_min)
        self.selector_refine_point_gate_detach = bool(selector_refine_point_gate_detach)
        self.selector_interaction = (
            CandidateInteractionSelector(
                d_model=self.selector_dim,
                num_layers=int(selector_interaction_layers),
                nhead=int(selector_interaction_heads),
                dropout=float(selector_interaction_dropout),
                radius=float(selector_interaction_radius),
            )
            if bool(selector_interaction)
            else None
        )
        self.selector_local_competition = (
            LocalCompetitionSelector(
                d_model=self.selector_dim,
                radius=float(selector_local_competition_radius),
                temperature=float(selector_local_competition_temperature),
                max_strength=float(selector_local_competition_strength),
                init_strength=float(selector_local_competition_init_strength),
                residual_scale=float(selector_local_competition_residual_scale),
                exclude_self=bool(selector_local_competition_exclude_self),
            )
            if bool(selector_local_competition)
            else None
        )
        self.use_proposal_prior = bool(use_proposal_prior)
        self.proposal_prior_head = (
            ProposalPriorHead(in_ch=fpn_c, hidden=int(proposal_prior_hidden))
            if self.use_proposal_prior
            else None
        )

    def encode(self, images):
        # images: [B,in_ch,H,W]
        return self.backbone(images)

    def predict_proposal_prior(self, feats):
        if self.proposal_prior_head is None:
            raise RuntimeError("proposal prior head is disabled")
        p4, _, _ = feats
        return self.proposal_prior_head(p4)

    def _selector_logits(
            self,
            points_m11: torch.Tensor,
            pro: torch.Tensor,
            valid_mask: torch.Tensor = None,
            selector_prior_maps=None,
            check_finite=None,
            name_prefix: str = "selector",
            use_local_features: bool = True,
            use_interaction: bool = True,
    ):
        if valid_mask is None:
            valid_mask = torch.ones(
                points_m11.shape[:2],
                dtype=torch.bool,
                device=points_m11.device,
            )
        else:
            valid_mask = valid_mask.bool()

        with torch.autocast(device_type="cuda", enabled=False):
            points_f = points_m11.to(device=pro.device, dtype=torch.float32)
            valid_f = valid_mask.unsqueeze(-1).to(device=pro.device, dtype=torch.float32)
            base_feat = pro.float() * valid_f
            conf_feat = base_feat
            concat_parts = [base_feat]
            if self.conf_pos_proj is not None:
                pos_enc = fourier_encode_points(points_f, self.conf_pos_bands)
                pos_feat = self.conf_pos_proj(pos_enc)
                pos_feat = pos_feat * valid_f
                if self.selector_feature_fusion == "concat_proj":
                    concat_parts.append(pos_feat)
                else:
                    conf_feat = conf_feat + pos_feat
                if check_finite is not None:
                    check_finite(f"{name_prefix}_pos_feat", pos_feat)
            if use_local_features and self.selector_local_proj is not None:
                local_feat = selector_local_features(
                    points_f,
                    valid_mask,
                    radii=(3.0, 6.0),
                    H=256,
                    W=256,
                )
                local_feat = local_feat.to(device=pro.device, dtype=torch.float32)
                local_ctx = self.selector_local_proj(local_feat)
                local_ctx = local_ctx * valid_f
                if self.selector_feature_fusion == "concat_proj":
                    concat_parts.append(local_ctx)
                else:
                    conf_feat = conf_feat + local_ctx
                if check_finite is not None:
                    check_finite(f"{name_prefix}_local_ctx", local_ctx)
            elif self.selector_feature_fusion == "concat_proj" and self.selector_local_proj is not None:
                concat_parts.append(torch.zeros_like(base_feat))
            if self.selector_prior_proj is not None:
                prior_feat = sample_selector_prior_features(
                    selector_prior_maps,
                    points_f,
                    valid_mask,
                )
                if prior_feat is not None:
                    prior_feat = prior_feat.to(device=pro.device, dtype=torch.float32)
                    prior_ctx = self.selector_prior_proj(prior_feat)
                    prior_ctx = prior_ctx * valid_f
                    if self.selector_feature_fusion == "concat_proj":
                        concat_parts.append(prior_ctx)
                    else:
                        conf_feat = conf_feat + prior_ctx
                    if check_finite is not None:
                        check_finite(f"{name_prefix}_prior_ctx", prior_ctx)
                elif self.selector_feature_fusion == "concat_proj":
                    concat_parts.append(torch.zeros_like(base_feat))
            if self.selector_feature_fusion == "concat_proj":
                conf_feat = torch.cat(concat_parts, dim=-1)
                conf_feat = self.selector_fuse_proj(conf_feat)
                conf_feat = conf_feat * valid_mask.unsqueeze(-1).to(
                    device=conf_feat.device,
                    dtype=conf_feat.dtype,
                )
                if check_finite is not None:
                    check_finite(f"{name_prefix}_fused_feat", conf_feat)
            elif self.selector_fuse_proj is not None:
                conf_feat = self.selector_fuse_proj(conf_feat)
                conf_feat = conf_feat * valid_mask.unsqueeze(-1).to(
                    device=conf_feat.device,
                    dtype=conf_feat.dtype,
                )
                if check_finite is not None:
                    check_finite(f"{name_prefix}_fused_feat", conf_feat)
            if use_interaction and self.selector_interaction is not None:
                conf_feat = self.selector_interaction(conf_feat, points_f, valid_mask)
                if check_finite is not None:
                    check_finite(f"{name_prefix}_interaction_feat", conf_feat)
            logits = self.conf_from_pro(conf_feat)
            if use_interaction and self.selector_local_competition is not None:
                logits = self.selector_local_competition(logits, conf_feat, points_f, valid_mask)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits.clamp(-20, 20)
        if check_finite is not None:
            check_finite(f"{name_prefix}_logits", logits)
        return logits

    def denoise(
            self,
            feats,
            p_t,
            t,
            abar_t=None,
            clamp_eps=1e-6,
            cond_cache=None,
            need_exist=True,
            selector_prior_maps=None,
    ):
        P4, P8, P16 = feats
        cache = cond_cache
        if cache is None:
            cache = self.cond.precompute(P4, P8, P16)

        def check_finite(name, x):
            if x is None:
                return
            if not torch.isfinite(x).all():
                x_det = x.detach()
                nan_count = torch.isnan(x_det).sum().item()
                inf_count = torch.isinf(x_det).sum().item()
                safe_abs = torch.nan_to_num(
                    x_det.abs(), nan=0.0, posinf=0.0, neginf=0.0
                )
                print(f"[BAD] {name} not finite | nan={nan_count} inf={inf_count} max_abs={safe_abs.max().item():.6f}")
                raise RuntimeError(f"{name} became non-finite")

        # time embedding
        te = self.temb(t)
        if te.dim() == 3 and te.size(1) == 1:
            te_img = te.squeeze(1)
        elif te.dim() == 3:
            te_img = te[:, 0, :]
        else:
            te_img = te

        check_finite("p_t", p_t)
        check_finite("te_img", te_img)

        # DiffusionDet-style refine
        p_noisy = p_t
        p_ref = p_t

        pf = self.cond.forward_cached(cache, p_ref)  # [B,N,4C]
        check_finite("pf", pf)

        check_finite("pf_proj.weight", self.pf_proj.weight)
        if self.pf_proj.bias is not None:
            check_finite("pf_proj.bias", self.pf_proj.bias)

        # Run the projection in FP32 to avoid half-precision overflow right
        # after conditioning, which is where training has been blowing up.
        with torch.autocast(device_type="cuda", enabled=False):
            pro = self.pf_proj(pf.float())  # [B,N,C]
        pro = torch.nan_to_num(pro, nan=0.0, posinf=1e4, neginf=-1e4)
        check_finite("pro_after_proj", pro)

        eps_pred = None
        x0_hat_last = None

        # prepare abar
        if abar_t is not None:
            abar = abar_t
            if abar.dim() == 3 and abar.size(1) == 1:
                abar = abar.expand(p_noisy.size(0), p_noisy.size(1), 1)
            elif abar.dim() == 1:
                B = p_noisy.size(0)
                abar = abar.view(B, 1, 1).expand(B, p_noisy.size(1), 1)
            elif abar.dim() == 2:
                B = p_noisy.size(0)
                abar = abar.view(B, 1, 1).expand(B, p_noisy.size(1), 1)

            sqrt_abar = (abar + clamp_eps).sqrt()
            sqrt_onem = (1.0 - abar).clamp_min(0).sqrt()

            check_finite("abar", abar)
            check_finite("sqrt_abar", sqrt_abar)
            check_finite("sqrt_onem", sqrt_onem)

        for i, head in enumerate(self.head_eps):
            tok4 = sample_point_tokens(cache["q4"], p_ref, patch=self.cond.patch)
            tok8 = sample_point_tokens(cache["q8"], p_ref, patch=self.cond.patch)
            tok16 = sample_point_tokens(cache["q16"], p_ref, patch=self.cond.patch)
            local_tokens = torch.cat([tok4, tok8, tok16], dim=0)

            check_finite(f"tok4_{i}", tok4)
            check_finite(f"tok8_{i}", tok8)
            check_finite(f"tok16_{i}", tok16)
            check_finite(f"local_tokens_{i}", local_tokens)

            with torch.autocast(device_type="cuda", enabled=False):
                eps_i, pro = head(
                    local_tokens.float(),
                    pro.float(),
                    te_img.float()
                )

            # 暫時止血用；如果你想先定位根因，也可以先註解掉這兩行
            eps_i = torch.nan_to_num(eps_i, nan=0.0, posinf=1e4, neginf=-1e4)
            pro = torch.nan_to_num(pro, nan=0.0, posinf=1e4, neginf=-1e4)

            check_finite(f"eps_i_{i}", eps_i)
            check_finite(f"pro_after_head_{i}", pro)

            eps_pred = eps_i

            if abar_t is not None:
                # 這段也先強制 FP32
                with torch.autocast(device_type="cuda", enabled=False):
                    x0_hat = (p_noisy.float() - sqrt_onem.float() * eps_i.float()) / (sqrt_abar.float() + 1e-12)
                    x0_hat = x0_hat.clamp(-1.0 + 1e-3, 1.0 - 1e-3)

                x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=1.0 - 1e-3, neginf=-1.0 + 1e-3)
                check_finite(f"x0_hat_{i}", x0_hat)

                x0_hat_last = x0_hat
                gate_prob = None
                need_feature_gate = (
                    self.selector_refine_gate
                    and self.selector_refine_gate_strength > 0.0
                )
                need_point_gate = (
                    self.selector_refine_point_gate
                    and self.selector_refine_point_gate_strength > 0.0
                )
                need_any_refine_gate = (
                    i < len(self.head_eps) - 1
                    and (need_feature_gate or need_point_gate)
                )
                if need_any_refine_gate:
                    gate_logits = self._selector_logits(
                        x0_hat.detach(),
                        pro,
                        valid_mask=None,
                        selector_prior_maps=selector_prior_maps,
                        check_finite=check_finite,
                        name_prefix=f"refine_gate_{i}",
                        use_local_features=False,
                        use_interaction=False,
                    )
                    gate_prob = selector_object_prob(gate_logits).unsqueeze(-1)

                if need_any_refine_gate and need_feature_gate:
                    feature_gate_prob = gate_prob
                    if self.selector_refine_gate_detach:
                        feature_gate_prob = feature_gate_prob.detach()
                    gate_center = feature_gate_prob.mean(dim=1, keepdim=True)
                    gate_strength = float(self.selector_refine_gate_strength)
                    gate_scale = 1.0 + gate_strength * (feature_gate_prob - gate_center)
                    gate_scale = gate_scale.clamp(
                        min=max(0.05, 1.0 - gate_strength),
                        max=1.0 + gate_strength,
                    )
                    pro = pro * gate_scale.to(dtype=pro.dtype)
                    check_finite(f"pro_after_refine_gate_{i}", pro)

                p_next = x0_hat
                if need_any_refine_gate and need_point_gate:
                    point_gate_prob = gate_prob
                    if self.selector_refine_point_gate_detach:
                        point_gate_prob = point_gate_prob.detach()
                    point_strength = float(self.selector_refine_point_gate_strength)
                    point_min = float(self.selector_refine_point_gate_min)
                    update_scale = (1.0 - point_strength) + point_strength * point_gate_prob
                    update_scale = update_scale.clamp(
                        min=max(0.0, min(point_min, 1.0)),
                        max=1.0,
                    )
                    p_next = p_ref + update_scale.to(dtype=x0_hat.dtype) * (x0_hat - p_ref)
                    p_next = p_next.clamp(-1.0 + 1e-3, 1.0 - 1e-3)
                    check_finite(f"p_ref_after_point_gate_{i}", p_next)

                p_ref = p_next.detach()
            else:
                p_ref = p_ref

        cls_logits = None
        pro_out = pro
        pred_points_for_cls = x0_hat_last
        self.last_merge_stats = None
        pred_valid_mask = torch.ones(
            (pro.size(0), pro.size(1)),
            dtype=torch.bool,
            device=pro.device
        )
        if need_exist:
            if x0_hat_last is None:
                raise RuntimeError("need_exist=True but x0_hat_last is None")

            merge_weight = None
            if self.selector_conf_weighted_merge:
                raw_selector_logits = self._selector_logits(
                    x0_hat_last.detach(),
                    pro,
                    valid_mask=None,
                    selector_prior_maps=selector_prior_maps,
                    check_finite=check_finite,
                    name_prefix="premerge_selector",
                    use_interaction=False,
                )
                merge_weight = selector_object_prob(raw_selector_logits)
                if self.selector_merge_weight_detach:
                    merge_weight = merge_weight.detach()

            # merge 也很可能是 NaN 源頭，先強制 FP32
            with torch.autocast(device_type="cuda", enabled=False):
                merged_x0_hat, merged_pro, merged_valid_mask = merge_slots_same_pixel(
                    x0_hat=x0_hat_last.detach().float(),
                    pro=pro.float(),
                    H=256,
                    W=256,
                    max_slots=pro.size(1),
                    weight=merge_weight,
                    weight_min=self.selector_merge_weight_min,
                    merge_radius_px=self.selector_merge_radius_px,
                )

            merged_pro = torch.nan_to_num(merged_pro, nan=0.0, posinf=1e4, neginf=-1e4)
            check_finite("merged_pro", merged_pro)
            raw_slot_count = torch.full(
                (x0_hat_last.size(0),),
                int(x0_hat_last.size(1)),
                dtype=torch.long,
                device=x0_hat_last.device,
            )
            merged_valid_count = merged_valid_mask.long().sum(dim=1)
            self.last_merge_stats = {
                "raw_slot_count": raw_slot_count.detach(),
                "merged_valid_count": merged_valid_count.detach(),
                "merge_drop_count": (raw_slot_count - merged_valid_count).detach(),
                "raw_x0_hat": x0_hat_last.detach(),
                "merged_x0_hat": merged_x0_hat.detach(),
                "merged_valid_mask": merged_valid_mask.detach(),
            }

            cls_logits = self._selector_logits(
                merged_x0_hat,
                merged_pro,
                valid_mask=merged_valid_mask,
                selector_prior_maps=selector_prior_maps,
                check_finite=check_finite,
                name_prefix="selector",
            )

            pro_out = merged_pro
            pred_points_for_cls = merged_x0_hat.to(pro.device)
            pred_valid_mask = merged_valid_mask

        return eps_pred, cls_logits, pro_out, pred_points_for_cls, pred_valid_mask
