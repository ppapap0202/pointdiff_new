import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.convnext import convnext_small, ConvNeXt_Small_Weights
import math
from torchvision.ops import misc as misc_ops  # LayerNorm2d 在這裡
# ---------- Encoder + FPN ----------
class EncoderFPN(nn.Module):
    """
    輸入:  [B, in_ch, H, W]
    輸出:  P4, P8, P16  分別是 [B, out_c, H/4, W/4], [B, out_c, H/8, W/8], [B, out_c, H/16, W/16]

    backbone 選項:
      - 'resnet18'         (預設)
      - 'convnext_small'   (新增)
    """
    def __init__(self, in_ch=1, out_c=128, backbone: str = "convnext_small", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

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

        elif self.backbone_name == "convnext_small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            m = convnext_small(weights=weights)

            # torchvision ConvNeXt 結構：m.features = [stem, stage1, stage2, stage3, stage4]
            # strides: stem(/4) -> s1(/4) -> s2(/8) -> s3(/16) -> s4(/32)
            # channels (small): stem out=96, s1=96, s2=192, s3=384, s4=768
            # 若 in_ch != 3，需要同時替換 stem 的 LayerNorm2d(3) 與 Conv2d(3,96,4,4)
            if in_ch != 3:
                # conv
                old_conv = m.features[0][0]
                m.features[0][0] = nn.Conv2d(
                    in_ch, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                # norm（Conv2dNormActivation 預設用 LayerNorm2d）
                from torchvision.ops import misc as misc_ops
                m.features[0][1] = misc_ops.LayerNorm2d(in_ch, eps=1e-6)

                # （可選）把 RGB 預訓練權重平均到單通道
                with torch.no_grad():
                    if hasattr(old_conv, "weight") and old_conv.weight.shape[1] == 3:
                        new_w = old_conv.weight.data.mean(dim=1, keepdim=True)  # [96,1,4,4]
                        m.features[0][0].weight.copy_(new_w)

            # 取到 /4, /8, /16 的特徵
            self.convnext = m
            # /4 輸出位置：經 stem 與 stage1
            # /8：stage2；/16：stage3
            C4, C8, C16 = 96, 192, 384

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # FPN lateral + smooth（統一輸出通道 out_c）
        self.lat4   = nn.Conv2d(C4,  out_c, 1)
        self.lat8   = nn.Conv2d(C8,  out_c, 1)
        self.lat16  = nn.Conv2d(C16, out_c, 1)

        self.smooth4  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth8  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth16 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x):
        if self.backbone_name == "resnet18":
            x  = self.stem(x)      # /4
            c4 = self.layer1(x)    # /4,  C=64
            c8 = self.layer2(c4)   # /8,  C=128
            c16= self.layer3(c8)   # /16, C=256

        elif self.backbone_name == "convnext_small":
            # ConvNeXt features flow:
            # stem -> stage1 -> stage2 -> stage3 -> stage4
            f = self.convnext.features
            x = f[0](x)        # stem, /4
            c4 = f[1](x)       # stage1, /4, C=96
            s2 = f[2](c4)
            c8 = f[3](s2)      # stage2, /8, C=192
            s3 = f[4](c8)
            c16= f[5](s3)      # stage3, /16, C=384
            # （stage4 是 /32，本模組不需要）
            assert c4.shape[1] == 96 and c8.shape[1] == 192 and c16.shape[1] == 384, \
                f"unexpected channels: c4={c4.shape}, c8={c8.shape}, c16={c16.shape}"
        # 形狀護欄
        assert c4.shape[1] in (64, 96), f"c4 channels={c4.shape[1]} unexpected"
        assert c8.shape[1] in (128, 192), f"c8 channels={c8.shape[1]} unexpected"
        assert c16.shape[1] in (256, 384), f"c16 channels={c16.shape[1]} unexpected"
        # top-down FPN
        l16 = self.lat16(c16)
        l8  = self.lat8(c8)  + F.interpolate(l16, size=c8.shape[-2:], mode='nearest')
        l4  = self.lat4(c4)  + F.interpolate(l8,  size=c4.shape[-2:], mode='nearest')

        P16 = self.smooth16(l16)
        P8  = self.smooth8(l8)
        P4  = self.smooth4(l4)
        return P4, P8, P16

# ---------- ROI-free 點特徵取樣 ----------
def sample_point_feats(P, p_norm):
    """
    P: [B,C,h,w],  p_norm: [B,N,2] in [-1,1]
    這裡使用 align_corners=True，需與你用 (W-1)/2, (H-1)/2 的正規化完全對齊
    """
    B, N, _ = p_norm.shape
    grid = p_norm.view(B, N, 1, 2)
    feat = F.grid_sample(
        P, grid, mode='bilinear',
        align_corners=True,          # ← 關鍵
        padding_mode='border'        # ← 邊界更穩定
    )  # [B,C,N,1]
    return feat.squeeze(-1).transpose(1, 2)  # [B,N,C]

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

class PointConditioner(nn.Module):
    def __init__(self, c_fpn=128, cond_c=64, patch=1, with_gate=False):
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
        self.out_dim = cond_c * 3

    def forward(self, P4, P8, P16, p_norm):
        f4  = sample_point_feats(self.c4(P4),  p_norm, patch=self.patch)
        f8  = sample_point_feats(self.c8(P8),  p_norm, patch=self.patch)
        f16 = sample_point_feats(self.c16(P16), p_norm, patch=self.patch)
        if self.patch > 1:
            f4  = self.flat4(f4);  f8  = self.flat8(f8);  f16 = self.flat16(f16)
        if self.with_gate:
            return self.gate(f4, f8, f16)  # [B,N,3C]（結構更穩、泛化更好）
        return torch.cat([f4, f8, f16], dim=-1)

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

# ---------- 共享 Denoiser Head ----------
class DenoiserHead(nn.Module):
    def __init__(self, in_dim, hidden=256, with_score=True):
        super().__init__()
        out_dim = 2 + (1 if with_score else 0)
        self.with_score = with_score
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):  # x: [B,N,in_dim]
        y = self.mlp(x)
        if self.with_score:
            return y[..., :2], y[..., 2]  # eps_pred, score
        else:
            return y, None
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

class ConfidenceHead(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, f):  # f: [B,N,in_dim]
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
        self.cond = PointConditioner(c_fpn=fpn_c, cond_c=cond_c)
        self.head_eps = DenoiserHeadRes(in_dim=cond_c*3 + t_dim + 2, hidden=384, depth=3, dropout=0.2)
        self.conf_head = ConfidenceHead(in_dim=cond_c*3, hidden=256)

    def encode(self, images):
        # images: [B,in_ch,H,W]
        return self.backbone(images)

    def denoise(self, feats, p_t, t, abar_t=None, clamp_eps=1e-6):
        # feats: tuple(P4,P8,P16), p_t: [B,N,2] in [-1,1], t: [B,1] or [B,N]
        P4, P8, P16 = feats
        pf = self.cond(P4, P8, P16, p_t)            # [B,N,cond_c*3]
        te = self.temb(t)                           # [B,1,t_dim] or [B,N,t_dim]
        if te.dim()==3 and te.size(1)==1:
            te = te.expand(pf.size(0), pf.size(1), te.size(-1))
        x = torch.cat([pf, te, p_t], dim=-1)        # [B,N,cond_c*3 + t_dim + 2]
        #print(x.shape)
        eps_pred = self.head_eps(x)  # [B,N,2]
        # 若傳入 abar_t，還原 x0_hat；否則預設 t=0 的簡化（不傳也行）
        if abar_t is None:
            # 只有當 t=0 才合理；一般訓練會給 abar_t
            x0_hat = p_t
        else:
            # x0 = (p_t - sqrt(1-abar)*eps) / sqrt(abar)
            abar = abar_t
            if abar.dim() == 3 and abar.size(1) == 1:  # [B,1,1] → [B,N,1]
                abar = abar.expand(pf.size(0), pf.size(1), 1)
            sqrt_abar = (abar + clamp_eps).sqrt()
            sqrt_onem = (1.0 - abar).clamp_min(0).sqrt()
            # print("[DEBUG] p_t", p_t.shape, p_t.dtype)
            # print("[DEBUG] abar", abar.shape, abar.dtype)
            # print("[DEBUG] eps_pred", eps_pred.shape, eps_pred.dtype)
            x0_hat = (p_t - sqrt_onem * eps_pred) / sqrt_abar
            x0_hat = x0_hat.clamp(-1.0+1e-3, 1.0-1e-3)

        # 在 x0_hat 位置再次取樣特徵，回歸存在分數
        pf_hat = self.cond(P4, P8, P16, x0_hat.detach())  # detach 可選：讓 conf 先穩
        exist_logit = self.conf_head(pf_hat)              # [B,N]
        return eps_pred, exist_logit


