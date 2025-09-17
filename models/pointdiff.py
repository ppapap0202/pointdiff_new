import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math
# ---------- Encoder + FPN ----------
class EncoderFPN(nn.Module):
    """
    輸入:  [B, in_ch, H, W]
    輸出:  P4, P8, P16  分別是 [B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16]
    """
    def __init__(self, in_ch=1, out_c=128):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_ch != 3:
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # /4
        self.layer1 = m.layer1   # /4,  C=64
        self.layer2 = m.layer2   # /8,  C=128
        self.layer3 = m.layer3   # /16, C=256

        C4, C8, C16 = 64, 128, 256
        self.lat4   = nn.Conv2d(C4,  out_c, 1)
        self.lat8   = nn.Conv2d(C8,  out_c, 1)
        self.lat16  = nn.Conv2d(C16, out_c, 1)

        self.smooth4  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth8  = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.smooth16 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x):
        x  = self.stem(x)      # /4
        c4 = self.layer1(x)    # /4
        c8 = self.layer2(c4)   # /8
        c16= self.layer3(c8)   # /16

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
    P: [B,C,h,w],  p_norm: [B,N,2] in [-1,1], align_corners=False
    return: [B,N,C]
    """
    B, N, _ = p_norm.shape
    grid = p_norm.view(B, N, 1, 2)
    feat = F.grid_sample(P, grid, mode='bilinear', align_corners=False)  # [B,C,N,1]
    return feat.squeeze(-1).transpose(1, 2)  # [B,N,C]

class PointConditioner(nn.Module):
    def __init__(self, c_fpn=128, cond_c=64):
        super().__init__()
        self.c4  = nn.Conv2d(c_fpn, cond_c, 1)
        self.c8  = nn.Conv2d(c_fpn, cond_c, 1)
        self.c16 = nn.Conv2d(c_fpn, cond_c, 1)
        self.out_dim = cond_c * 3

    def forward(self, P4, P8, P16, p_norm):
        f4  = sample_point_feats(self.c4(P4),  p_norm)  # [B,N,cond_c]
        f8  = sample_point_feats(self.c8(P8),  p_norm)
        f16 = sample_point_feats(self.c16(P16), p_norm)
        return torch.cat([f4, f8, f16], dim=-1)         # [B,N,cond_c*3]

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

    @torch.no_grad()
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


