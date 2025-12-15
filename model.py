import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Tuple

# ----------------- UncerNet ------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MLP1x1(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)

class FBlock(nn.Module):
    def __init__(self, in_x_ch: int, feat_ch: int):
        super().__init__()
        self.conv_x = ConvBNReLU(in_x_ch, feat_ch, k=3)
        self.conv_z = ConvBNReLU(feat_ch, feat_ch * 4, k=3)  
        self.ps = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x, z):
        x_feat = self.conv_x(x)
        z_up = self.ps(self.conv_z(z))
        if z_up.shape[-2:] != x_feat.shape[-2:]:
            z_up = F.interpolate(z_up, size=x_feat.shape[-2:], mode="bilinear", align_corners=False)
        return x_feat + z_up

class JointFPN(nn.Module):
    def __init__(self, base_ch=64, seg_in_ch=3, seg_down_mode="bilinear"):
        super().__init__()
        self.base_ch = base_ch
        self.seg_in_ch = seg_in_ch
        self.seg_down_mode = seg_down_mode

        c = base_ch
        self.f3_conv = nn.Conv2d(3, c, kernel_size=1, bias=False)
        self.s3_conv = nn.Conv2d(seg_in_ch, c, kernel_size=1, bias=False)

        self.f_block_2 = FBlock(in_x_ch=3, feat_ch=c)
        self.f_block_1 = FBlock(in_x_ch=3, feat_ch=c)
        self.s_block_2 = FBlock(in_x_ch=seg_in_ch, feat_ch=c)
        self.s_block_1 = FBlock(in_x_ch=seg_in_ch, feat_ch=c)

    def _down_rgb(self, x):
        H, W = x.shape[-2:]
        return F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=False)

    def _down_seg(self, x):
        H, W = x.shape[-2:]
        if self.seg_down_mode == "nearest":
            return F.interpolate(x, size=(H // 2, W // 2), mode="nearest")
        return F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=False)

    def forward(self, y_coarse, phi_hat):
        y1 = y_coarse
        y2 = self._down_rgb(y1)
        y3 = self._down_rgb(y2)
        p1 = phi_hat
        p2 = self._down_seg(p1)
        p3 = self._down_seg(p2)

        f3 = self.f3_conv(y3)
        s3 = self.s3_conv(p3)
        f2 = self.f_block_2(y2, f3)
        s2 = self.s_block_2(p2, s3)
        f1 = self.f_block_1(y1, f2)
        s1 = self.s_block_1(p1, s2)

        return (f1, f2, f3), (s1, s2, s3), (y1, y2, y3)

class ChannelPool(nn.Module):
    def forward(self, x):
        mx, _ = torch.max(x, dim=1, keepdim=True)
        avg = torch.mean(x, dim=1, keepdim=True)
        return mx, avg

class UncerMask(nn.Module):
    def __init__(self, feat_ch, sem_ch, hidden_ch=64):
        super().__init__()
        self.fuse = ConvBNReLU(feat_ch + sem_ch, hidden_ch, k=3)
        self.ctx = ConvBNReLU(hidden_ch, hidden_ch, k=3)
        self.pool = ChannelPool()
        self.mlp = MLP1x1(in_ch=hidden_ch + 2, hidden_ch=hidden_ch)

    def forward(self, f_k, s_k):
        z = self.fuse(torch.cat([f_k, s_k], dim=1))
        z_ctx = self.ctx(z)
        z_max, z_avg = self.pool(z)
        q = torch.cat([z_max, z_avg, z_ctx], dim=1)
        logits = self.mlp(q)
        return torch.sigmoid(logits) 

class SPADE(nn.Module):
    def __init__(self, feat_ch, sem_ch, hidden=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feat_ch, affine=False)

        self.sem_enc = nn.Sequential(
            ConvBNReLU(sem_ch, hidden, k=3),
            ConvBNReLU(hidden, hidden, k=3),
        )
        self.gamma = nn.Conv2d(hidden, feat_ch, kernel_size=3, padding=1, bias=True)
        self.beta = nn.Conv2d(hidden, feat_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, f_k, s_k, U_k):
        h = self.sem_enc(s_k)
        g = self.gamma(h)
        b = self.beta(h)
        fn = self.norm(f_k)
        return (1.0 + U_k * g) * fn + U_k * b


class RefineBlock(nn.Module):
    def __init__(self, feat_ch):
        super().__init__()
        self.y_path = ConvBNReLU(3, feat_ch, k=7, p=3)      
        self.h_conv = ConvBNReLU(2 * feat_ch, feat_ch, k=3) 
        self.o_conv = ConvBNReLU(2 * feat_ch, feat_ch, k=3) 
        self.to_rgb = nn.Conv2d(feat_ch, 3, kernel_size=1, bias=True)

    def forward(self, y_pre, f_k, f_ks):
        H, W = f_k.shape[-2:]
        if y_pre.shape[-2:] != (H, W):
            y_pre = F.interpolate(y_pre, size=(H, W), mode="bilinear", align_corners=False)

        y_feat = self.y_path(y_pre)
        h = self.h_conv(torch.cat([y_feat, f_k], dim=1))
        o = self.o_conv(torch.cat([h, f_ks], dim=1))
        y_hat = self.to_rgb(o)
        return y_hat


class USENetwork(nn.Module):
    def __init__(self, base_ch=64, seg_in_ch=3, seg_down_mode="bilinear"):
        super().__init__()
        self.joint_fpn = JointFPN(base_ch=base_ch, seg_in_ch=seg_in_ch, seg_down_mode=seg_down_mode)
        self.um_head_1 = UncerMask(feat_ch=base_ch, sem_ch=base_ch, hidden_ch=base_ch)
        self.um_head_2 = UncerMask(feat_ch=base_ch, sem_ch=base_ch, hidden_ch=base_ch)
        self.um_head_3 = UncerMask(feat_ch=base_ch, sem_ch=base_ch, hidden_ch=base_ch)

        self.spade_1 = SPADE(feat_ch=base_ch, sem_ch=base_ch, hidden=128)
        self.spade_2 = SPADE(feat_ch=base_ch, sem_ch=base_ch, hidden=128)
        self.spade_3 = SPADE(feat_ch=base_ch, sem_ch=base_ch, hidden=128)
        
        self.refine_1 = RefineBlock(feat_ch=base_ch)
        self.refine_2 = RefineBlock(feat_ch=base_ch)
        self.refine_3 = RefineBlock(feat_ch=base_ch)

    def forward(self, y_coarse, phi_hat):
        (f1, f2, f3), (s1, s2, s3), (y1, y2, y3) = self.joint_fpn(y_coarse, phi_hat)

        U3 = self.um_head_3(f3, s3)
        U2 = self.um_head_2(f2, s2)
        U1 = self.um_head_1(f1, s1)
        
        f3s = self.spade_3(f3, s3, U3)
        f2s = self.spade_2(f2, s2, U2)
        f1s = self.spade_1(f1, s1, U1)
        
        y_ref3 = self.refine_3(y3, f3, f3s)
        y_ref2 = self.refine_2(y_ref3, f2, f2s)
        y_ref1 = self.refine_1(y_ref2, f1, f1s)

        return (y_ref1, y_ref2, y_ref3), (U1, U2, U3), (y1, y2, y3)


# ----------------- Progressive Mask ------------------
class Model(nn.Module):
    def __init__(self, in_channels, max_mask_epoch=500):
        super().__init__()
        self.in_channels    = in_channels
        self.max_mask_epoch = max_mask_epoch

        # self.Encoder = ...
        # other modules ...
    def progression_mask(self, epoch: int, mask_size=(1, 256, 28, 28), max_epoch: int = 500):
        device = next(self.parameters()).device
        N = math.prod(mask_size)
    
        e = max(0, min(int(epoch), int(max_epoch)))
        if e >= max_epoch:
            num_black = N
        else:
            num_black = int((e ** 2) * (N / (max_epoch ** 2)))
    
        num_black = max(0, min(num_black, N))
    
        if num_black == 0:
            return torch.ones(mask_size, device=device, dtype=torch.float32)
        if num_black == N:
            return torch.zeros(mask_size, device=device, dtype=torch.float32)
    
        mask = torch.ones((N,), device=device, dtype=torch.float32)
        idx = torch.randperm(N, device=device)[:num_black]   # distinct indices
        mask[idx] = 0.0
        return mask.view(*mask_size)
