# 3D segmentation head
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers3D import SpatialTransformer, ResizeTransform, VecInt, conv_block, predict_flow, conv2D, MatchCost,\
    DoubleConv, Down, Up, Up2, Up3, OutConv, conv3D
import sys

from torch.distributions.normal import Normal

from moe_layer import *
from mrb3d import ModalityRetainmentBlock3D

class Feature_Extractor(nn.Module):
    def __init__(self, n_channels):
        super(Feature_Extractor, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class Reg_Head(nn.Module):
    def __init__(self, inshape):
        super(Reg_Head, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(32, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer()

    def forward(self, source, encoded_src):
        
        flow_field = self.flow(encoded_src)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        return y_source, pos_flow

class Seg_Head(nn.Module):
    def __init__(self, inshape, n_classes):
        super(Seg_Head, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.seg_head = OutConv(32, n_classes)

    def forward(self, x1):

        x = self.seg_head(x1)
        
        return x

class Architecture(nn.Module):
    def __init__(self, inshape=[128, 128, 128], n_channels=1, n_classes=14,
                 use_moe=False, use_mrb=True):
        super(Architecture, self).__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], f'ndims should be one of 1, 2, or 3. found: {ndims}'

        self.fea_extractor = Feature_Extractor(n_channels)
        self.Reg_Head = Reg_Head(inshape)
        self.Seg_Head = Seg_Head(inshape, n_classes)
        self.use_moe = bool(use_moe)
        self.use_mrb = bool(use_mrb)

        # FusionRegister's 2-D MRB is adapted to 3-D and inserted at the
        # 1/4-resolution registration stage (128 channels).  This retains
        # local source/target details without paying its cost at 128^3.
        self.reg_mrb = (
            ModalityRetainmentBlock3D(128, corr_max_disp=1)
            if self.use_mrb else nn.Identity()
        )

        if self.use_moe:
            self.pw1 = share_PWExperts(512)
            self.pw2 = PWExperts(128)
            self.pw3 = PWExperts(64)

            self.reg_gate1 = GateModule(512)
            self.reg_gate2 = GateModule(128)
            self.reg_gate3 = GateModule(64)

            self.seg_gate1 = GateModule(512)
            self.seg_gate2 = GateModule(128)
            self.seg_gate3 = GateModule(64)
        else:
            self.pw1 = nn.Identity()
            self.pw2 = nn.Identity()
            self.pw3 = nn.Identity()
            self.reg_gate1 = nn.Identity()
            self.reg_gate2 = nn.Identity()
            self.reg_gate3 = nn.Identity()
            self.seg_gate1 = nn.Identity()
            self.seg_gate2 = nn.Identity()
            self.seg_gate3 = nn.Identity()

        self.reg_up1 = Up3(512, 128, tag=2)
        self.reg_up2 = Up3(128, 64, tag=3)
        self.reg_up3 = Up3(64, 32, tag=3)

        self.seg_up1 = Up3(512, 128, tag=2)
        self.seg_up2 = Up3(128, 64, tag=3)
        self.seg_up3 = Up3(64, 32, tag=3)

    def _run_reg_branch(self, encoded_src, encoded_tgt, source):
        s1, s2, s3, s4 = encoded_src
        t1, t2, t3, t4 = encoded_tgt

        if not self.use_moe:
            x = self.reg_up1(torch.cat([s4, t4], dim=1), s3, t3)
            if self.use_mrb:
                x = self.reg_mrb(x, s3, t3)
            x = self.reg_up2(x, s2, t2)
            x = self.reg_up3(x, s1, t1)
            return self.Reg_Head(source, x)

        # stage 1
        x4 = torch.cat([s4, t4], dim=1)
        B, C, D, H, W = x4.shape
        x4_flat = x4.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        reg_str1, reg_tex1, reg_geo1 = self.pw1(x4)
        expert_outputs1 = torch.cat([reg_str1, reg_tex1, reg_geo1], dim=1)
        gates1 = self.reg_gate1(x4_flat, train=self.training)
        moe_out1 = torch.sum(
            gates1.unsqueeze(-1) * expert_outputs1, dim=1
        ).view(B, D, H, W, 512).permute(0, 4, 1, 2, 3).contiguous()

        x = self.reg_up1(moe_out1, s3, t3)
        if self.use_mrb:
            x = self.reg_mrb(x, s3, t3)

        # stage 2
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        reg_str2, reg_tex2, reg_geo2 = self.pw2(x)
        expert_outputs2 = torch.cat([reg_str2, reg_tex2, reg_geo2], dim=1)
        gates2 = self.reg_gate2(x_flat, train=self.training)
        moe_out2 = torch.sum(
            gates2.unsqueeze(-1) * expert_outputs2, dim=1
        ).view(B, D, H, W, 128).permute(0, 4, 1, 2, 3).contiguous()

        x = self.reg_up2(moe_out2, s2, t2)

        # stage 3
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        reg_str3, reg_tex3, reg_geo3 = self.pw3(x)
        expert_outputs3 = torch.cat([reg_str3, reg_tex3, reg_geo3], dim=1)
        gates3 = self.reg_gate3(x_flat, train=self.training)
        moe_out3 = torch.sum(
            gates3.unsqueeze(-1) * expert_outputs3, dim=1
        ).view(B, D, H, W, 64).permute(0, 4, 1, 2, 3).contiguous()

        x = self.reg_up3(moe_out3, s1, t1)

        y_source, pos_flow = self.Reg_Head(source, x)

        return y_source, pos_flow

    def _run_seg_branch(self, encoded_tgt):
        s1, s2, s3, s4 = encoded_tgt
        t1, t2, t3, t4 = encoded_tgt

        if not self.use_moe:
            x = self.seg_up1(torch.cat([s4, t4], dim=1), s3, t3)
            x = self.seg_up2(x, s2, t2)
            x = self.seg_up3(x, s1, t1)
            return self.Seg_Head(x)

        # stage 1
        x4 = torch.cat([s4, t4], dim=1)
        B, C, D, H, W = x4.shape
        x4_flat = x4.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        seg_str1, seg_tex1, seg_geo1 = self.pw1(x4)
        expert_outputs1 = torch.cat([seg_str1, seg_tex1, seg_geo1], dim=1)
        gates1 = self.seg_gate1(x4_flat, train=self.training)
        moe_out1 = torch.sum(
            gates1.unsqueeze(-1) * expert_outputs1, dim=1
        ).view(B, D, H, W, 512).permute(0, 4, 1, 2, 3).contiguous()

        x = self.seg_up1(moe_out1, s3, t3)

        # stage 2
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        seg_str2, seg_tex2, seg_geo2 = self.pw2(x)
        expert_outputs2 = torch.cat([seg_str2, seg_tex2, seg_geo2], dim=1)
        gates2 = self.seg_gate2(x_flat, train=self.training)
        moe_out2 = torch.sum(
            gates2.unsqueeze(-1) * expert_outputs2, dim=1
        ).view(B, D, H, W, 128).permute(0, 4, 1, 2, 3).contiguous()

        x = self.seg_up2(moe_out2, s2, t2)

        # stage 3
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        seg_str3, seg_tex3, seg_geo3 = self.pw3(x)
        expert_outputs3 = torch.cat([seg_str3, seg_tex3, seg_geo3], dim=1)
        gates3 = self.seg_gate3(x_flat, train=self.training)
        moe_out3 = torch.sum(
            gates3.unsqueeze(-1) * expert_outputs3, dim=1
        ).view(B, D, H, W, 64).permute(0, 4, 1, 2, 3).contiguous()

        x = self.seg_up3(moe_out3, s1, t1)

        t_logits = self.Seg_Head(x)

        return t_logits
    
    def forward(self, source: torch.Tensor, target: torch.Tensor):
        encoded_src = self.fea_extractor(source)
        encoded_tgt = self.fea_extractor(target)

        y_source, pos_flow = self._run_reg_branch(
            encoded_src, encoded_tgt, source
        )
        t_logits = self._run_seg_branch(encoded_tgt)

        return y_source, pos_flow, None, t_logits
