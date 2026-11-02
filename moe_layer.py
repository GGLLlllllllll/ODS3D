import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FeatureSplit(nn.Module):
    def __init__(self, in_ch, out_ch, cutoff_ratio=0.25):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

        self.str_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.tex_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.geo_conv = nn.Sequential(
            nn.Conv3d(in_ch + 3, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _build_gaussian_lowpass(self, D, H, W, device, dtype):
        """
          Gaussian   mask mask 
        """
        z = torch.arange(D, device=device, dtype=torch.float32)
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)

        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

        cz, cy, cx = D // 2, H // 2, W // 2
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2

        max_radius = min(D, H, W) / 2.0
        sigma = self.cutoff_ratio * max_radius

        low_mask = torch.exp(-dist2 / (2 * sigma * sigma)).to(dtype)

        return low_mask

    def _fourier_decompose(self, x):
        """
         
        - Gaussian 
        -  
        """
        B, C, D, H, W = x.shape
        device, dtype = x.device, x.dtype

        # FFT
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))

        # Gaussian low-pass
        low_mask = self._build_gaussian_lowpass(D, H, W, device, dtype)
        low_mask = low_mask[None, None, :, :, :]

        x_low_fft = x_fft * low_mask

        # inverse shift
        x_low_fft = torch.fft.ifftshift(x_low_fft, dim=(-3, -2, -1))

        # iFFT
        x_low = torch.fft.ifftn(x_low_fft, dim=(-3, -2, -1)).real

        x_high = x - x_low

        return x_low, x_high

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_low, x_high = self._fourier_decompose(x)

        f_str = self.str_conv(x+x_low)

        f_tex = self.tex_conv(x+x_high)

        z = torch.linspace(-1, 1, D, device=x.device, dtype=x.dtype)
        y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        x_ = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)

        zz, yy, xx = torch.meshgrid(z, y, x_, indexing="ij")
        grid = torch.stack([zz, yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1, -1)

        f_geo = self.geo_conv(torch.cat([x, grid], dim=1))

        return f_str, f_tex, f_geo

class PWExperts(nn.Module):
    """3 fixed experts: geometry, texture, structure"""
    def __init__(self, in_channels, dropout=0.0):
        super(PWExperts, self).__init__()
        self.feature_split = FeatureSplit(in_channels, in_channels, cutoff_ratio=0.25)
        
        self.geo_expert = PWLayer(in_channels, in_channels, dropout)
        self.tex_expert = PWLayer(in_channels, in_channels, dropout)
        self.str_expert = PWLayer(in_channels, in_channels, dropout)

    def forward(self, x):
        B, C, D, H, W = x.shape
        f_str, f_tex, f_geo = self.feature_split(x)
        # f_str, f_tex, f_geo = x, x, x
        
        f_str = f_str.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        f_tex = f_tex.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        f_geo = f_geo.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        str_out = self.str_expert(f_str).unsqueeze(1)
        tex_out = self.tex_expert(f_tex).unsqueeze(1)
        geo_out = self.geo_expert(f_geo).unsqueeze(1)  # (B,1,D)
        
        return str_out, tex_out, geo_out

class share_PWExperts(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super(share_PWExperts, self).__init__()
        
        self.shared_lin = nn.Linear(in_channels, in_channels, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
        self.bias_geo = nn.Parameter(torch.zeros(in_channels))
        self.bias_tex = nn.Parameter(torch.zeros(in_channels))
        self.bias_str = nn.Parameter(torch.zeros(in_channels))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        x = self.dropout(x)

        geo_out = self.shared_lin(x - self.bias_geo).unsqueeze(1)  # (B,1,D)
        tex_out = self.shared_lin(x - self.bias_tex).unsqueeze(1)
        str_out = self.shared_lin(x - self.bias_str).unsqueeze(1)

        return str_out, tex_out, geo_out

class GateModule(nn.Module):
    """Compute gating weights for experts"""
    def __init__(self, in_channels, n_experts=3, noise=True):
        super(GateModule, self).__init__()
        self.noisy_gating = noise
        self.w_gate = nn.Parameter(torch.zeros(in_channels, n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(in_channels, n_experts), requires_grad=True)

    def forward(self, x, r=None, train=None, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and (train or self.training):
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits
        if r is not None:
            gates = F.softmax(logits / torch.sigmoid(r), dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates  # (B, n_experts)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[2] - x1.size()[2]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, tag=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear :
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if tag == 2:
                self.up = nn.ConvTranspose3d(in_channels, in_channels // 4, kernel_size=2, stride=2)
                self.conv = DoubleConv((in_channels // 4) * 3, out_channels)
            if tag == 3:
                self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv((in_channels // 2) * 3, out_channels)

    def forward(self, x, x2, x3):
        x1 = self.up(x)
        # input is CHW       
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[2] - x1.size()[2]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)
