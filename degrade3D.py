# -*- coding: utf-8 -*-
"""
 Degradation Engine, DE  3D   MR  
================================================

 DORA TPAMI 2026, *Breathing New Life Into Small Object Detection With
Detection-Oriented Rectification*  III-A   1) "Degradation Engine" 

 ** ** 
  mild-to-moderate ** ** 

 " " 

| DORA  |  3D   MR  |   |
|---|---|---|
| JPEG   /   | k   | MR   JPEG   k  /  |
|  /  | Rician   | MR   Rician  SNR   |
|  /  | ** **  |   data/datasets_mixed.normalize_image_intensity   min-max  
   /     ** ** gamma 
  ** **  |
|   | PSF   /   |   MR   |
|   |   |  " "  |

  ndimage.zoom   128^3
 data/datasets_mixed.py 
 / 

 
  *   min-max   [0, 1]  clamp   [0, 1]
        train/test  
  *   [B, C, D, H, W]  [C,D,H,W] / [D,H,W] 
  *   nn.Module  checkpoint
  *   generator     no_grad 

 
    degrade(vol, family, severity, ...)            #   +  
    sample_degradation(vol, ...)                   #   DORA 
    family_table() / FAMILY_DOC                    #  
"""

import torch
import torch.nn.functional as F

FAMILIES = (
    'noise',           #
    'rician',          #
    'bias',            #
    'gamma',           #
    'blur',            #
    'slice',           #
)

FAMILY_DOC = {
    'noise': ' sigma   [0,1]  sigma   0.06 ',
    'rician': 'Rician  sqrt((x+n1)^2+n2^2)  SNR   MR  ',
    'bias': '  gain=1+A*tanh(field) ',
    'gamma': '  x^gamma gamma=1 0.45 min-max  / ',
    'blur': '  PSF  sigma   1.6  / / ',
    'slice': ' /  PSF  (  3  ) ',
    'partial_volume': ' ( ) ',
    'resample': ' (  2  ,  ) ',
    'kspace': 'k   (  35%)   Gibbs  ',
}

SEVERITY_RANGE = {
    'noise': ('sigma', 0.0, 0.06),
    'rician': ('sigma', 0.0, 0.05),
    'bias': ('amp', 0.0, 0.35),
    'gamma': ('gamma', 1.0, 1.45),      #
    'blur': ('sigma_vox', 0.0, 1.6),
    'slice': ('factor', 1.0, 3.0),
    'partial_volume': ('mix', 0.0, 0.5),
    'resample': ('factor', 1.0, 2.0),
    'kspace': ('keep', 1.0, 0.35),
}

_EPS = 1e-6

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _as5d(vol):
    """  [D,H,W] / [C,D,H,W] / [B,C,D,H,W]   [B,C,D,H,W] """
    if not torch.is_tensor(vol):
        raise TypeError('degrade3D   torch.Tensor  %r' % type(vol))
    if vol.dim() == 5:
        return vol, (lambda t: t)
    if vol.dim() == 4:
        return vol.unsqueeze(0), (lambda t: t.squeeze(0))
    if vol.dim() == 3:
        return vol.reshape(1, 1, *vol.shape), (lambda t: t.reshape(*vol.shape))
    raise ValueError('  3/4/5  [D,H,W] / [C,D,H,W] / [B,C,D,H,W]  %d  ' % vol.dim())

def _gen_ok(generator, device):
    """torch.Generator  CPU   generator   CUDA  """
    if generator is None:
        return False
    if generator.device != torch.device(device):
        raise ValueError('Generator device must match input device')
    return True

def _randn(shape, ref, generator=None):
    return torch.randn(shape, device=ref.device, dtype=torch.float32,
                       generator=generator if _gen_ok(generator, ref.device) else None)

def _uniform(shape, ref, generator=None):
    return torch.rand(shape, device=ref.device, dtype=torch.float32,
                      generator=generator if _gen_ok(generator, ref.device) else None)

def _pad_for_axis(axis, r):
    """5D   F.pad   pad   (W, H, D)   pad  """
    pad = [0, 0, 0, 0, 0, 0]
    pos = {2: 4, 3: 2, 4: 0}[axis]
    pad[pos] = r
    pad[pos + 1] = r
    return pad

def _blur_axis(x, sigma, axis, truncate=3.0):
    """  replicate  constant padding  """
    if sigma is None or sigma <= _EPS:
        return x
    r = max(1, int(round(truncate * float(sigma))))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / float(sigma)) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    shape = [1, 1, 1, 1, 1]
    shape[axis] = 2 * r + 1
    w = k.view(shape).expand(c, 1, *shape[2:]).contiguous()
    return F.conv3d(F.pad(x, _pad_for_axis(axis, r), mode='replicate'), w, groups=c)

def _blur_iso(x, sigma):
    if sigma is None or sigma <= _EPS:
        return x
    for ax in (2, 3, 4):
        x = _blur_axis(x, sigma, ax)
    return x

def _resample_axis(x, factor, axis):
    """ """
    n = x.shape[axis]
    m = int(round(n / float(factor)))
    m = max(2, min(int(n), m))
    if m == n:
        return x
    if factor > 1.0:
        x = _blur_axis(x, 0.5 * float(factor), axis)     #
    size = list(x.shape[2:])                             #
    size[axis - 2] = m
    y = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
    size[axis - 2] = n
    return F.interpolate(y, size=size, mode='trilinear', align_corners=False)

def _gradmag(x):
    """ """
    g = torch.zeros_like(x)
    for ax in (2, 3, 4):
        if x.shape[ax] < 3:
            continue
        d = (x.narrow(ax, 2, x.shape[ax] - 2) - x.narrow(ax, 0, x.shape[ax] - 2)).abs()
        sl = [slice(None)] * 5
        sl[ax] = slice(1, -1)
        g[tuple(sl)] += d
    return g

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _fam_noise(x, p, generator=None, axis=2, sign=1):
    sigma = p['sigma']
    if sigma <= _EPS:
        return x
    return x + sigma * _randn(x.shape, x, generator)

def _fam_rician(x, p, generator=None, axis=2, sign=1):
    sigma = p['sigma']
    if sigma <= _EPS:
        return x
    n1 = sigma * _randn(x.shape, x, generator)
    n2 = sigma * _randn(x.shape, x, generator)
    return torch.sqrt((x + n1) ** 2 + n2 ** 2)

def _fam_bias(x, p, generator=None, axis=2, sign=1):
    amp = p['amp']
    if amp <= _EPS:
        return x
    shp = list(x.shape[2:])
    low = _randn((x.shape[0], x.shape[1], 4, 4, 4), x, generator)
    field = F.interpolate(low, size=shp, mode='trilinear', align_corners=False)
    field = field - field.mean(dim=(2, 3, 4), keepdim=True)
    field = field / (field.abs().mean(dim=(2, 3, 4), keepdim=True) + _EPS)
    gain = 1.0 + amp * torch.tanh(field)          #
    return x * gain

def _fam_gamma(x, p, generator=None, axis=2, sign=1):
    g = p['gamma']
    if sign < 0:
        g = 1.0 / g                                #
    if abs(g - 1.0) <= _EPS:
        return x
    return torch.clamp(x, 0.0, 1.0) ** g

def _fam_blur(x, p, generator=None, axis=2, sign=1):
    return _blur_iso(x, p['sigma_vox'])

def _fam_slice(x, p, generator=None, axis=2, sign=1):
    """ /  axis   PSF """
    factor = p['factor']
    if factor <= 1.0 + _EPS:
        return x
    y = _blur_axis(x, 0.5 * float(factor), axis)
    return _resample_axis(y, factor, axis)

def _fam_partial_volume(x, p, generator=None, axis=2, sign=1):
    mix = p['mix']
    if mix <= _EPS:
        return x
    xb = _blur_iso(x, 1.2)
    g = _gradmag(xb)
    g = g / (g.amax(dim=(2, 3, 4), keepdim=True) + _EPS)
    local_mean = _blur_iso(xb, 2.4)
    a = (mix * g).clamp(0.0, 1.0)
    return x * (1.0 - a) + local_mean * a

def _fam_resample(x, p, generator=None, axis=2, sign=1):
    factor = p['factor']
    if factor <= 1.0 + _EPS:
        return x
    for ax in (2, 3, 4):
        x = _resample_axis(x, factor, ax)
    return x

def _fam_kspace(x, p, generator=None, axis=2, sign=1):
    keep = p['keep']
    if keep >= 1.0 - _EPS:
        return x
    dims = (2, 3, 4)
    spec = torch.fft.fftshift(torch.fft.fftn(x, dim=dims), dim=dims)
    mask = 1.0
    for i, n in enumerate(x.shape[2:]):
        k = max(1, int(round(keep * n / 2.0)))
        idx = torch.arange(n, device=x.device, dtype=torch.float32) - (n // 2)
        v = (idx.abs() <= float(k)).to(torch.float32)
        shape = [1, 1, 1, 1, 1]
        shape[2 + i] = n
        mask = mask * v.view(*shape)
    spec = spec * mask
    out = torch.fft.ifftn(torch.fft.ifftshift(spec, dim=dims), dim=dims).real
    return out

_FAMILY_FN = {
    'noise': _fam_noise,
    'rician': _fam_rician,
    'bias': _fam_bias,
    'gamma': _fam_gamma,
    'blur': _fam_blur,
    'slice': _fam_slice,
    'partial_volume': _fam_partial_volume,
    'resample': _fam_resample,
    'kspace': _fam_kspace,
}

def severity_to_params(family, severity, sign=1):
    """severity [0,1] ->   SEVERITY_RANGE """
    if family not in _FAMILY_FN:
        raise KeyError('  %r : %s' % (family, list(FAMILIES)))
    s = float(severity)
    if not 0.0 <= s <= 1.0:
        raise ValueError('severity must be finite and in [0,1]')
    name, lo, hi = SEVERITY_RANGE[family]
    p = {name: lo + (hi - lo) * s}
    if family == 'gamma':
        p['sign'] = int(1 if sign >= 0 else -1)
    return p

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def degrade(vol, family, severity, generator=None, axis=2, sign=1):
    """
     **  +  **  /  

      (vol_deg, info) 
        vol_deg :   dtype  [0,1]
        info    : dict(family, severity, params, axis)
    """
    if axis not in (2, 3, 4):
        raise ValueError('axis must be a spatial dimension of a 5D tensor: 2,3,4')
    if not vol.is_floating_point() or not torch.isfinite(vol).all():
        raise ValueError('degradation requires a finite floating-point image')
    x, restore = _as5d(vol)
    dtype = x.dtype
    x = x.to(torch.float32)
    params = severity_to_params(family, severity, sign=sign)
    with torch.no_grad():
        y = _FAMILY_FN[family](x, params, generator=generator, axis=axis, sign=sign)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        y = torch.clamp(y, 0.0, 1.0)
    return restore(y.to(dtype)), {'family': family, 'severity': float(severity),
                                  'params': params, 'axis': int(axis)}

def sample_degradation(vol, generator=None, families=None,
                       severity_range=(0.25, 0.75), axis=2):
    """
     ** **  +   DORA "a single degradation operation is
    stochastically sampled"  mild-to-moderate 

      (vol_deg, info) info   family/severity/params/axis  TensorBoard 
    """
    fams = list(FAMILIES) if families in (None, 'all', '', ()) else list(families)
    bad = [f for f in fams if f not in _FAMILY_FN]
    if bad:
        raise KeyError('  %s : %s' % (bad, list(FAMILIES)))
    if not fams:
        raise ValueError('families  ')
    x, _restore = _as5d(vol)
    lo, hi = float(severity_range[0]), float(severity_range[1])
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError('severity_range   0<=lo<=hi<=1  %r' % (severity_range,))
    u = _uniform((2,), x, generator)
    idx = min(int(float(u[0]) * len(fams)), len(fams) - 1)
    sev = lo + (hi - lo) * float(u[1])
    family = fams[idx]
    sign = 1 if float(_uniform((1,), x, generator)[0]) > 0.5 else -1
    vol_deg, info = degrade(vol, family, sev, generator=generator, axis=axis, sign=sign)
    info['index'] = idx
    return vol_deg, info

def family_table():
    """  /  """
    lines = [' %d  severity 0 1   mild-to-moderate ' % len(FAMILIES)]
    for f in FAMILIES:
        name, lo, hi = SEVERITY_RANGE[f]
        lines.append('  %-15s %-10s %.3g   %.3g   %s' % (f, name, lo, hi, FAMILY_DOC[f]))
    return '\n'.join(lines)

__all__ = ['FAMILIES', 'FAMILY_DOC', 'SEVERITY_RANGE', 'degrade', 'sample_degradation',
           'severity_to_params', 'family_table']
