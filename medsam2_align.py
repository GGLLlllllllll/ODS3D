# -*- coding: utf-8 -*-
"""
  D  MedSAM2  

 
----
`Architecture_hyper`   3D  
  NCC  atlas   flow   Dice one-shot 
MedSAM2 SAM2.1 Hiera-Tiny ** ** 

 
----
  MedSAM2  
"  /  " ** ** 

    L = L_task + lambda * mean_{level, x, y} [ 1 - cos( proj_l(native_l), detach(sam_k) ) ]

* `proj_l`   1x1x1   native   l   MedSAM2   256  
* MedSAM2  `no_grad`  `detach` 

 
------------
  128^3 MedSAM2   2D   `res` 
MedSAM2   FPN   k   `res/4/2^k`  j   `128/2^j` 
  `res=512`   `j = k`  `res=256`   `j = k + 1` ** ** 
 

 native   j   d   [d*2^j, (d+1)*2^j) 
  `z = d*2^j + 2^(j-1)`   MedSAM2 
** ** " " 

 
----
* `models_hyper.py` / `layers3D.py`  checkpoint  `model.state_dict()`
    8.54M  `test_and_save_results.py`  
* MedSAM2     ** /  0  **
*  
"""

import json
import hashlib
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CFG = 'configs/sam2.1_hiera_t512.yaml'
DEFAULT_CKPT = os.path.join(_HERE, 'MedSAM2_latest.pt')

def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def build_medsam2_image_encoder(ckpt_path: str = DEFAULT_CKPT,
                                cfg_name: str = DEFAULT_CFG,
                                scalp: int = 0,
                                device='cuda',
                                overrides: Optional[Sequence[str]] = None):
    """  MedSAM2 image encoder  image_encoder.*  

    scalp=0 ->   4   FPN (1/4, 1/8, 1/16, 1/32)
    scalp=1 ->   3  
    """
    import sam2  #
    from hydra import compose
    from hydra.utils import instantiate

    ov = ['model.image_encoder.scalp=%d' % int(scalp)]
    if overrides:
        ov = list(overrides) + ov
    cfg = compose(config_name=cfg_name, overrides=ov)
    encoder = instantiate(cfg.model.image_encoder, _recursive_=True)

    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    enc_sd = {k[len('image_encoder.'):]: v for k, v in sd.items() if k.startswith('image_encoder.')}
    if not enc_sd:
        raise RuntimeError('checkpoint %s   image_encoder.*  ' % ckpt_path)
    encoder.load_state_dict(enc_sd, strict=True)   #

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder

class MedSAM2Aligner(nn.Module):
    """  MedSAM2   +  

    forward(vols, native_feats) -> (align_loss, stats)
        vols:         [ [1,1,X,Y,Z], ... ]         /  [0,1] 
        native_feats: [ (x1,x2,x3,x4), ... ]        4  
    """

    def __init__(self,
                 ckpt_path: str = DEFAULT_CKPT,
                 cfg_name: str = DEFAULT_CFG,
                 native_channels: Sequence[int] = (32, 64, 128, 256),
                 res: int = 256,
                 slice_batch: int = 0,
                 axis: int = 2,
                 lambda_: float = 0.05,
                 mask_background: bool = True,
                 sam_levels: Optional[Sequence[int]] = None,
                 volume_names: Sequence[str] = ('src', 'tgt'),
                 gate_mode: str = 'none',
                 gate_ckpt: Optional[str] = None,
                 gate_cfg: str = 'configs/sam2.1_hiera_l512.yaml',
                 gate_coefs: Optional[str] = None,
                 gate_quantile: float = -1.0,
                 gate_above_mean: bool = False,
                 gate_slice_batch: int = 0,
                 device='cuda'):
        super(MedSAM2Aligner, self).__init__()

        assert res in (128, 256, 512), 'res   128/256/512 MedSAM2   patch stride   4 '
        self.res = int(res)
        if slice_batch in (0, None):
            slice_batch = {512: 1, 256: 16, 128: 32}[self.res]
        self.slice_batch = int(slice_batch)
        if self.slice_batch < 1:
            raise ValueError('slice_batch must be positive (or 0 for automatic selection)')
        self.axis = int(axis)
        if self.axis not in (0, 1, 2):
            raise ValueError('axis must be 0, 1 or 2 (spatial axes)')
        if gate_mode in ('online', 'cached') and self.res != 256:
            raise ValueError('Gate coefficients/cache currently require res=256; disable gating for other resolutions')
        self.lambda_ = float(lambda_)
        self.mask_background = bool(mask_background)
        self.volume_names = list(volume_names)
        self.native_channels = tuple(native_channels)

        self.offset = int(round(math.log2(512.0 / self.res)))

        if sam_levels is None:
            sam_levels = [k for k in range(4) if (k + self.offset) < len(self.native_channels)]
        self.sam_levels = tuple(int(k) for k in sam_levels)
        self.native_levels = tuple(k + self.offset for k in self.sam_levels)
        if not self.sam_levels:
            raise ValueError('res=%d   native_channels=%s  '
                             % (self.res, self.native_channels))

        scalp = 0 if max(self.sam_levels) >= 3 else 1
        self.encoder = build_medsam2_image_encoder(ckpt_path, cfg_name, scalp, device)
        self.scalp = scalp

        self.proj = nn.ModuleList([
            nn.Conv3d(self.native_channels[j], 256, kernel_size=1)
            for j in self.native_levels])

        self.gate_channels = {1: 0, 2: 1, 3: 2}

        self.gate_mode = gate_mode
        self.gate_quantile = float(gate_quantile)
        self.gate_above_mean = bool(gate_above_mean)
        self.encoder2 = None
        self.coefs = None
        if gate_mode == 'online':
            assert gate_ckpt is not None and gate_coefs is not None,\
                'online   gate_ckpt   gate_coefs'
            self.encoder2 = build_medsam2_image_encoder(gate_ckpt, gate_cfg, scalp=scalp, device=device)
            blob = torch.load(gate_coefs, map_location='cpu', weights_only=False)
            if blob.get('res') != self.res or blob.get('axis') != self.axis:
                raise ValueError('Online gate coefficients have incompatible res/axis')
            expected = dict(target=[cfg_name, checkpoint_sha256(ckpt_path)],
                            teacher=[gate_cfg, checkpoint_sha256(gate_ckpt)])
            if blob.get('online_provenance') != expected:
                raise ValueError('Online coefficients lack matching checkpoint/config provenance; rebuild cache')
            self.coefs = blob['coefs']                      # {level_str: {W, mu_x, sd_x, mu_y, sd_y, dim}}
            self.gate_slice_batch = int(gate_slice_batch) or self.slice_batch
            rule = ('above-mean' if self.gate_above_mean else
                    ('soft/ ' if self.gate_quantile <= 0 else 'q=%.2f' % self.gate_quantile))
            self.gate_info = 'online(sam2=%s, %s)' % (os.path.basename(gate_ckpt), rule)
        else:
            self.gate_slice_batch = 0
            self.gate_info = gate_mode

        self.register_buffer('img_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._device = torch.device(device)

    # ------------------------------------------------------------------ utils
    def describe(self) -> str:
        pairs = ', '.join('sam_l%d<->native_l%d(%dch)' % (k, j, self.native_channels[j])
                          for k, j in zip(self.sam_levels, self.native_levels))
        return ('MedSAM2 aligner: ckpt res=%d scalp=%d slice_batch=%d axis=%d lambda=%.3f '
                'mask_bg=%s gate=%s | %s' % (self.res, self.scalp, self.slice_batch, self.axis,
                                             self.lambda_, self.mask_background, self.gate_info, pairs))

    def trainable_parameters(self):
        """  requires_grad=False   grad=None """
        return list(self.proj.parameters())

    def train(self, mode: bool = True):
        """  eval  DropPath/Dropout  """
        super(MedSAM2Aligner, self).train(mode)
        self.encoder.eval()
        if self.encoder2 is not None:      #
            self.encoder2.eval()
        return self

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.proj.state_dict()

    @torch.no_grad()
    def _sam_slices(self, vol: torch.Tensor, level: int, zs: Sequence[int],
                    enc: Optional[nn.Module] = None, slice_batch: Optional[int] = None) -> torch.Tensor:
        """  [len(zs), 256, h, w]  detach """
        enc = self.encoder if enc is None else enc
        sbm = self.slice_batch if slice_batch is None else int(slice_batch)
        v = vol[0, 0].movedim(self.axis, 0)                       # [Z, X, Y]
        z_idx = torch.as_tensor(list(zs), dtype=torch.long, device=v.device)
        sl = v.index_select(0, z_idx).unsqueeze(1)                # [n, 1, X, Y]
        sl = F.interpolate(sl, size=(self.res, self.res), mode='bilinear', align_corners=False)
        sl = sl.repeat(1, 3, 1, 1)                                #
        sl = (sl - self.img_mean) / self.img_std

        outs = []
        for i in range(0, sl.shape[0], sbm):
            outs.append(enc(sl[i:i + sbm])['backbone_fpn'][level])
        return torch.cat(outs, 0)

    @torch.no_grad()
    def _online_intersection(self, vol: torch.Tensor, j: int, step: int,
                             sam_med: torch.Tensor, zs: Sequence[int]) -> Optional[torch.Tensor]:
        """ 

        a_i = 1 - ||Ys - W Xs||^2 / dim  Xs=MedSAM2  Ys= W  

        gate_quantile <= 0 :  w = max(a, 0)        
        gate_above_mean         :  w = 1[a >= mean(a)]    
        gate_quantile > 0       :  w = 1[a >=  ]     
        """
        k = j - self.offset
        if self.encoder2 is None or self.coefs is None or str(j) not in self.coefs:
            return None
        sam2 = self._sam_slices(vol, k, zs, enc=self.encoder2, slice_batch=self.gate_slice_batch)
        coef = self.coefs[str(j)]
        dev = sam_med.device
        X = sam_med.permute(0, 2, 3, 1).reshape(-1, sam_med.shape[1])
        Y = sam2.permute(0, 2, 3, 1).reshape(-1, sam2.shape[1])
        Xs = (X - coef['mu_x'].to(dev).float()) / coef['sd_x'].to(dev).float()
        Ys = (Y - coef['mu_y'].to(dev).float()) / coef['sd_y'].to(dev).float()
        pred = Xs @ coef['W'].to(dev).float()
        a = 1.0 - (Ys - pred).pow(2).sum(1) / float(coef['dim'])
        h, w = sam_med.shape[-2], sam_med.shape[-1]
        a = a.reshape(-1, h, w)                                   # [dn, h, w]

        m = (vol > 1e-3).float()
        m = F.max_pool3d(m, kernel_size=step, stride=step)[0, 0].movedim(self.axis, 0)   # [dn,h,w]
        inside = m > 0
        if int(inside.sum()) < 16:
            return None

        if self.gate_above_mean:
            thr = a[inside].mean()
            return ((a >= thr) & inside).float()
        if self.gate_quantile is None or self.gate_quantile <= 0:
            return a.clamp(min=0.0) * inside.float()
        thr = torch.quantile(a[inside], 1.0 - self.gate_quantile)
        return ((a >= thr) & inside).float()

    def _pool_gate(self, gate: torch.Tensor, j: int, step: int) -> Optional[torch.Tensor]:
        """ [1,C,128,128,128]  native   L1/L2/L3   j  

          [dn, h, w]   None 
        """
        ch = self.gate_channels.get(j)
        if ch is None or ch >= gate.shape[1]:
            return None
        g = gate[:, ch:ch + 1]                                    # [1,1,128,128,128]
        g = F.max_pool3d(g, kernel_size=step, stride=step)         # [1,1,X',Y',Z']
        return g[0, 0].movedim(self.axis, 0)                            # [dn, h, w]

    def _align_level(self, proj: nn.Module, vol: torch.Tensor, native_feat: torch.Tensor,
                     k: int, j: int, gate: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        depth = int(vol.shape[self.axis + 2])                     # vol: [1,1,X,Y,Z]
        step = 2 ** j                                             #
        dn = depth // step
        if dn < 1:
            return None
        zs = [min(step * d + step // 2, depth - 1) for d in range(dn)]

        sam = self._sam_slices(vol, k, zs)                        # [dn, 256, h, w] (no grad)
        nat = native_feat[0].movedim(self.axis + 1, 0)                  # [dn, C, h, w]
        if sam.shape[-2:] != nat.shape[-2:] or sam.shape[0] != nat.shape[0]:
            raise ValueError('  sam=%s native=%s  128^3 '
                             '  medsam2_res   native  '
                             % (tuple(sam.shape), tuple(nat.shape)))
        pred = proj(nat.unsqueeze(2)).squeeze(2)                  # [dn, 256, h, w]
        loss = 1.0 - F.cosine_similarity(pred, sam, dim=1, eps=1e-6)   # [dn, h, w]

        w = self._pool_gate(gate, j, step) if gate is not None else None
        if w is None and self.gate_mode == 'online':
            w = self._online_intersection(vol, j, step, sam, zs)
        if w is None and self.mask_background:
            m = (vol > 1e-3).float()
            m = F.max_pool3d(m, kernel_size=step, stride=step)    # [1,1,X',Y',Z']
            w = m[0, 0].movedim(self.axis, 0)                          # [dn, h, w]

        if w is None:
            return loss.mean()
        denom = w.sum()
        if float(denom) < 1.0:                                    #
            return None
        return (loss * w).sum() / denom

    # --------------------------------------------------------------- forward
    def forward(self, vols: Sequence[torch.Tensor],
                native_feats: Sequence[Sequence[torch.Tensor]],
                gates: Optional[Sequence[torch.Tensor]] = None):
        """gates:   [1,3,128,128,128]  0/1/2 = native L1/L2/L3 

          build_teacher_gate.py  
          3D  
        """
        if len(vols) != len(native_feats) or (gates is not None and len(gates) != len(vols)):
            raise ValueError('volumes, feature lists and gates must have equal lengths')
        if any(v.ndim != 5 or v.shape[:2] != (1, 1) or v.shape[2:] != (128, 128, 128) for v in vols):
            raise ValueError('MedSAM2Aligner currently requires individual [1,1,128,128,128] volumes')
        if self.gate_mode == 'online' and gates is not None:
            raise ValueError('Online gating cannot be overridden by cached gates')
        total = None
        n = 0
        stats = {}
        for i, (vol, feats) in enumerate(zip(vols, native_feats)):
            name = self.volume_names[i] if i < len(self.volume_names) else 'vol%d' % i
            gate = None if gates is None else gates[i]
            for pi, (k, j) in enumerate(zip(self.sam_levels, self.native_levels)):
                l = self._align_level(self.proj[pi], vol, feats[j], k, j, gate)
                if l is None:
                    continue
                stats['align/%s_samL%d_natL%d' % (name, k, j)] = float(l.detach())
                total = l if total is None else total + l
                n += 1
        if total is None:
            # A valid gate can reject every voxel; skip its loss without inventing
            # positive weights or falling back to ungated supervision.
            total = sum(feats[j].sum() * 0.0 for feats in native_feats for j in self.native_levels)
            stats['align/valid_layers'] = 0
            return total, stats
        return total / n, stats

class GateLoader(object):
    """  build_teacher_gate.py  " " 

      .npy 
        [3, 128, 128, 128] uint8  0/1/2 = native L1/L2/L3  0..255 0= / 
    meta gate_meta.json uint8   CCA  

    __call__(case_name) -> [1,3,128,128,128] float  hard/soft  
     / 
    """

    def __init__(self, gate_dir: str, meta_path: Optional[str] = None,
                 quantile: float = -1.0, mode: str = 'hard', above_mean: bool = False,
                 device='cuda', cache_size: int = 24):
        self.dir = gate_dir
        meta_path = meta_path or os.path.join(gate_dir, 'gate_meta.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.above_mean = bool(above_mean)
        self.soft = (not self.above_mean) and ((mode == 'soft') or (quantile is None) or (float(quantile) <= 0))
        if self.soft:
            self.thresholds = {}
        elif self.above_mean:
            self.thresholds = self.meta.get('mean_uint8')
            if not self.thresholds:
                raise KeyError('gate_meta.json   mean_uint8  build_teacher_gate.py  ')
        else:
            key = '%.2f' % float(quantile)
            if key not in self.meta['thresholds']:
                raise KeyError('gate_meta.json   quantile=%s  : %s'
                               % (key, sorted(self.meta['thresholds'].keys())))
            self.thresholds = self.meta['thresholds'][key]      # {"1": v, "2": v, "3": v} (0..255)
        self.mode = mode
        self.device = device
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.missing = set()

    def _load_raw(self, case_name: str) -> Optional[np.ndarray]:
        if case_name in self._cache:
            self._cache.move_to_end(case_name)
            return self._cache[case_name]
        path = os.path.join(self.dir, case_name + '.npy')
        if not os.path.exists(path):
            self.missing.add(case_name)
            raise FileNotFoundError('Required gate map missing: %s' % path)
        arr = np.load(path)                                  # [3,128,128,128] uint8
        if arr.shape != (3, 128, 128, 128) or arr.dtype != np.uint8:
            raise ValueError('Invalid gate map shape/dtype: %s' % path)
        self._cache[case_name] = arr
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr

    def __call__(self, case_name: str) -> Optional[torch.Tensor]:
        arr = self._load_raw(case_name)
        if arr is None:
            return None
        raw = torch.from_numpy(np.ascontiguousarray(arr)).float()
        # Builder encodes round(w*254)+1, reserving zero for outside the brain.
        g = ((raw - 1.0) / 254.0).clamp(0.0, 1.0)
        if not self.soft:
            out = torch.zeros_like(g)
            for j, ch in ((1, 0), (2, 1), (3, 2)):
                thr = float(self.thresholds[str(j)])
                out[ch] = ((raw[ch] > 0) & (raw[ch] >= thr)).float()
            g = out
        return g.unsqueeze(0).to(self.device)

    def stats(self, case_name: str) -> Dict[str, float]:
        arr = self._load_raw(case_name)
        if arr is None:
            return {}
        if self.soft:
            return {}
        out = {}
        for ch, j in ((0, 1), (1, 2), (2, 3)):
            v = arr[ch]
            brain = v > 0
            thr = float(self.thresholds[str(j)])
            out['gate/L%d_sel_frac' % j] = float(((v > 0) & (v >= thr)).sum()) / max(int(brain.sum()), 1)
        return out
