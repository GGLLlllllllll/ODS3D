# -*- coding: utf-8 -*-
"""
 Degradation-Aware Training, DAT  DORA  "  +  "
  3D   MR  - 
============================================================================

  DORA  " " 
  (1)   mild-to-moderate   degrade3D.py 
  (2)   +  ** / ** 
       ** ** 
  (3)   +   13  18 

  (2)(3)  ** **  
`models_hyper.py` / `layers3D.py`  `model.state_dict()`  
156 keys `test_and_save_results.py`  

 

  A.   L_deg  4  " " 
         y_deg  ** ** + ** ** 
         atlas   Dice " " 
         loss  _deg=0.5   Dice  

  B.   /   L_sc  18  
       DORA: J(x, x ) = (1-x)^    log(1+exp(x - x )) x  x   
        x  =  " " detach 
             x  =  " " 
         (1-x)^   ** ** 
          " "    2 

  C.   L_struct  13   Alignment Loss  
       DORA: log(1+exp( max_{k (j)} sim(V_j,E_k) - sim(V_j,E_{ (j)}) +   ))
        V =  E_k =   k  ** **
              L2  =0.15 
        " " 
              " "     " " 
        cyclic 

 / "  +  " 
  `--degrade_every_k`  
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import degrade3D
from layers3D import SpatialTransformer

DEGRADE_DEFAULTS = dict(
    families='all',        #
    family_scope='iteration',  #
    severity_min=0.25,     #
    severity_max=0.75,     #
    axis=2,                #
    every_k=1,             #
    warmup_epochs=2,       #
    lambda_deg=0.5,        #
    gt_mode='hard',        #
    lambda_cons=0.1,       #
    cons_symmetric=False,  #
    lambda_sc=0.5,         #
    sc_gamma=2.0,          #
    lambda_struct=0.1,     #
    struct_margin=0.15,    #
    struct_samples=256,    #
    struct_cyclic=True,    #
    class_weight_file='',  #
)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _masked_mean(x, mask):
    """  mask  mask  **  0**  nan """
    m = mask.to(x.dtype)
    den = m.sum()
    if float(den) < 1.0:
        return x.sum() * 0.0, False
    return (x * m).sum() / den, True

def parse_families(s):
    """'all' / '' / None ->  'noise,blur' -> ['noise','blur'] """
    if s in (None, '', 'all'):
        return list(degrade3D.FAMILIES)
    fams = [t.strip() for t in str(s).split(',') if t.strip()]
    bad = [f for f in fams if f not in degrade3D.FAMILIES]
    if bad:
        raise KeyError('  %s : %s' % (bad, list(degrade3D.FAMILIES)))
    return fams

def load_class_weight(path, num_classes=14, device=None):
    """
      = num_classes  0   1.0 
     ** ** 
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(' : %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and 'label_values' in obj and num_classes == 14:
        expected = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55]
        if obj['label_values'] != expected:
            raise ValueError('Class-weight label order does not match the model')
    w = obj if isinstance(obj, list) else obj.get('weights') if isinstance(obj, dict) else None
    if w is None:
        raise KeyError('  "weights"  : %s' % path)
    w = [float(v) for v in w]
    if len(w) != num_classes:
        raise ValueError('  %d != num_classes %d %s ' % (len(w), num_classes, path))
    if not all(v == v and abs(v) != float('inf') and v > 0 for v in w):
        raise ValueError(' : %s' % w)
    t = torch.tensor(w, dtype=torch.float32, device=device)
    return t

def hard_multiclass_dice(prob, gt_idx, num_classes=14, class_weight=None):
    """
      losses3D.Dice   2*|A B|/(|A|+|B|)  
    absent   0 ** **  14   one-hot 
      -dice  Dice  
    """
    vals = []
    for c in range(num_classes):
        g = (gt_idx == c).to(prob.dtype)
        p = prob[:, c:c + 1]
        dims = tuple(range(2, p.ndim))
        top = 2.0 * (g * p).sum(dim=dims)
        bottom = torch.clamp(g.sum(dim=dims) + p.sum(dim=dims), min=1e-5)
        vals.append(top / bottom)
    d = torch.cat(vals, dim=1)
    if class_weight is not None:
        w = class_weight.to(d.dtype)
        d = ((d * w).sum(dim=1) / torch.clamp(w.sum(), min=1e-6)).mean()
    else:
        d = d.mean()
    return -d

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class DegradationScheduler:
    def __init__(self, families='all', severity_range=(0.25, 0.75), axis=2, seed=0):
        self.families = parse_families(families) if not isinstance(families, (list, tuple))\
            else list(families)
        self.severity_range = (float(severity_range[0]), float(severity_range[1]))
        if not (0.0 <= self.severity_range[0] <= self.severity_range[1] <= 1.0):
            raise ValueError('severity_range   0<=lo<=hi<=1')
        self.axis = int(axis)
        if self.axis not in (2, 3, 4) or not self.families:
            raise ValueError('Invalid degradation axis or empty family list')
        self.seed = int(seed)
        self.counts = {f: 0 for f in self.families}
        self.n_calls = 0

    def family_for_epoch(self, epoch):
        """  epoch   N   N   RNG """
        cycle, offset = divmod(int(epoch), len(self.families))
        gen = torch.Generator(device='cpu')
        gen.manual_seed((self.seed * 100003 + cycle * 104729 + 17) % (2 ** 31 - 1))
        idx = int(torch.randperm(len(self.families), generator=gen)[offset].item())
        return self.families[idx]

    def __call__(self, vol, step, family=None):
        if family is not None and family not in self.families:
            raise KeyError('  %r   %s  ' % (family, self.families))
        gen = torch.Generator(device=vol.device)
        gen.manual_seed((self.seed * 100003 + int(step) * 7919) % (2 ** 31 - 1))
        out, info = degrade3D.sample_degradation(
            vol, generator=gen, families=[family] if family is not None else self.families,
            severity_range=self.severity_range, axis=self.axis)
        self.n_calls += 1
        self.counts[info['family']] = self.counts.get(info['family'], 0) + 1
        return out, info

    def describe(self):
        return ' %d   /   [%.2f, %.2f] /   %d' % (
            len(self.families), self.severity_range[0], self.severity_range[1], self.axis)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class SegFeatureTap:
    """
     `model.Seg_Decoder.seg_head`  
    `model.Seg_Head.seg_head`  forward_pre_hook 
      take()  
     
        tap = SegFeatureTap(model)
        tap.begin('clean');    y_logits     = model.Seg_Decoder(encoded_y)
        tap.begin('degraded'); y_logits_deg = model.Seg_Decoder(encoded_y_deg)
        f_clean, f_deg = tap.take('clean'), tap.take('degraded')
    """

    def __init__(self, model):
        if hasattr(model, 'Seg_Decoder') and hasattr(model.Seg_Decoder, 'seg_head'):
            seg_head = model.Seg_Decoder.seg_head
        elif hasattr(model, 'Seg_Head') and hasattr(model.Seg_Head, 'seg_head'):
            seg_head = model.Seg_Head.seg_head
        else:
            raise AttributeError(
                'model   Seg_Decoder.seg_head   Seg_Head.seg_head ')
        self._store = {}
        self._cur = None
        self._handle = seg_head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if self._cur is not None and len(inputs) > 0:
            self._store[self._cur] = inputs[0]

    def begin(self, tag):
        self._cur = tag

    def take(self, tag):
        self._cur = None
        return self._store.pop(tag, None)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._store.clear()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def structure_margin_loss(feat_query, feat_key, gt_idx, mask=None,
                          margin=0.15, samples=256, cyclic=True, num_classes=14):
    """
    feat_query / feat_key : [B,C,D,H,W]  query   key  
    gt_idx                : [B,1,D,H,W]   atlas  
    mask                  : [B,1,D,H,W] bool     
      (loss, stats)  2   0 
    """
    if samples < 1:
        raise ValueError('samples must be positive')
    if feat_query.shape != feat_key.shape or gt_idx.shape != (feat_query.shape[0], 1, *feat_query.shape[2:]):
        raise ValueError('feature / label shapes do not match')
    # Each subject has its own labels and prototypes. Never reuse subject 0 labels.
    if feat_query.shape[0] > 1:
        results = [structure_margin_loss(feat_query[i:i+1], feat_key[i:i+1],
                   gt_idx[i:i+1], None if mask is None else mask[i:i+1],
                   margin, samples, cyclic, num_classes) for i in range(feat_query.shape[0])]
        valid = [(loss, stats) for loss, stats in results if stats['struct_pairs'] > 0]
        if not valid:
            return (feat_query.sum() + feat_key.sum()) * 0, {'struct_classes': 0, 'struct_pairs': 0}
        return torch.stack([v[0] for v in valid]).mean(), {
            'struct_classes': sum(v[1]['struct_classes'] for v in valid),
            'struct_pairs': sum(v[1]['struct_pairs'] for v in valid),
            'struct_gap': sum(v[1]['struct_gap'] for v in valid) / len(valid)}
    b, c = feat_query.shape[0], feat_query.shape[1]
    qf = feat_query.reshape(b, c, -1)
    kf = feat_key.reshape(b, c, -1)
    gt = gt_idx.reshape(b, -1).long()
    if mask is not None:
        gt = torch.where(mask.reshape(b, -1), gt, torch.zeros_like(gt))

    pos = (gt[0] > 0).nonzero(as_tuple=False).squeeze(1)
    if pos.numel() == 0:
        return (feat_query.sum() + feat_key.sum()) * 0.0, {'struct_classes': 0, 'struct_pairs': 0}
    vals = gt[0][pos]
    idx_list, lab_list = [], []
    for cls in torch.unique(vals).tolist():
        if cls < 1 or cls >= num_classes:
            continue
        sub = pos[vals == cls]
        if sub.numel() > samples:
            sub = sub[torch.randperm(sub.numel(), device=sub.device)[:samples]]
        idx_list.append(sub)
        lab_list.append(torch.full_like(sub, int(cls)))
    if len(idx_list) < 2:      #
        return (feat_query.sum() + feat_key.sum()) * 0.0, {'struct_classes': len(idx_list),
                                                           'struct_pairs': 0}

    idx = torch.cat(idx_list)
    lab = torch.cat(lab_list)
    present = torch.unique(lab, sorted=True)
    if present.numel() < 2:
        return (feat_query.sum() + feat_key.sum()) * 0.0, {'struct_classes': int(present.numel()),
                                                           'struct_pairs': 0}

    qn = F.normalize(qf[:, :, idx], dim=1)          # [B,C,N]
    kn = F.normalize(kf[:, :, idx], dim=1)

    protos = []
    for cls in present.tolist():
        sel = (lab == cls)
        protos.append(F.normalize(kn[:, :, sel].mean(dim=2), dim=1))
    proto = torch.stack(protos, dim=2)              # [B,C,K]

    own = torch.searchsorted(present, lab)           #
    sim = torch.einsum('bcn,bck->bnk', qn, proto)    # [B,N,K]
    onehot = F.one_hot(own, present.numel()).bool().unsqueeze(0)
    other = sim.masked_fill(onehot, -1e4)
    best_other = other.max(dim=2).values
    own_sim = sim.gather(2, own.view(1, -1, 1)).squeeze(2)
    loss = F.softplus(best_other - own_sim + float(margin)).mean()

    if cyclic:
        qn2, kn2 = kn, qn
        protos2 = []
        for cls in present.tolist():
            sel = (lab == cls)
            protos2.append(F.normalize(kn2[:, :, sel].mean(dim=2), dim=1))
        proto2 = torch.stack(protos2, dim=2)
        sim2 = torch.einsum('bcn,bck->bnk', qn2, proto2)
        other2 = sim2.masked_fill(onehot, -1e4)
        best_other2 = other2.max(dim=2).values
        own_sim2 = sim2.gather(2, own.view(1, -1, 1)).squeeze(2)
        loss = 0.5 * (loss + F.softplus(best_other2 - own_sim2 + float(margin)).mean())

    stats = {'struct_classes': int(present.numel()), 'struct_pairs': int(idx.numel()),
             'struct_gap': float((own_sim - best_other).mean())}
    return loss, stats

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class DegradationAwareAux(nn.Module):
    """
     
      train.py  -> describe() ->   sample()/begin()/compute() 
    """

    def __init__(self, cfg=None, num_classes=14, device=None, seed=0, axis=None):
        super(DegradationAwareAux, self).__init__()
        self.cfg = dict(DEGRADE_DEFAULTS)
        if cfg:
            unknown = [k for k in cfg if k not in DEGRADE_DEFAULTS]
            if unknown:
                raise KeyError(' : %s' % unknown)
            self.cfg.update(cfg)
        for key in ('lambda_deg', 'lambda_cons', 'lambda_sc', 'lambda_struct', 'sc_gamma'):
            value = float(self.cfg[key])
            if not 0 <= value < float('inf'):
                raise ValueError('%s must be finite and nonnegative' % key)
        if int(self.cfg['every_k']) < 1 or int(self.cfg['struct_samples']) < 1 or int(self.cfg['warmup_epochs']) < 0:
            raise ValueError('Invalid degradation schedule/sample count')
        if self.cfg['gt_mode'] not in ('hard', 'soft'):
            raise ValueError('gt_mode must be hard or soft')
        if self.cfg['family_scope'] not in ('epoch', 'iteration'):
            raise ValueError('family_scope must be epoch or iteration')
        if axis is None:
            axis = self.cfg['axis']
        self.num_classes = int(num_classes)
        self.device = device
        self.sched = DegradationScheduler(families=self.cfg['families'],
                                          severity_range=(self.cfg['severity_min'],
                                                          self.cfg['severity_max']),
                                          axis=axis, seed=seed)
        self.st_soft = SpatialTransformer(mode='bilinear')
        self.st_hard = SpatialTransformer(mode='nearest')
        self.tap = None
        self.class_weight = None
        self.last_info = None

    def attach(self, model):
        """  C   hook _struct<=0  """
        if self.cfg['lambda_struct'] > 0 and self.tap is None:
            self.tap = SegFeatureTap(model)
        return self

    def close(self):
        if self.tap is not None:
            self.tap.close()
            self.tap = None

    def begin(self, tag):
        if self.tap is not None:
            self.tap.begin(tag)

    def take(self, tag):
        return self.tap.take(tag) if self.tap is not None else None

    def active(self, epoch):
        """warmup  DORA  """
        return int(epoch) >= int(self.cfg['warmup_epochs'])

    def due(self, step):
        k = max(int(self.cfg['every_k']), 1)
        return (int(step) % k) == 0

    def describe(self):
        c = self.cfg
        lines = [
            ' DAT ',
            '  %s %s' % (self.sched.describe(),
                         '  epoch  '
                         if c['family_scope'] == 'epoch' else ' '),
            '   : epoch >= %d   %d  ' % (c['warmup_epochs'], max(int(c['every_k']), 1)),
            '  A   Dice       =%.3f   dice_seg  flow   detach ' % c['lambda_deg'],
            '  B   KL       =%.3f  %s ' % (c['lambda_cons'],
                                                     ' ' if c['cons_symmetric'] else ' '),
            '  C        =%.3f   =%.2f  18  (1-x)^   '
            % (c['lambda_sc'], c['sc_gamma']),
            '  D        =%.3f   =%.2f    %d    %s 13  '
            % (c['lambda_struct'], c['struct_margin'], c['struct_samples'],
               ' ' if c['struct_cyclic'] else ' '),
        ]
        if self.class_weight is not None:
            lines.append('   : %s' % (c['class_weight_file']))
        else:
            lines.append('   :  ')
        return '\n'.join(lines)

    def set_class_weight(self, weight):
        self.class_weight = weight
        return self

    def epoch_family(self, epoch):
        """epoch  iteration   None """
        if self.cfg['family_scope'] == 'epoch':
            active_epoch = int(epoch) - int(self.cfg['warmup_epochs'])
            return self.sched.family_for_epoch(max(active_epoch, 0))
        return None

    def sample(self, y, step, family=None):
        y_deg, info = self.sched(y, step, family=family)
        self.last_info = info
        return y_deg, info

    def compute(self, y_logits_clean, y_logits_deg, x_seg, flow, y_clean,
                seg_dice_fn=None, n_class=None):
        """
          (loss, stats) loss     loss  
        seg_dice_fn:   reg_criterion.seg_dice  LossFunction_dice 
                     A   Dice  
        """
        n_class = self.num_classes if n_class is None else int(n_class)
        c = self.cfg
        out = {}

        flow_d = flow.detach()      #
        with torch.no_grad():
            gt_idx = self.st_hard(x_seg.float(), flow_d).round().long()
            brain = y_clean[:, :1] > 1e-3
            fg = (gt_idx > 0) & brain
        gt_idx = gt_idx.to(y_logits_clean.dtype)
        fg_any = bool(fg.any())

        need_prob = (c['lambda_cons'] > 0) or (c['lambda_sc'] > 0 and fg_any)
        if need_prob:
            logp = F.log_softmax(y_logits_clean, dim=1)
            logq = F.log_softmax(y_logits_deg, dim=1)
            p, q = logp.exp(), logq.exp()

        loss_deg = None
        if c['lambda_deg'] > 0:
            if c['gt_mode'] == 'soft' and seg_dice_fn is None:
                raise ValueError('soft gt_mode requires seg_dice_fn')
            if c['gt_mode'] == 'soft':
                loss_deg = seg_dice_fn(F.softmax(y_logits_deg, dim=1).float(),
                                       x_seg, flow_d, n_class,
                                       class_weight=self.class_weight)
            else:
                p_deg = F.softmax(y_logits_deg, dim=1)
                loss_deg = hard_multiclass_dice(p_deg, gt_idx, n_class, self.class_weight)
            out['deg_dice_loss'] = float(loss_deg)
            out['deg_dice'] = -float(loss_deg)

        loss_cons = None
        if c['lambda_cons'] > 0:
            kl = (p.detach() * (logp.detach() - logq)).sum(1, keepdim=True)
            if c['cons_symmetric']:
                kl = 0.5 * (kl + (q.detach() * (logq.detach() - logp)).sum(1, keepdim=True))
            loss_cons, ok = _masked_mean(kl, brain)
            out['deg_cons'] = float(loss_cons)
            out['deg_cons_valid'] = float(ok)

        loss_sc = None
        if c['lambda_sc'] > 0 and fg_any:
            g = gt_idx.long()
            xt = torch.gather(p, 1, g)                #
            x = torch.gather(q, 1, g).detach()        #
            w = torch.clamp(1.0 - x, 0.0, 1.0) ** float(c['sc_gamma'])
            loss_sc, ok = _masked_mean(w * F.softplus(x - xt), fg)
            out['deg_sc'] = float(loss_sc)
            out['deg_sc_valid'] = float(ok)
            out['deg_sc_wmean'] = float(w[fg].mean())
            out['deg_prob_clean_fg'] = float(xt[fg].mean())
            out['deg_prob_deg_fg'] = float(x[fg].mean())

        loss_struct = None
        if c['lambda_struct'] > 0 and self.tap is None:
            raise RuntimeError('Structure loss requires attach(model) before forward')
        if c['lambda_struct'] > 0 and self.tap is not None:
            f_clean, f_deg = self.take('clean'), self.take('degraded')
            if f_clean is not None and f_deg is not None:
                loss_struct, s_stats = structure_margin_loss(
                    f_deg, f_clean, gt_idx, mask=fg, margin=c['struct_margin'],
                    samples=int(c['struct_samples']), cyclic=bool(c['struct_cyclic']),
                    num_classes=n_class)
                out['deg_struct'] = float(loss_struct)
                out.update({'deg_' + k: v for k, v in s_stats.items()})
            else:
                raise RuntimeError('Structure loss enabled but clean/degraded feature hooks are missing')
        self.take('clean')
        self.take('degraded')

        total = y_logits_clean.sum() * 0.0      #
        if loss_deg is not None:
            total = total + float(c['lambda_deg']) * loss_deg
        if loss_cons is not None:
            total = total + float(c['lambda_cons']) * loss_cons
        if loss_sc is not None:
            total = total + float(c['lambda_sc']) * loss_sc
        if loss_struct is not None:
            total = total + float(c['lambda_struct']) * loss_struct
        out['deg_total'] = float(total)
        if self.last_info is not None:
            out['deg_family'] = self.last_info['family']
            out['deg_severity'] = float(self.last_info['severity'])
        return total, out

__all__ = ['DEGRADE_DEFAULTS', 'DegradationScheduler', 'SegFeatureTap', 'DegradationAwareAux',
           'structure_margin_loss', 'hard_multiclass_dice', 'load_class_weight', 'parse_families']
