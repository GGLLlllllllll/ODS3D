import sys
from torch.utils.tensorboard import SummaryWriter
import os
import random
import glob
import json
import hashlib
import math
import time
import losses3D
import degrade3D
import degrade_aware
import argparse
from torch.utils.data import DataLoader
from data import datasets_mixed as datasets
import numpy as np
import torch
from torch import optim
import utils.utils as utils
import torch.nn.functional as F
import utils.augmentation as aug
from datetime import datetime
from scipy import ndimage

from models import *

shape = (128, 128, 128)
torch.set_printoptions(precision=8)

def segmentation_state_dict(model):
    """ 

      Architecture  
    ``model.load_state_dict(weights, strict=False)``  
    """
    prefixes = (
        'fea_extractor.', 'Seg_Head.',
        'seg_up1.', 'seg_up2.', 'seg_up3.',
    )
    if model.use_moe:
        prefixes += (
            'pw1.', 'pw2.', 'pw3.',
            'seg_gate1.', 'seg_gate2.', 'seg_gate3.',
        )

    state = model.state_dict()
    selected = {key: value for key, value in state.items()
                if key.startswith(prefixes)}
    missing = [prefix for prefix in prefixes
               if not any(key.startswith(prefix) for key in selected)]
    if missing:
        raise RuntimeError(' : {}'.format(', '.join(missing)))
    return selected

def seed_everything(seed):
    """ MoE CPU/CUDA """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """ Windows DataLoader worker """
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "/logfile.log", "a", encoding='utf-8', errors='replace')
        print('save path: ', save_dir)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

_HERE = os.path.dirname(os.path.abspath(__file__))
GATE_CACHE_DIR = os.path.join(_HERE, 'gate_cache')

# ============================================================================
# ============================================================================
GATEKEEPERS = [
    dict(name='SAM2-L', kind='sam2',
         ckpt=os.path.join(_HERE, 'sam2.1_hiera_large.pt'),
         cfg='configs/sam2.1_hiera_l512.yaml'),

    dict(name='MedSAM2-pretrain', kind='sam2',
         ckpt=os.path.join(_HERE, 'MedSAM2_pretrain.pth'),
         cfg='configs/sam2.1_hiera_t512.yaml'),
]

GATE_BUILD = dict(
    fit_volumes=20,  #
    samples_per_vol=2500,  #
    erode=2,  #
    quantiles='0.40,0.50,0.60,0.70',
    teacher_reliability='',  #
    slice_batch=8,  #
    auto=True,  #
)

GATE_USE_SAM3 = False
GATE_SAM3_CKPT = os.path.join(_HERE, 'medsam3.pt')  #
GATE_SAM3_BASE_CKPT = os.path.join(_HERE, 'sam3_base.pt')  #

def _gate_case_name(path):
    name = os.path.basename(path)
    for suf in ('.nii.gz', '.nii', '.npz', '.mgz'):
        if name.endswith(suf):
            return name[:-len(suf)]
    return name

def discover_atlas_pairs(atlas_dir):
    """  atlas   5  / 

      atlas   atlas_vol/atlas_seg_14labels  
    PPMI   vol/seg   __vol/__seg
     
    """
    atlas_dir = os.path.abspath(atlas_dir)
    if not os.path.isdir(atlas_dir):
        raise FileNotFoundError('atlas  ' + atlas_dir)

    images, labels = {}, {}
    files = sorted(
        path for path in glob.glob(os.path.join(atlas_dir, '**', '*'), recursive=True)
        if os.path.isfile(path) and path.lower().endswith(('.nii', '.nii.gz'))
    )
    for path in files:
        stem = _gate_case_name(path)
        lower = stem.lower()
        if lower == 'atlas_vol':
            images['atlas'] = path
        elif lower.startswith('atlas_seg'):
            labels['atlas'] = path
        elif lower.endswith('_vol'):
            images[lower[:-4]] = path
        elif lower.endswith('_seg'):
            labels[lower[:-4]] = path

    keys = sorted(set(images) | set(labels))
    incomplete = [key for key in keys if key not in images or key not in labels]
    if incomplete:
        raise FileNotFoundError('atlas  / {}'.format(incomplete))
    pairs = [(images[key], labels[key]) for key in keys]
    if len(pairs) != 5:
        raise ValueError(
            '  5   atlas  {}  {}'.format(len(pairs), keys)
        )
    return pairs

def _prepare_multi_atlas_sample(target_path, atlas_image_path, atlas_label_path):
    """  TrainOneShot/InferDataset   128^3  """
    target = datasets.load_volfile(target_path)
    source = datasets.load_volfile(atlas_image_path)
    target = datasets.normalize_image_intensity(target)
    source = datasets.normalize_image_intensity(source)
    source_seg = datasets.convert(datasets.load_volfile(atlas_label_path))

    c, h, w = source.shape[:3]
    factors = (128 / c, 128 / h, 128 / w)
    target = ndimage.zoom(target, factors, order=1)[None, ...]
    source = ndimage.zoom(source, factors, order=1)[None, ...]
    source_seg = ndimage.zoom(source_seg, factors, order=0)[None, ...]
    return tuple(
        torch.from_numpy(np.ascontiguousarray(array))
        for array in (source, target, source_seg)
    )

class MultiAtlasTrainDataset(torch.utils.data.Dataset):
    """  OASIS   5   atlas  """
    def __init__(self, args):
        self.args = args
        self.targets = sorted(glob.glob(args.train_path))
        self.atlas_pairs = list(args.atlas_pairs)
        if not self.targets:
            raise FileNotFoundError(' ' + args.train_path)

    def __len__(self):
        return len(self.targets) * len(self.atlas_pairs)

    def __getitem__(self, index):
        atlas_count = len(self.atlas_pairs)
        target_path = self.targets[index // atlas_count]
        atlas_image, atlas_label = self.atlas_pairs[index % atlas_count]
        sample = _prepare_multi_atlas_sample(target_path, atlas_image, atlas_label)
        if getattr(self.args, 'return_case_name', False):
            return sample + (_gate_case_name(target_path), _gate_case_name(atlas_image))
        return sample

class MultiAtlasInferDataset(torch.utils.data.Dataset):
    """  5   atlas  """
    def __init__(self, args):
        self.targets = sorted(glob.glob(args.test_path))
        self.atlas_pairs = list(args.atlas_pairs)
        if not self.targets:
            raise FileNotFoundError(' ' + args.test_path)

    def __len__(self):
        return len(self.targets) * len(self.atlas_pairs)

    def __getitem__(self, index):
        atlas_count = len(self.atlas_pairs)
        target_path = self.targets[index // atlas_count]
        atlas_image, atlas_label = self.atlas_pairs[index % atlas_count]
        source, target, source_seg = _prepare_multi_atlas_sample(
            target_path, atlas_image, atlas_label
        )
        target_label_path = target_path.replace('Testvol', 'Testseg').replace('_vol', '_seg')
        if not os.path.isfile(target_label_path):
            parent_dir = os.path.dirname(target_path)
            if os.path.basename(parent_dir).lower() == 'vol':
                target_label_path = os.path.join(
                    os.path.dirname(parent_dir), 'seg',
                    os.path.basename(target_path).replace('_vol', '_seg')
                )
        if not os.path.isfile(target_label_path):
            raise FileNotFoundError(' ' + target_label_path)
        target_seg = datasets.convert(datasets.load_volfile(target_label_path))
        c, h, w = target_seg.shape[:3]
        target_seg = ndimage.zoom(
            target_seg, (128 / c, 128 / h, 128 / w), order=0
        )[None, ...]
        target_seg = torch.from_numpy(np.ascontiguousarray(target_seg))
        return source, target, source_seg, target_seg

def _gate_load_volume_128(path, device):
    """  TrainOneShot  min-max   + zoom   128^3"""
    vol = datasets.normalize_image_intensity(datasets.load_volfile(path))
    c, h, w = vol.shape
    vol = ndimage.zoom(vol, (128 / c, 128 / h, 128 / w), order=1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(vol))[None, None].to(device)

def _gate_slice_indices(depth, step):
    """native   j  """
    dn = depth // step
    return [min(step * d + step // 2, depth - 1) for d in range(dn)], dn

@torch.no_grad()
def _gate_level_features(enc, vol_t, level_k, zs, res, slice_batch, axis, mean, std):
    """[1,1,X,Y,Z] -> [n, 256, h, w]  MedSAM2Aligner._sam_slices  """
    v = vol_t[0, 0].movedim(axis, 0)
    idx = torch.as_tensor(zs, dtype=torch.long, device=v.device)
    sl = v.index_select(0, idx).unsqueeze(1)
    sl = F.interpolate(sl, size=(res, res), mode='bilinear', align_corners=False)
    sl = sl.repeat(1, 3, 1, 1)
    sl = (sl - mean) / std
    outs = []
    for i in range(0, sl.shape[0], slice_batch):
        outs.append(enc(sl[i:i + slice_batch])['backbone_fpn'][level_k])
    return torch.cat(outs, 0)

def _gate_brain_mask_128(vol_t, erode=2):
    m = (vol_t[0, 0] > 1e-3).float()[None, None]  # [1,1,X,Y,Z]
    if erode > 0:
        k = 2 * erode + 1
        m = -F.max_pool3d(-m, kernel_size=k, stride=1, padding=erode)
    return m[0, 0]  # [X,Y,Z]

def _gate_flat_feats(feat):
    """[dn,C,h,w] -> [dn*h*w, C]"""
    return feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])

def _gate_sample_indices(mask_flat, n, rng):
    """  n  rho 0 """
    idx = torch.nonzero(mask_flat.reshape(-1) > 0, as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    if idx.numel() > n:
        return torch.from_numpy(rng.choice(idx.cpu().numpy(), size=n, replace=False)).to(idx.device)
    return idx

def _gate_cca_fit(X, Y, ridge=1e-3):
    """X[N,p], Y[N,q] -> CCA  float64  """
    X = X.double()
    Y = Y.double()
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-6
    mu_y, sd_y = Y.mean(0), Y.std(0) + 1e-6
    Xs = (X - mu_x) / sd_x
    Ys = (Y - mu_y) / sd_y
    n = Xs.shape[0]
    p, q = Xs.shape[1], Ys.shape[1]
    eye_p = torch.eye(p, dtype=torch.float64, device=X.device)
    eye_q = torch.eye(q, dtype=torch.float64, device=X.device)
    Cxx = Xs.T @ Xs / (n - 1) + ridge * eye_p
    Cyy = Ys.T @ Ys / (n - 1) + ridge * eye_q
    Cxy = Xs.T @ Ys / (n - 1)
    Lx = torch.linalg.cholesky(Cxx)
    Ly = torch.linalg.cholesky(Cyy)
    tmp = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    K = torch.linalg.solve_triangular(Ly, tmp.T, upper=False).T
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    A = torch.linalg.solve_triangular(Lx.T, U, upper=True)
    B = torch.linalg.solve_triangular(Ly.T, Vh.T, upper=True)
    return dict(mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, A=A, B=B, rho=S)

def _gate_cca_project(X, coef, side, k):
    X = X.float()
    if side == 'x':
        mu, sd, W = coef['mu_x'], coef['sd_x'], coef['A']
    else:
        mu, sd, W = coef['mu_y'], coef['sd_y'], coef['B']
    return ((X - mu.to(X.device).float()) / sd.to(X.device).float()) @ W.to(X.device).float()[:, :k]

def _gate_cca_holdout_rho(coef, X, Y, k):
    """ """
    zx = _gate_cca_project(X, coef, 'x', k)
    zy = _gate_cca_project(Y, coef, 'y', k)
    zx = zx - zx.mean(0, keepdim=True)
    zy = zy - zy.mean(0, keepdim=True)
    return (zx * zy).mean(0) / (zx.std(0) * zy.std(0) + 1e-8)

def _gate_ridge_map(X, Y, lam=1e-2):
    """   (MedSAM2) ->  """
    X = X.double()
    Y = Y.double()
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-6
    mu_y, sd_y = Y.mean(0), Y.std(0) + 1e-6
    Xs = (X - mu_x) / sd_x
    Ys = (Y - mu_y) / sd_y
    p = Xs.shape[1]
    A = Xs.T @ Xs + lam * Xs.shape[0] * torch.eye(p, dtype=torch.float64, device=X.device)
    W = torch.linalg.solve(A, Xs.T @ Ys)
    return dict(W=W, mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, dim=int(p))

def _gate_agreement(coef, f_med, f_other, mask_flat):
    """ " "  a = 1 - ||Ys - W Xs||^2 / dim

    r = 1 - a  r = 1  " " 
      r < 1  a > 0 
    """
    X = _gate_flat_feats(f_med)
    Y = _gate_flat_feats(f_other)
    dev = X.device
    Xs = (X - coef['mu_x'].to(dev).float()) / coef['sd_x'].to(dev).float()
    Ys = (Y - coef['mu_y'].to(dev).float()) / coef['sd_y'].to(dev).float()
    pred = Xs @ coef['W'].to(dev).float()
    return (1.0 - (Ys - pred).pow(2).sum(1) / float(coef['dim'])).reshape(mask_flat.shape)

def _gate_endorsement(a_raw, reliability=1.0):
    """  e = 1 - c*(1 - a)  [0,1] c=1   e == a """
    return (1.0 - float(reliability) * (1.0 - a_raw)).clamp(0.0, 1.0)

def _gate_pct_from_hist(hist, q):
    cdf = np.cumsum(hist).astype(np.float64)
    total = cdf[-1] if cdf[-1] > 0 else 1.0
    return int(np.searchsorted(cdf, q * total))

def _gate_build_sam3_encoder(ckpt_path, base_ckpt, device):
    """SAM3  " " """
    missing = []
    if not os.path.exists(base_ckpt):
        missing.append('SAM3  : %s' % base_ckpt)
    if not os.path.exists(ckpt_path):
        missing.append('SAM3 LoRA  : %s' % ckpt_path)
    try:
        import sam3  # noqa: F401
    except Exception as e:
        missing.append('sam3  import sam3  : %s ' % e)
    if missing:
        raise RuntimeError('GATE_USE_SAM3=True  \n  - '
                           + '\n  - '.join(missing) +
                           '\n medsam3.pt   LoRA delta rank16/alpha32  SAM3  '
                           '  sam3   GATE_USE_SAM3   False  ')
    raise RuntimeError('SAM3  '
                       '  _gate_build_sam3_encoder()   sam3  ')

def build_gate_cache(args, paths, atlas_paths):
    """  args.align_gate_dir  True """
    from medsam2_align import build_medsam2_image_encoder, checkpoint_sha256, IMAGENET_MEAN, IMAGENET_STD

    if int(args.medsam2_res) != 256:
        raise ValueError('Gate cache builder currently supports res=256 only')
    cfg = GATE_BUILD
    dev = args.device
    rng = np.random.RandomState(0)
    out_dir = args.align_gate_dir
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    gate_specs = [dict(sp) for sp in GATEKEEPERS]
    if GATE_USE_SAM3:
        gate_specs.append(dict(name='SAM3-LoRA', kind='sam3',
                               ckpt=GATE_SAM3_CKPT, cfg=''))
    if not gate_specs:
        print('!! GATEKEEPERS  ')
        return False
    if not any(sp['kind'] == 'sam2' for sp in gate_specs):
        print('!! GATEKEEPERS   kind="sam2"  ')
        return False
    rel = ([float(v) for v in cfg['teacher_reliability'].split(',') if v.strip() != '']
           if str(cfg['teacher_reliability']).strip() else [1.0] * len(gate_specs))
    if len(rel) != len(gate_specs):
        print('!! teacher_reliability   %d   %d  ' % (len(rel), len(gate_specs)))
        return False

    t_start = time.time()
    print('=' * 90)
    print(' MedSAM2   min ')
    print('   /  MedSAM2 : %s' % args.medsam2_ckpt)
    for t, sp in enumerate(gate_specs):
        print('    #%d %-16s: %s   (  c=%.2f)' % (t + 1, sp['name'], sp['ckpt'], rel[t]))
    print('   : w(v) = min_t clamp(1 - c_t*(1 - a_t(v)), 0, 1)')
    print('   : %s' % out_dir)
    print('=' * 90)

    enc_med = build_medsam2_image_encoder(args.medsam2_ckpt, args.medsam2_cfg, scalp=1, device=dev)
    gate_encs = []
    for sp in gate_specs:
        if sp['kind'] == 'sam2':
            gate_encs.append(build_medsam2_image_encoder(sp['ckpt'], sp['cfg'], scalp=1, device=dev))
        else:
            gate_encs.append(_gate_build_sam3_encoder(sp['ckpt'], GATE_SAM3_BASE_CKPT, dev))
    print('MedSAM2   %.1fM  %s'
          % (sum(p.numel() for p in enc_med.parameters()) / 1e6,
             ' + '.join('%.1fM' % (sum(p.numel() for p in e.parameters()) / 1e6) for e in gate_encs)))

    all_paths = list(paths) + list(atlas_paths)
    print(' : %d  %d + atlas %d '
          % (len(all_paths), len(paths), len(atlas_paths)))

    levels = (1, 2, 3)  #
    steps = {j: 2 ** j for j in levels}
    n_gate = len(gate_specs)
    res, axis, sb = int(args.medsam2_res), int(args.medsam2_axis), int(cfg['slice_batch'])

    idx_fit = np.linspace(0, len(paths) - 1, num=min(int(cfg['fit_volumes']), len(paths))).astype(int)
    if len(idx_fit) < 2:
        print('!! fit_volumes   2 ')
        return False
    half = len(idx_fit) // 2
    print('\n[1/2]   %d   %d   / %d  '
          % (len(idx_fit), half, len(idx_fit) - half))
    X_all = {j: [] for j in levels}
    Y_all = [{j: [] for j in levels} for _ in range(n_gate)]
    counts = {j: [] for j in levels}
    t0 = time.time()
    for n, vi in enumerate(idx_fit):
        vol_t = _gate_load_volume_128(paths[vi], dev)
        mask128 = _gate_brain_mask_128(vol_t, int(cfg['erode']))
        for j in levels:
            k, step = j - 1, steps[j]
            zs, dn = _gate_slice_indices(128, step)
            f_med = _gate_level_features(enc_med, vol_t, k, zs, res, sb, axis, mean, std)
            m = F.max_pool3d(mask128[None, None], kernel_size=step, stride=step)[0, 0].movedim(axis, 0)
            sel = _gate_sample_indices(m, int(cfg['samples_per_vol']), rng)
            if sel is None:
                counts[j].append(0)
                continue
            X_flat = _gate_flat_feats(f_med)
            X_all[j].append(X_flat[sel].cpu())  #
            for t, enc in enumerate(gate_encs):
                f_t = _gate_level_features(enc, vol_t, k, zs, res, sb, axis, mean, std)
                Y_all[t][j].append(_gate_flat_feats(f_t)[sel].cpu())
                del f_t
            counts[j].append(int(sel.numel()))
            del f_med, X_flat
        del vol_t
        torch.cuda.empty_cache()
        if (n + 1) % 5 == 0:
            print('    ... %d/%d  ,   %.1fs' % (n + 1, len(idx_fit), time.time() - t0))

    ridges = [{j: None for j in levels} for _ in range(n_gate)]
    cca_info = [{j: None for j in levels} for _ in range(n_gate)]
    rho_hold_k = [{j: None for j in levels} for _ in range(n_gate)]
    for j in levels:
        X = torch.cat(X_all[j], 0)
        cut = int(sum(counts[j][:half]))
        for t in range(n_gate):
            Y = torch.cat(Y_all[t][j], 0)
            c_fit = _gate_cca_fit(X[:cut], Y[:cut])
            rho_hold = _gate_cca_holdout_rho(c_fit, X[cut:], Y[cut:], min(64, X.shape[1]))
            k = max(4, min(64, int((rho_hold >= 0.5).sum())))
            ridges[t][j] = _gate_ridge_map(X, Y)
            rho_hold_k[t][j] = float(rho_hold[min(k, len(rho_hold)) - 1])
            cca_info[t][j] = dict(
                k=k,
                rho_fit_top20=[round(float(v), 3) for v in _gate_cca_fit(X, Y)['rho'][:20]],
                rho_holdout_top20=[round(float(v), 3) for v in rho_hold[:20]],
                rho_holdout_at_k=round(rho_hold_k[t][j], 3),
                n_samples=int(X.shape[0]))
            print('   native L%d | %-16s:   %d,   rho   5 = %s,   k=%d'
                  % (j, gate_specs[t]['name'], X.shape[0],
                     [round(float(v), 3) for v in rho_hold[:5]], k))
            del Y
        del X, X_all[j]
        for t in range(n_gate):
            Y_all[t][j] = None
    del X_all, Y_all

    first_sam2 = next(sp for sp in gate_specs if sp['kind'] == 'sam2')
    coef_path = os.path.join(out_dir, 'gate_coefs.pt')
    torch.save({
        'online_provenance': dict(
            target=[args.medsam2_cfg, checkpoint_sha256(args.medsam2_ckpt)],
            teacher=[first_sam2['cfg'], checkpoint_sha256(first_sam2['ckpt'])]),
        'levels': [int(j) for j in levels],
        'coefs': {str(j): ridges[gate_specs.index(first_sam2)][j] for j in levels},  #
        'coefs_multi': {str(j): [ridges[t][j] for t in range(n_gate)] for j in levels},
        'teachers': [sp['name'] for sp in gate_specs],
        'teacher_reliability': [float(v) for v in rel],
        'fusion': 'min_t clamp(1 - c_t*(1 - a_t), 0, 1)',
        'cca': {str(j): [cca_info[t][j] for t in range(n_gate)] for j in levels},
        'res': res, 'axis': axis, 'erode': int(cfg['erode']),
        'medsam2_ckpt': args.medsam2_ckpt,
        'teacher_ckpts': [sp['ckpt'] for sp in gate_specs],
    }, coef_path)
    print(' : %s' % coef_path)

    print('\n[2/2]  %d  ...' % len(all_paths))
    hist = {j: np.zeros(256, dtype=np.int64) for j in levels}
    hist_t = [{j: np.zeros(256, dtype=np.int64) for j in levels} for _ in range(n_gate)]
    core_sum = {j: 0.0 for j in levels}
    core_cnt = {j: 0 for j in levels}
    shell_sum = {j: 0.0 for j in levels}
    shell_cnt = {j: 0 for j in levels}
    core_sum_t = [{j: 0.0 for j in levels} for _ in range(n_gate)]
    shell_sum_t = [{j: 0.0 for j in levels} for _ in range(n_gate)]
    wmin = {j: 1e9 for j in levels}
    wmax = {j: -1e9 for j in levels}
    t0 = time.time()
    for n, path in enumerate(all_paths):
        cname = _gate_case_name(path)
        vol_t = _gate_load_volume_128(path, dev)
        mask_core = _gate_brain_mask_128(vol_t, int(cfg['erode']))
        mask_full = (vol_t[0, 0] > 1e-3).float()
        out = torch.zeros(3, 128, 128, 128, dtype=torch.uint8)
        for j in levels:
            k, step = j - 1, steps[j]
            zs, dn = _gate_slice_indices(128, step)
            f_med = _gate_level_features(enc_med, vol_t, k, zs, res, sb, axis, mean, std)
            m_core = F.max_pool3d(mask_core[None, None], kernel_size=step, stride=step)[0, 0].movedim(axis, 0)
            m_full = F.max_pool3d(mask_full[None, None], kernel_size=step, stride=step)[0, 0].movedim(axis, 0)
            shell = (m_full > 0) & (m_core <= 0)
            e_fused = None
            for t, enc in enumerate(gate_encs):
                f_t = _gate_level_features(enc, vol_t, k, zs, res, sb, axis, mean, std)
                a_t = _gate_agreement(ridges[t][j], f_med, f_t, m_full)
                core_sum_t[t][j] += float(a_t[m_core > 0].sum())
                shell_sum_t[t][j] += float(a_t[shell].sum())
                if (m_core > 0).any():
                    qa = torch.clamp(torch.round(torch.clamp(a_t, 0.0, 1.0) * 254.0) + 1.0, 1, 255)
                    hist_t[t][j] += np.bincount(qa[m_core > 0].cpu().numpy().astype(np.int64),
                                                minlength=256)[:256]
                e_t = _gate_endorsement(a_t, rel[t])
                e_fused = e_t if e_fused is None else torch.minimum(e_fused, e_t)
                del f_t, a_t, e_t
            w = e_fused  #
            core_sum[j] += float(w[m_core > 0].sum())
            core_cnt[j] += int((m_core > 0).sum())
            shell_sum[j] += float(w[shell].sum())
            shell_cnt[j] += int(shell.sum())
            wmin[j] = min(wmin[j], float(w.min()))
            wmax[j] = max(wmax[j], float(w.max()))
            w128 = F.interpolate(w[None, None], size=(128, 128, 128), mode='nearest')[0, 0]
            w128 = w128.movedim(0, axis)
            q = torch.clamp(torch.round(torch.clamp(w128, 0.0, 1.0) * 254.0) + 1.0, 1, 255).to(torch.uint8)
            q = q * (mask_core > 0).to(torch.uint8)  #
            out[j - 1] = q.cpu()
            inmask = q[mask_core > 0]
            if inmask.numel():
                hist[j] += np.bincount(inmask.cpu().numpy().astype(np.int64), minlength=256)[:256]
            del f_med, w, w128, q, inmask
        np.save(os.path.join(out_dir, cname + '.npy'), out.numpy())
        del vol_t, out
        torch.cuda.empty_cache()
        if (n + 1) % 20 == 0 or (n + 1) == len(all_paths):
            el = time.time() - t0
            print('    ... %d/%d  ,   %.1f min,   %.1f min'
                  % (n + 1, len(all_paths), el / 60, (el / (n + 1)) * (len(all_paths) - n - 1) / 60))

    qs = [float(x) for x in str(cfg['quantiles']).split(',')]
    thr_table = {}
    for q in qs:
        key = '%.2f' % q
        thr_table[key] = {}
        for j in levels:
            cdf = np.cumsum(hist[j]).astype(np.float64)
            total = cdf[-1] if cdf[-1] > 0 else 1.0
            t = int(np.searchsorted(cdf, (1.0 - q) * total))  #
            thr_table[key][str(j)] = int(min(max(t, 1), 255))
    mean_uint8 = {}
    for j in levels:
        v = np.arange(256, dtype=np.float64)
        total = hist[j].sum()
        mean_uint8[str(j)] = int(round(float((v * hist[j]).sum() / total))) if total else 128

    per_teacher = {}
    for t, sp in enumerate(gate_specs):
        per_teacher[sp['name']] = {str(j): dict(
            reliability=float(rel[t]),
            holdout_rho_at_k=round(float(rho_hold_k[t][j]), 4),
            core_mean=round(core_sum_t[t][j] / max(core_cnt[j], 1), 4),
            shell_mean=round(shell_sum_t[t][j] / max(shell_cnt[j], 1), 4),
            p10=_gate_pct_from_hist(hist_t[t][j], 0.10),
            p50=_gate_pct_from_hist(hist_t[t][j], 0.50),
            p90=_gate_pct_from_hist(hist_t[t][j], 0.90)) for j in levels}

    meta = dict(
        signature=_gate_signature(args),
        created=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        res=res, axis=axis, erode=int(cfg['erode']),
        res_note='MedSAM2   res  ',
        fusion='w(v) = min_t clamp(1 - c_t*(1 - a_t(v)), 0, 1)   #  ',
        target_teacher=dict(name='MedSAM2', ckpt=args.medsam2_ckpt, cfg=args.medsam2_cfg),
        gatekeepers=[dict(name=sp['name'], ckpt=sp['ckpt'], cfg=sp['cfg'],
                          reliability=float(rel[t])) for t, sp in enumerate(gate_specs)],
        medsam2_ckpt=args.medsam2_ckpt, medsam2_cfg=args.medsam2_cfg,
        n_volumes=len(all_paths), fit_volumes=[_gate_case_name(paths[i]) for i in idx_fit],
        cca={str(j): [cca_info[t][j] for t in range(n_gate)] for j in levels},
        thresholds=thr_table, mean_uint8=mean_uint8, per_teacher=per_teacher,
        diag={str(j): dict(
            core_mean_weight=round(core_sum[j] / max(core_cnt[j], 1), 4), core_voxels=core_cnt[j],
            shell_mean_weight=round(shell_sum[j] / max(shell_cnt[j], 1), 4), shell_voxels=shell_cnt[j],
            weight_min=round(wmin[j], 4), weight_max=round(wmax[j], 4),
            weight_p10=int(_gate_pct_from_hist(hist[j], 0.10)),
            weight_p50=int(_gate_pct_from_hist(hist[j], 0.50)),
            weight_p90=int(_gate_pct_from_hist(hist[j], 0.90))) for j in levels})
    with open(os.path.join(out_dir, 'gate_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 90)
    print('  %.1f min' % ((time.time() - t_start) / 60))
    for t, sp in enumerate(gate_specs):
        for j in levels:
            d = per_teacher[sp['name']][str(j)]
            print('  [%s] native L%d:   rho(k)=%.3f |   %.3f vs   %.3f |   p10/p50/p90 = %d/%d/%d'
                  % (sp['name'], j, d['holdout_rho_at_k'], d['core_mean'], d['shell_mean'],
                     d['p10'], d['p50'], d['p90']))
    for j in levels:
        d = meta['diag'][str(j)]
        print('  [ ] native L%d:   %.3f vs   %.3f |   [%.3f, %.3f] | p10/p50/p90 = %d/%d/%d'
              % (j, d['core_mean_weight'], d['shell_mean_weight'], d['weight_min'], d['weight_max'],
                 d['weight_p10'], d['weight_p50'], d['weight_p90']))
    print(' (quantile -> uint8):', json.dumps(thr_table, ensure_ascii=False))
    print('=' * 90)

    del enc_med, gate_encs
    torch.cuda.empty_cache()
    return True

def _gate_signature(args):
    def checkpoint(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    specs = list(GATEKEEPERS)
    if GATE_USE_SAM3:
        specs += [dict(name='SAM3-LoRA', ckpt=GATE_SAM3_CKPT, cfg='', kind='sam3')]
    paths = sorted(glob.glob(args.train_path)) + [pair[0] for pair in args.atlas_pairs]
    names = [_gate_case_name(p) for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate gate case names; map files would overwrite one another')
    return dict(version=2, res=int(args.medsam2_res), axis=int(args.medsam2_axis),
                build=GATE_BUILD, target=[args.medsam2_cfg, checkpoint(args.medsam2_ckpt)],
                teachers=[[s['name'], s['kind'], s['cfg'], checkpoint(s['ckpt'])] for s in specs],
                inputs=[[os.path.abspath(p), os.stat(p).st_size, os.stat(p).st_mtime_ns] for p in paths])

def _gate_cache_problems(args, n_expected):
    """  gate_cache   =  """
    d = args.align_gate_dir
    meta_p = os.path.join(d, 'gate_meta.json')
    coefs_p = os.path.join(d, 'gate_coefs.pt')
    problems = []
    if not os.path.isdir(d):
        problems.append(' ')
        return problems
    if not os.path.exists(meta_p):
        problems.append('  gate_meta.json')
    if not os.path.exists(coefs_p):
        problems.append('  gate_coefs.pt')
    expected_paths = sorted(glob.glob(args.train_path)) + [pair[0] for pair in args.atlas_pairs]
    expected = [_gate_case_name(p) for p in expected_paths]
    missing = [name for name in expected if not os.path.isfile(os.path.join(d, name + '.npy'))]
    if missing:
        problems.append('Missing gate cases: %s' % missing[:5])
    npy = glob.glob(os.path.join(d, '*.npy'))
    if len(npy) < n_expected:
        problems.append(' %d   /   %d  ' % (len(npy), n_expected))
    if os.path.exists(meta_p):
        try:
            with open(meta_p, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('signature') != _gate_signature(args):
                problems.append('Cache provenance/configuration mismatch; rebuild required')
            if int(meta.get('axis', -1)) != int(args.medsam2_axis):
                problems.append('Gate axis mismatch')
            got = [(g['name'], os.path.basename(g['ckpt'])) for g in meta.get('gatekeepers', [])]
            want = [(sp['name'], os.path.basename(sp['ckpt'])) for sp in GATEKEEPERS] +\
                   ([('SAM3-LoRA', os.path.basename(GATE_SAM3_CKPT))] if GATE_USE_SAM3 else [])
            if got != want:
                problems.append('  %s  %s'
                                % ([g[0] for g in got], [w[0] for w in want]))
            if int(meta.get('res', -1)) != int(args.medsam2_res):
                problems.append('res   %s  %s' % (meta.get('res'), args.medsam2_res))
        except Exception as e:
            problems.append('gate_meta.json  : %s' % e)
    return problems

def ensure_gate_cache(args):
    """ /  16   True= """
    d = args.align_gate_dir
    paths = sorted(glob.glob(args.train_path))
    if not paths:
        print('!!  %s' % args.train_path)
        return False
    atlas_paths = [pair[0] for pair in args.atlas_pairs]
    n_expected = len(paths) + len(atlas_paths)
    problems = _gate_cache_problems(args, n_expected)
    if not problems:
        print(' %d   + gate_meta.json + gate_coefs.pt'
              % n_expected)
        return True
    print('=' * 90)
    print(' ' + ' '.join(problems))
    for t, sp in enumerate(GATEKEEPERS):
        print('    #%d: %-18s %s' % (t + 1, sp['name'], sp['ckpt']))
    if not GATE_BUILD['auto']:
        print('  GATE_BUILD["auto"]=False     ')
        print('=' * 90)
        return False
    print('    16  ')
    print('    %s  ' % d)
    print('=' * 90)
    try:
        ok = build_gate_cache(args, paths, atlas_paths)
    except Exception as e:
        print('!!  %s: %s' % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
        return False
    if not ok:
        return False
    problems = _gate_cache_problems(args, n_expected)
    if problems:
        print('!!  ' + ' '.join(problems))
        return False
    print(' ')
    return True

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cover_path', type=bool, default=True,
                        help=" true, false")
    
    parser.add_argument('--save_path', type=str,
                        default=os.path.join(
                            _HERE, 'results', 'medsam2_degrade_multiatlas'
                        ),
                        help="all log save path Windows  ")
    
    parser.add_argument(
        '--train_path',
        type=str,
        default='',
        help="train dataset path"
    )
    
    parser.add_argument('--test_path', type=str,
                        default='',
                        help="test dataset path")

    parser.add_argument('--atlas_dir', type=str,
                        default='',
                        help="  5   atlas  ")
    
    parser.add_argument('--pre_path', type=str, default='',
                        help="pretrain model path")

    parser.add_argument('--num_workers', type=int, default=2,
                        help="  DataLoader   worker  train/test   2  "
                             "  0  commit ")
    parser.add_argument('--pin_memory', type=int, default=0,
                        help="1=  0")

    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--epochs', type=int, default=1000, help="the number of global epochs: G")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--cont_training', type=bool, default=False, help="continue training")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for model/data order/augmentation")

    parser.add_argument('--no_medsam2_align', action='store_true', default=False,
                        help="  MedSAM2  ")
    parser.add_argument('--medsam2_ckpt', type=str,
                        default='',
                        help="  MedSAM2  ")
    parser.add_argument('--medsam2_cfg', type=str, default='configs/sam2.1_hiera_t512.yaml')
    parser.add_argument('--medsam2_res', type=int, default=256, choices=[128, 256, 512],
                        help="MedSAM2  256   native L1~L3")
    parser.add_argument('--medsam2_slice_batch', type=int, default=0,
                        help=" 0= ")
    parser.add_argument('--medsam2_axis', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--align_lambda', type=float, default=0.01)
    parser.add_argument('--align_no_mask', action='store_true')

    parser.add_argument('--align_gate_dir', type=str, default=GATE_CACHE_DIR,
                        help=" ")
    parser.add_argument('--align_gate_meta', type=str, default='')
    parser.add_argument('--align_gate_quantile', type=float, default=-1.0)
    parser.add_argument('--align_gate_above_mean', action='store_true')
    parser.add_argument('--align_gate_mode', type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--align_gate_verbose', action='store_true', default=True)

    parser.add_argument('--align_decay', type=str, default='auto', choices=['none', 'epochs', 'auto'])
    parser.add_argument('--align_decay_start', type=int, default=10)
    parser.add_argument('--align_decay_len', type=int, default=5)
    parser.add_argument('--align_decay_impr', type=float, default=0.0)
    parser.add_argument('--align_decay_rate_ratio', type=float, default=0.13)
    parser.add_argument('--align_decay_patience', type=int, default=10)
    parser.add_argument('--align_grad_probe_every', type=int, default=100)

    parser.add_argument('--align_gate_online', action='store_true')
    parser.add_argument('--gate_sam2_ckpt', type=str, default='')
    parser.add_argument('--gate_medsam2_ckpt', type=str, default='')
    parser.add_argument('--gate_ckpt', type=str,
                        default='')
    parser.add_argument('--gate_cfg', type=str, default='configs/sam2.1_hiera_l512.yaml')
    parser.add_argument('--gate_coefs', type=str,
                        default=os.path.join(GATE_CACHE_DIR, 'gate_coefs.pt'))
    parser.add_argument('--gate_slice_batch', type=int, default=0)

    parser.add_argument('--disable_degrade', action='store_true', default=False,
                        help="  MedSAM2  ")
    parser.add_argument('--degrade_families', type=str, default='all',
                        help=" all= ")
    parser.add_argument('--degrade_family_scope', type=str, default='iteration',
                        choices=['epoch', 'iteration'],
                        help="iteration= epoch= ")
    parser.add_argument('--degrade_severity_min', type=float, default=0.25)
    parser.add_argument('--degrade_severity_max', type=float, default=0.75)
    parser.add_argument('--degrade_axis', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--degrade_every_k', type=int, default=1,
                        help="  k  1= ")
    parser.add_argument('--degrade_warmup_epochs', type=int, default=2,
                        help=" ")
    parser.add_argument('--degrade_lambda_deg', type=float, default=0.5)
    parser.add_argument('--degrade_gt_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--degrade_lambda_cons', type=float, default=0.1)
    parser.add_argument('--degrade_cons_symmetric', action='store_true', default=False)
    parser.add_argument('--degrade_lambda_sc', type=float, default=0)
    parser.add_argument('--degrade_sc_gamma', type=float, default=2.0)
    parser.add_argument('--degrade_lambda_struct', type=float, default=0)
    parser.add_argument('--degrade_struct_margin', type=float, default=0.15)
    parser.add_argument('--degrade_struct_samples', type=int, default=256)
    parser.add_argument('--degrade_struct_cyclic', action='store_true', default=True)
    parser.add_argument('--no_degrade_struct_cyclic', dest='degrade_struct_cyclic',
                        action='store_false')
    parser.add_argument('--degrade_class_weight_file', type=str, default='')
    parser.add_argument('--degrade_verbose_every', type=int, default=10)

    args = parser.parse_args()
    for key in ('train_path', 'test_path', 'atlas_dir'):
        if not getattr(args, key):
            parser.error(f'--{key} is required')
    if not args.no_medsam2_align and not args.medsam2_ckpt:
        parser.error('--medsam2_ckpt is required unless --no_medsam2_align is set')
    if not args.no_medsam2_align and args.align_gate_dir and not args.align_gate_online:
        for key in ('gate_sam2_ckpt', 'gate_medsam2_ckpt'):
            if not getattr(args, key):
                parser.error(f'--{key} is required for the offline gate')
        GATEKEEPERS[0]['ckpt'] = args.gate_sam2_ckpt
        GATEKEEPERS[1]['ckpt'] = args.gate_medsam2_ckpt
    args.atlas_pairs = discover_atlas_pairs(args.atlas_dir)
    args.src_path, args.sseg_path = args.atlas_pairs[0]
    print('Atlas directory:', args.atlas_dir)
    for atlas_index, (image_path, label_path) in enumerate(args.atlas_pairs, 1):
        print('  Atlas {}/5: {} | {}'.format(atlas_index, image_path, label_path))
    args.device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    seed_everything(args.seed)
    if args.align_gate_online:
        args.align_gate_dir = ''
    if not args.no_medsam2_align and (args.align_gate_online or args.align_gate_dir)\
            and args.medsam2_res != 256:
        raise ValueError('  medsam2_res=256 ')

    batch_size = 1
    best_sdice = 0
    best_rdice = 0
    best_spath = None
    best_rpath = None

    model_dir = args.save_path
    if args.cover_path:
        if os.path.exists(args.save_path):
            new_name = args.save_path + '_archived_' + get_timestamp()
            print('Path already exists. Rename it to [{:s}]'.format(new_name))
            os.rename(args.save_path, new_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = Logger(args.save_path)
    sys.stdout = logger
    sys.stderr = logger
    print('Random seed: {} (Python/NumPy/PyTorch/CUDA/DataLoader)'.format(args.seed))

    lr = args.lr
    epoch_start = 0
    img_size = shape
    n_class = 14

    # ========================
    # Initialize model
    # ========================
    model = Architecture(
        inshape=img_size,
        n_channels=1,
        n_classes=n_class,
        use_moe=False,
        use_mrb=False,
        # use_cgsa=False,
    ).to(args.device)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
    print('MoE enabled: {} (texture/geometry experts and voxel gates are disabled)'
          .format(model.use_moe))
    print('3-D MRB enabled: {} (disabled for MedSAM2-only experiment)'
          .format(model.use_mrb))
    # print('CGSA enabled: {} (disabled for this experiment)'.format(model.use_cgsa))
    print('Auxiliary branches: MedSAM2 + multi-teacher voxel gate + degradation-aware training')

    aligner = None
    if args.align_gate_dir and not args.no_medsam2_align:
        if not ensure_gate_cache(args):
            raise RuntimeError(' ')

    if not args.no_medsam2_align:
        from medsam2_align import MedSAM2Aligner
        gate_mode = 'online' if args.align_gate_online else ('cached' if args.align_gate_dir else 'none')
        aligner = MedSAM2Aligner(
            ckpt_path=args.medsam2_ckpt,
            cfg_name=args.medsam2_cfg,
            native_channels=(32, 64, 128, 256),
            res=args.medsam2_res,
            slice_batch=args.medsam2_slice_batch,
            axis=args.medsam2_axis,
            lambda_=args.align_lambda,
            mask_background=not args.align_no_mask,
            gate_mode=gate_mode,
            gate_ckpt=args.gate_ckpt if args.align_gate_online else None,
            gate_cfg=args.gate_cfg,
            gate_coefs=args.gate_coefs if args.align_gate_online else None,
            gate_quantile=args.align_gate_quantile,
            gate_above_mean=args.align_gate_above_mean,
            gate_slice_batch=args.gate_slice_batch,
            device=args.device).to(args.device).train()
        print(aligner.describe())
        print('MedSAM2 aligner trainable params: %d' %
              sum(p.numel() for p in aligner.trainable_parameters()))
    else:
        print('MedSAM2 align: DISABLED')

    gate_loader = None
    if aligner is not None and args.align_gate_dir:
        from medsam2_align import GateLoader
        gate_loader = GateLoader(
            args.align_gate_dir,
            meta_path=args.align_gate_meta or None,
            quantile=args.align_gate_quantile,
            mode=args.align_gate_mode,
            above_mean=args.align_gate_above_mean,
            device=args.device)
        args.return_case_name = True
        print(' : dir={} mode={} quantile={:.2f}'
              .format(args.align_gate_dir, args.align_gate_mode, args.align_gate_quantile))

    # ========================
    # Spatial transformation
    # ========================
    reg_model = utils.register_model('nearest').to(args.device)
    reg_model_bilin = utils.register_model('bilinear').to(args.device)

    # ========================
    # Continue training
    # ========================
    if args.cont_training:
        epoch_start = 0
        model_path = ''
        best_model = torch.load(model_path)
        print('Model: {} loaded!'.format(model_path))
        model.load_state_dict(best_model, strict=False)

    # ========================
    # Data augmentation
    # ========================
    spatial_aug = aug.SpatialTransform(
        do_rotation=True,
        angle_x=(-np.pi / 18, np.pi / 18),
        angle_y=(-np.pi / 18, np.pi / 18),
        angle_z=(-np.pi / 18, np.pi / 18),
        do_scale=True,
        scale=(0.9, 1.1)
    )
    
    # ========================
    # Loss / Hyper optimizer
    # ========================
    reg_seg_criterion = losses3D.LossFunction_Reg_Seg_opt(num_classes=n_class).to(args.device)

    deg_aux = None
    if not args.disable_degrade:
        deg_aux = degrade_aware.DegradationAwareAux(
            dict(families=args.degrade_families,
                 family_scope=args.degrade_family_scope,
                 severity_min=args.degrade_severity_min,
                 severity_max=args.degrade_severity_max,
                 axis=args.degrade_axis,
                 every_k=args.degrade_every_k,
                 warmup_epochs=args.degrade_warmup_epochs,
                 lambda_deg=args.degrade_lambda_deg,
                 gt_mode=args.degrade_gt_mode,
                 lambda_cons=args.degrade_lambda_cons,
                 cons_symmetric=args.degrade_cons_symmetric,
                 lambda_sc=args.degrade_lambda_sc,
                 sc_gamma=args.degrade_sc_gamma,
                 lambda_struct=args.degrade_lambda_struct,
                 struct_margin=args.degrade_struct_margin,
                 struct_samples=args.degrade_struct_samples,
                 struct_cyclic=args.degrade_struct_cyclic,
                 class_weight_file=args.degrade_class_weight_file),
            num_classes=n_class, device=args.device, seed=args.seed,
            axis=args.degrade_axis).attach(model)
        if args.degrade_class_weight_file:
            weights = degrade_aware.load_class_weight(
                args.degrade_class_weight_file, num_classes=n_class, device=args.device)
            deg_aux.set_class_weight(weights)
        print(degrade3D.family_table())
        print(deg_aux.describe())
        print('Degradation-aware training: ENABLED')
    else:
        print('Degradation-aware training: DISABLED')
    
    s_params = (
        list(model.Seg_Head.parameters()) +
        list(model.seg_up1.parameters()) + list(model.seg_up2.parameters()) + list(model.seg_up3.parameters())
    )

    r_params = (
        list(model.Reg_Head.parameters()) +
        list(model.reg_up1.parameters()) + list(model.reg_up2.parameters()) + list(model.reg_up3.parameters())
    )

    s_ids = set(id(p) for p in s_params)
    r_ids = set(id(p) for p in r_params)
    all_params = list(model.parameters())
    rest_params = [p for p in all_params if id(p) not in s_ids and id(p) not in r_ids]
    if aligner is not None:
        rest_params += aligner.trainable_parameters()

    s_optimizer = optim.Adam(s_params, lr=lr)
    r_optimizer = optim.Adam(r_params, lr=lr)
    optimizer = optim.Adam(rest_params, lr=lr)

    # ========================
    # Data loaders
    # ========================
    train_set = MultiAtlasTrainDataset(args)
    test_set = MultiAtlasInferDataset(args)
    print('Multi-atlas training pairs: {} targets x {} atlases = {} samples/epoch'
          .format(len(train_set.targets), len(train_set.atlas_pairs), len(train_set)))
    print('Multi-atlas test pairs: {} targets x {} atlases = {} pairs'
          .format(len(test_set.targets), len(test_set.atlas_pairs), len(test_set)))

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    test_generator = torch.Generator()
    test_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory),
        worker_init_fn=seed_worker, generator=train_generator
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory), drop_last=True,
        worker_init_fn=seed_worker, generator=test_generator
    )
    print('DataLoader: num_workers=%d  pin_memory=%s'
          % (args.num_workers, bool(args.pin_memory)))

    gate_atlas_cache = {}
    if gate_loader is not None:
        for atlas_image, _ in args.atlas_pairs:
            atlas_name = _gate_case_name(atlas_image)
            atlas_gate = gate_loader(atlas_name)
            gate_atlas_cache[atlas_name] = atlas_gate
            if atlas_gate is None:
                print(' :   atlas   {}  atlas  '.format(atlas_name))
            else:
                print('atlas  : {}  {:.3f}'
                      .format(atlas_name, float((atlas_gate > 0).float().mean())))

    tensorboard_dir = os.path.join(model_dir, 'tensorboard')
    try:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    except Exception as tensorboard_error:
        tensorboard_dir = os.path.join(
            _HERE, 'tensorboard_logs', 'medsam2_degrade_multiatlas_' + get_timestamp()
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(' : TensorBoard  {} {}'
              .format(tensorboard_error, tensorboard_dir))
        writer = SummaryWriter(log_dir=tensorboard_dir)
    print('TensorBoard log directory:', tensorboard_dir)

    decay_state = {'start': None, 'reason': ''}
    align_hist = []
    if args.align_decay == 'epochs':
        decay_state['start'] = int(args.align_decay_start)
        decay_state['reason'] = 'fixed(epoch %d)' % args.align_decay_start
    elif args.align_decay == 'auto':
        decay_state['reason'] = ('auto( :  %d <%.1f%% ,  %d )'
                                 % (int(args.align_decay_patience),
                                    args.align_decay_rate_ratio * 100,
                                    int(args.align_decay_start)))

    def current_align_lambda(ep):
        if args.align_decay == 'none' or decay_state['start'] is None:
            return args.align_lambda
        t = (ep - decay_state['start']) / max(int(args.align_decay_len), 1)
        if t >= 1.0:
            return 0.0
        if t <= 0.0:
            return args.align_lambda
        return args.align_lambda * 0.5 * (1.0 + math.cos(math.pi * t))

    # ========================
    # Train loop
    # ========================
    for epoch in range(epoch_start, args.epochs):

        lam_ep = current_align_lambda(epoch)
        if aligner is not None:
            closed = ' [ ]' if lam_ep <= 0 else ''
            print('Epoch {}: align lambda = {:.5f} {}{}'
                  .format(epoch, lam_ep, decay_state['reason'], closed))
        align_all, grad_ratio_all, gate_sel_all = [], [], []
        deg_terms_all = []
        deg_active = bool(deg_aux is not None and deg_aux.active(epoch))
        epoch_deg_family = deg_aux.epoch_family(epoch) if deg_aux is not None else None
        if deg_aux is not None:
            if deg_active:
                scope_msg = ('  {}'.format(epoch_deg_family)
                             if epoch_deg_family is not None else '  iteration  ')
                print('Epoch {}: DAT active ({})'.format(epoch, scope_msg))
            else:
                print('Epoch {}: DAT warmup ({}/{})'.format(
                    epoch, epoch, args.degrade_warmup_epochs))

        dice_list0, dice_list1, dice_list2 = [], [], []
        ncc_list0, ncc_list1, ncc_list2 = [], [], []

        print('Training Starts')

        for iter_num, data in enumerate(train_loader):
            gate_tgt_base = None
            gate_src_base = None
            if gate_loader is not None:
                data = list(data)
                case_name = data[3][0] if isinstance(data[3], (list, tuple)) else str(data[3])
                atlas_name = data[4][0] if isinstance(data[4], (list, tuple)) else str(data[4])
                data = data[:3]
                gate_tgt_base = gate_loader(case_name)
                gate_src_base = gate_atlas_cache.get(atlas_name)
                if gate_tgt_base is None and iter_num == 0:
                    print(' :   {}  '.format(case_name))
            data = [t.float().to(args.device, non_blocking=True) for t in data]
            x = data[0]       # atlas
            y = data[1]       # moving
            x_seg = data[2]

            code_spa = spatial_aug.rand_coords(x.shape[2:])
            x = spatial_aug.augment_spatial(x, code_spa)
            x_seg = spatial_aug.augment_spatial(x_seg, code_spa, mode='nearest')
            y = spatial_aug.augment_spatial(y, code_spa)

            model.train()

            global_step = epoch * len(train_loader) + iter_num
            deg_run = bool(deg_aux is not None and deg_active and deg_aux.due(global_step))

            # ==========================================
            # Forward
            # ==========================================
            encoded_x = model.fea_extractor(x)
            encoded_y = model.fea_extractor(y)
            x_y, flow = model._run_reg_branch(encoded_x, encoded_y, x)
            if deg_run:
                deg_aux.begin('clean')
            ylogits = model._run_seg_branch(encoded_y)
            xlogits = None

            # ==========================================
            # Normal network update
            # ==========================================
            loss, info = reg_seg_criterion(y, x_y, ylogits, x_seg, flow)

            deg_stats = None
            if deg_run:
                y_deg, _deg_info = deg_aux.sample(
                    y, global_step, family=epoch_deg_family)
                deg_aux.begin('degraded')
                encoded_y_deg = model.fea_extractor(y_deg)
                ylogits_deg = model._run_seg_branch(encoded_y_deg)

                def target_seg_dice(prob, atlas_seg, detached_flow, _n_class,
                                    class_weight=None):
                    if class_weight is not None:
                        raise ValueError('soft gt_mode   hard  ')
                    return reg_seg_criterion.seg_module(prob, atlas_seg, detached_flow)

                deg_loss, deg_stats = deg_aux.compute(
                    ylogits, ylogits_deg, x_seg, flow, y,
                    seg_dice_fn=target_seg_dice, n_class=n_class)
                loss = loss + deg_loss
                deg_terms_all.append(deg_stats)

            align_loss = None
            if aligner is not None and lam_ep > 0:
                gates = None
                if gate_loader is not None:
                    g_src = spatial_aug.augment_spatial(gate_src_base, code_spa, mode='nearest')\
                        if gate_src_base is not None else None
                    g_tgt = spatial_aug.augment_spatial(gate_tgt_base, code_spa, mode='nearest')\
                        if gate_tgt_base is not None else None
                    if g_src is not None and g_tgt is not None:
                        gates = [g_src, g_tgt]
                        if args.align_gate_verbose:
                            nch = max(int(g_tgt.shape[1]), 1)
                            gate_sel_all.append(float((g_tgt > 0).sum()) /
                                                max(nch * float((y > 1e-3).sum()), 1.0))
                align_loss, _align_stats = aligner(
                    [x, y], [encoded_x, encoded_y], gates=gates)
                task_loss = loss
                loss = loss + lam_ep * align_loss
                align_all.append(float(align_loss.detach()))

                if args.align_grad_probe_every > 0\
                        and iter_num % args.align_grad_probe_every == 0:
                    try:
                        feats = tuple(encoded_x) + tuple(encoded_y)
                        g_align = torch.autograd.grad(
                            lam_ep * align_loss, feats, retain_graph=True, allow_unused=True)
                        g_task = torch.autograd.grad(
                            task_loss, feats, retain_graph=True, allow_unused=True)
                        na = sum(float(v.norm() ** 2) for v in g_align if v is not None) ** 0.5
                        nt = sum(float(v.norm() ** 2) for v in g_task if v is not None) ** 0.5
                        grad_ratio_all.append(na / (nt + 1e-12))
                    except RuntimeError:
                        pass

            optimizer.zero_grad()
            r_optimizer.zero_grad()
            s_optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            r_optimizer.step()
            s_optimizer.step()

            # ==========================================
            # Logging
            # ==========================================
            if iter_num % 20 == 0:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Train: Epoch {} Iter {} / {} total {:.8f} ncc {:.8f} grad {:.8f} dice {:.8f}'.format(
                        epoch,
                        iter_num,
                        len(train_loader),
                        loss.item(),
                        info['ncc'].item(),
                        info['grad'].item(),
                        info['dice'].item()
                    )
                )
                if align_loss is not None:
                    print('align loss {:.8f} (lambda={:.5f})'
                          .format(align_loss.item(), lam_ep))
                if deg_stats is not None and args.degrade_verbose_every > 0\
                        and iter_num % max(args.degrade_verbose_every, 1) == 0:
                    def _df(key):
                        value = deg_stats.get(key, float('nan'))
                        return 'nan' if value is None or value != value else ('%.4f' % float(value))
                    print('deg[{}] sev={} loss={} | A dice_deg={} B cons={} '
                          'C sc={} D struct={} | P_clean_fg={} P_deg_fg={}'
                          .format(deg_stats.get('deg_family', '?'), _df('deg_severity'),
                                  _df('deg_total'), _df('deg_dice'), _df('deg_cons'),
                                  _df('deg_sc'), _df('deg_struct'),
                                  _df('deg_prob_clean_fg'), _df('deg_prob_deg_fg')))

        if align_all:
            mean_align = float(np.mean(align_all))
            print('Epoch {} mean align loss {:.8f}'.format(epoch, mean_align))
            writer.add_scalar('align_loss', mean_align, epoch)
            writer.add_scalar('align_lambda', lam_ep, epoch)
            align_hist.append(mean_align)
            if args.align_decay == 'auto' and decay_state['start'] is None:
                k = max(int(args.align_decay_patience), 1)
                if epoch >= args.align_decay_start and len(align_hist) > k:
                    prev = float(np.mean(align_hist[-k - 1:-1]))
                    cur = align_hist[-1]
                    rate = (prev - cur) / max(prev, 1e-12)
                    rate0 = (align_hist[0] - align_hist[k]) / max(align_hist[0], 1e-12)
                    hit, why = False, ''
                    if args.align_decay_rate_ratio > 0 and rate0 > 0\
                            and rate < args.align_decay_rate_ratio * rate0:
                        hit = True
                        why = ('  %.3f%%/%d    %.3f%%/%d    %.0f%%'
                               % (rate * 100, k, rate0 * 100, k,
                                  100 * rate / max(rate0, 1e-12)))
                    if args.align_decay_impr > 0 and rate < args.align_decay_impr:
                        hit = True
                        why = '  %.3f%% <   %.1f%%'\
                              % (rate * 100, args.align_decay_impr * 100)
                    if hit:
                        decay_state['start'] = epoch
                        decay_state['reason'] = 'auto( %d : %s)' % (epoch, why)
                        print('>>>  : ' + decay_state['reason'])
        if grad_ratio_all:
            ratio = float(np.mean(grad_ratio_all))
            print('Epoch {}   lam*|dL_align|/|dL_task| = {:.5f}'.format(epoch, ratio))
            writer.add_scalar('align_grad_ratio', ratio, epoch)
        if gate_sel_all:
            coverage = float(np.mean(gate_sel_all))
            print('Epoch {}  ( ) {:.3f}'.format(epoch, coverage))
            writer.add_scalar('gate_selected_frac', coverage, epoch)

        if deg_terms_all:
            deg_epoch = {}
            numeric_keys = set().union(*(d.keys() for d in deg_terms_all))
            for key in sorted(numeric_keys):
                values = [float(d[key]) for d in deg_terms_all
                          if key in d and isinstance(d[key], (int, float)) and d[key] == d[key]]
                if values:
                    deg_epoch[key] = float(np.mean(values))
            fam_counts = {}
            for d in deg_terms_all:
                family = d.get('deg_family', '?')
                fam_counts[family] = fam_counts.get(family, 0) + 1
            print('Epoch {} DAT: {}   |   {}'.format(
                epoch, len(deg_terms_all), fam_counts))
            print('DAT mean: A dice_deg={:.4f} B cons={:.4f} C sc={:.4f} '
                  'D struct={:.4f} total={:.4f} severity={:.3f}'
                  .format(deg_epoch.get('deg_dice', float('nan')),
                          deg_epoch.get('deg_cons', float('nan')),
                          deg_epoch.get('deg_sc', float('nan')),
                          deg_epoch.get('deg_struct', float('nan')),
                          deg_epoch.get('deg_total', float('nan')),
                          deg_epoch.get('deg_severity', float('nan'))))
            for key, value in deg_epoch.items():
                if value == value:
                    writer.add_scalar('DAT/' + key, value, epoch)

        # ========================
        # Validation / Test
        # ========================
        if epoch % 1 == 0:
            with torch.no_grad():
                for iter_num, data in enumerate(test_loader):
                    model.eval()

                    x = data[0].float().cuda(non_blocking=True)
                    y = data[1].float().cuda(non_blocking=True)
                    x_seg = data[2].float().cuda(non_blocking=True)
                    y_seg = torch.tensor(data[3], dtype=torch.long)

                    warped_x, flow, xlogits, ylogits = model(x, y)

                    def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                    vals0, _ = utils.dice(x_seg.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
                    dice_list0.append(np.average(vals0))

                    vals1, _ = utils.dice(def_out.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
                    dice_list1.append(np.average(vals1))

                    ncc0 = -losses3D.nas_ncc(y, x)
                    ncc1 = -losses3D.nas_ncc(y, warped_x)

                    ncc_list0.append(ncc0.item())
                    ncc_list1.append(ncc1.item())

                    full_mask = F.softmax(ylogits, dim=1).float()
                    mask = full_mask.argmax(dim=1)
                    vals2, _ = utils.dice(mask.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
                    dice_list2.append(np.average(vals2))

            result_dir = os.path.join(args.save_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            f_test = open(result_dir + '/result.txt', 'a')

            print('Epoch', epoch)
            f_test.write('Epoch ' + str(epoch) + '\n')

            print('------------------------Dice------------------------')
            print(' initial :', np.mean(dice_list0), np.std(dice_list0))
            print('   Reg   :', np.mean(dice_list1), np.std(dice_list1))
            print('   Seg   :', np.mean(dice_list2), np.std(dice_list2))

            f_test.write('------------------------Dice------------------------' + '\n')
            f_test.write(' initial :' + str(np.mean(dice_list0)) + str(np.std(dice_list0)) + '\n')
            f_test.write('   Reg   :' + str(np.mean(dice_list1)) + str(np.std(dice_list1)) + '\n')
            f_test.write('   Seg   :' + str(np.mean(dice_list2)) + str(np.std(dice_list2)) + '\n')

            print('------------------------NCC------------------------')
            print(' initial :', np.mean(ncc_list0), np.std(ncc_list0))
            print('  warped :', np.mean(ncc_list1), np.std(ncc_list1))

            f_test.write('------------------------NCC------------------------' + '\n')
            f_test.write('mean:' + str(np.mean(ncc_list0)) + '\n')
            f_test.write('std:' + str(np.std(ncc_list0)) + '\n')
            f_test.write('warped-mean:' + str(np.mean(ncc_list1)) + '\n')
            f_test.write('warped-std:' + str(np.std(ncc_list1)) + '\n')

            writer.add_scalar('Test_Reg_Dice', np.mean(dice_list1), epoch)
            writer.add_scalar('Test_Seg_Dice', np.mean(dice_list2), epoch)
            writer.add_scalar('Test_Ncc', np.mean(ncc_list1), epoch)

            r_dice = np.mean(dice_list1)
            s_dice = np.mean(dice_list2)

            if r_dice > best_rdice:
                best_rdice = r_dice
                if best_rpath is not None:
                    os.remove(best_rpath)
                best_rpath = os.path.join(args.save_path, 'best_reg_%d.ckpt' % epoch)
                torch.save(model.state_dict(), best_rpath)

            if s_dice > best_sdice:
                best_sdice = s_dice
                if best_spath is not None:
                    os.remove(best_spath)
                best_spath = os.path.join(args.save_path, 'best_seg_%d.ckpt' % epoch)
                torch.save(model.state_dict(), best_spath)

            f_test.write('Best Reg dice: {}, Best Reg path: {}\n'.format(best_rdice, best_rpath))
            f_test.write('Best Seg dice: {}, Best Seg path: {}\n'.format(best_sdice, best_spath))
            f_test.close()

        epoch_number = epoch + 1
        epoch_dir = os.path.join(args.save_path, 'checkpoints', 'epochs')
        os.makedirs(epoch_dir, exist_ok=True)
        epoch_path = os.path.join(epoch_dir, 'epoch_{:04d}.ckpt'.format(epoch_number))
        torch.save(model.state_dict(), epoch_path)
        print('Epoch {} complete model weights saved: {}'.format(epoch_number, epoch_path))
        seg_epoch_path = os.path.join(
            epoch_dir, 'seg_epoch_{:04d}.ckpt'.format(epoch_number)
        )
        torch.save(segmentation_state_dict(model), seg_epoch_path)
        print('Epoch {} segmentation weights saved: {}'.format(
            epoch_number, seg_epoch_path
        ))
        if aligner is not None:
            aligner_path = os.path.join(
                epoch_dir, 'medsam2_aligner_epoch_{:04d}.pt'.format(epoch_number)
            )
            torch.save(aligner.trainable_state_dict(), aligner_path)
            print('Epoch {} MedSAM2 aligner weights saved: {}'.format(
                epoch_number, aligner_path
            ))

    if aligner is not None:
        align_path = os.path.join(args.save_path, 'medsam2_aligner_proj.pt')
        torch.save(aligner.trainable_state_dict(), align_path)
        print('MedSAM2 aligner projection saved to {}'.format(align_path))

    if deg_aux is not None:
        deg_aux.close()

    writer.close()
    terminal = logger.terminal
    sys.stdout = terminal
    sys.stderr = terminal
    logger.close()

if __name__ == '__main__':
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    if GPU_num:
        torch.cuda.set_device(0)
        print('Currently using: ' + torch.cuda.get_device_name(0))
    else:
        print('CUDA is unavailable; use --gpu -1 for CPU execution.')

    main()
