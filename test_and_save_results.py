from ast import arg
import string
from torch.utils.tensorboard import SummaryWriter
import os, glob, losses3D
import sys
import argparse
from torch.utils.data import DataLoader
from data import datasets_mixed as datasets
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models_hyper import Architecture_hyper
from models import Architecture
# from models2 import *
import utils.utils as utils
import torch.nn.functional as F
import utils.augmentation as aug
import random
from datetime import datetime
import copy
import time
import nibabel as nib
from pathlib import Path
import json

shape = (128, 128, 128)
torch.set_printoptions(precision=8)

# ============================================================================
# ============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_VALUES = [0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55]   #

RUN_CONFIG = dict(
    seg_ckpt='',
    reg_ckpt='',
    save_path='',
    gpu=-1,                    #
    max_cases=0,
    min_free_gb=6.0,
    save_nifti=True,
    test_path='',
    src_path='',
    sseg_path='',
    train_path='',
)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "/logfile_nopretrain.log", "a", encoding='utf-8', errors='replace')
        print('save vvvv', save_dir)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()          #

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def save_nifti(data, affine, save_path):
    """
     NIfTI 
    
     :
    - data:  
    - affine:  
    - save_path:  
    """
    data = data.astype(np.int32)  #
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, save_path)
    print(f'  NIfTI   {save_path}')

def save_native_labels(indices, reference_path, save_path):
    """Undo the dataset's index-grid resize and restore original label IDs/geometry."""
    from scipy import ndimage
    reference = nib.load(reference_path)
    label_values = np.asarray([0, 1, 11, 22, 33, 44, 46, 47, 48, 49, 50, 51, 52, 55])
    indices = np.asarray(indices).round().astype(np.int64)
    if indices.min() < 0 or indices.max() >= len(label_values):
        raise ValueError('Prediction contains invalid class indices')
    resized = ndimage.zoom(indices, tuple(n / m for n, m in zip(reference.shape[:3], indices.shape)),
                           order=0, grid_mode=False)
    if resized.shape != reference.shape[:3]:
        raise ValueError('Restored prediction shape does not match reference')
    image = nib.Nifti1Image(label_values[resized].astype(np.int16), reference.affine, reference.header.copy())
    image.set_data_dtype(np.int16)
    nib.save(image, save_path)

def find_latest_ckpt(spec):
    """
     
      'auto' /   /   /    ** ** 
       
    """
    def _newest(paths, what):
        cands = [(os.path.getmtime(p), p) for p in paths if os.path.isfile(p)]
        if not cands:
            return None
        cands.sort()
        return cands[-1][1]

    if not spec or str(spec).lower() == 'auto':
        raise SystemExit('Specify --seg_ckpt and --reg_ckpt explicitly.')

    spec = str(spec)
    if not os.path.isabs(spec):
        spec = os.path.join(_HERE, spec)

    if os.path.isdir(spec):
        for pat in ('best_seg_*.ckpt', 'best_reg_*.ckpt', '*.ckpt'):
            got = _newest(glob.glob(os.path.join(spec, pat)), pat)
            if got:
                return got
        raise SystemExit('  *.ckpt: %s' % spec)

    if '*' in spec:
        got = _newest(glob.glob(spec), spec)
        if got:
            return got
        raise SystemExit(' : %s' % spec)

    if not os.path.isfile(spec):
        raise SystemExit(' : %s' % spec)
    return spec

def dice_per_class(pred, target, num_classes=14):
    """  Dice  nan """
    pred = np.asarray(pred).astype(np.int32)
    target = np.asarray(target).astype(np.int32)
    out = []
    for cls in range(1, num_classes):
        if np.sum(target == cls) == 0:
            out.append(np.nan)
            continue
        inter = np.sum((pred == cls) & (target == cls))
        union = np.sum(pred == cls) + np.sum(target == cls)
        out.append(2.0 * inter / (union + 1e-5))
    return out

def nanmean_rows(rows, n):
    a = np.asarray(rows, dtype=np.float64)
    if a.size == 0:
        return np.full(n, np.nan)
    cnt = np.sum(~np.isnan(a), axis=0)
    return np.where(cnt > 0, np.nansum(a, axis=0) / np.maximum(cnt, 1), np.nan)

def load_evaluation_model(ckpt_path, device):
    pretrained_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    pretrained_dict = {key.removeprefix('module.'): value
                       for key, value in pretrained_dict.items()}
    if 'Reg_Head.flow.weight' in pretrained_dict:
        use_moe = any(key.startswith('pw1.') for key in pretrained_dict)
        use_mrb = any(key.startswith('reg_mrb.') for key in pretrained_dict)
        model = Architecture(inshape=shape, n_channels=1, n_classes=14,
                             use_moe=use_moe, use_mrb=use_mrb).to(device)
        print('Checkpoint architecture: Architecture (MoE=%s, MRB=%s)' % (use_moe, use_mrb))
    elif 'Reg_Decoder.flow.weight' in pretrained_dict:
        model = Architecture_hyper(inshape=shape, n_channels=1, n_classes=14).to(device)
        print('Checkpoint architecture: Architecture_hyper')
    else:
        raise ValueError('  checkpoint   Reg_Head.flow.weight   Reg_Decoder.flow.weight')
    model.load_state_dict(pretrained_dict, strict=True)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(
        description='  checkpoint  Dice/NCC  NIfTI  Dice '
                    '  RUN_CONFIG  Run  ')
    parser.add_argument('--local_ep', type=int, default=RUN_CONFIG.get('local_ep', 1),
                        help="the number of local epochs: E")
    parser.add_argument('--save_path', type=str, default=RUN_CONFIG['save_path'],
                        help=" '' =   results_eval/< >")
    parser.add_argument('--epochs', type=int, default=200, help="the number of global epochs: G")
    parser.add_argument('--gpu', type=int, default=RUN_CONFIG['gpu'], help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    parser.add_argument('--seg_ckpt', default=RUN_CONFIG['seg_ckpt'], help=' ')
    parser.add_argument('--reg_ckpt', default=RUN_CONFIG['reg_ckpt'], help=' ')
    parser.add_argument('--test_path', type=str, default=RUN_CONFIG['test_path'])
    parser.add_argument('--src_path', type=str, default=RUN_CONFIG['src_path'])
    parser.add_argument('--sseg_path', type=str, default=RUN_CONFIG['sseg_path'])
    parser.add_argument('--train_path', type=str, default=RUN_CONFIG['train_path'])
    parser.add_argument('--save_nifti', type=int, default=int(bool(RUN_CONFIG['save_nifti'])),
                        help='1=  NIfTI 0= ')
    parser.add_argument('--max_cases', type=int, default=int(RUN_CONFIG.get('max_cases', 0)),
                        help='  N  0= ')
    parser.add_argument('--force', type=int, default=0,
                        help='1= ')
    args = parser.parse_args()
    for key in ('seg_ckpt', 'reg_ckpt', 'test_path', 'src_path', 'sseg_path'):
        if not getattr(args, key):
            parser.error(f'--{key} is required')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    seg_ckpt_path = find_latest_ckpt(args.seg_ckpt)
    reg_ckpt_path = find_latest_ckpt(args.reg_ckpt)
    if not args.save_path:
        stem = Path(seg_ckpt_path).stem + '__' + Path(reg_ckpt_path).stem
        args.save_path = os.path.join(_HERE, 'results_eval', stem)
    print('=' * 90)
    print('  RUN_CONFIG ')
    print('    : %s' % seg_ckpt_path)
    print('    : %s' % reg_ckpt_path)
    print('    : %s' % args.save_path)
    print('        : %s' % args.device)
    print('      : %s' % args.test_path)
    print('   NIfTI: %s' % bool(args.save_nifti))
    print('=' * 90)

    if args.device.type == 'cuda':
        _free_b, _total_b = torch.cuda.mem_get_info(args.device)
        _free_gb = _free_b / 1024 ** 3
        print('GPU  : %.2f GB / %.2f GB' % (_free_gb, _total_b / 1024 ** 3))
        if _free_gb < float(RUN_CONFIG['min_free_gb']) and not int(args.force):
            raise SystemExit(
                '\n  %.2f GB  %.1f GB  2~4 GB \n'
                '    OOM    \n'
                '   \n'
                '    1)  GPU   80   2  \n'
                '    2)   RUN_CONFIG["gpu"]   -1   CPU   15  \n'
                '    3)   --force 1 \n'
                % (_free_gb, float(RUN_CONFIG['min_free_gb'])))

    batch_size = 1
    best_sdice = 0
    best_rdice = 0
    best_spath = None
    best_rpath = None

    model_dir = args.save_path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    sys.stdout = Logger(args.save_path)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 200 #max traning epoch
    cont_training = False #if continue training
    img_size = shape

    '''
    Initialize model
    '''
    seg_network = load_evaluation_model(seg_ckpt_path, args.device)
    reg_network = load_evaluation_model(reg_ckpt_path, args.device)
    Path(args.save_path, 'evaluation_config.json').write_text(
        json.dumps(dict(seg_checkpoint=seg_ckpt_path, reg_checkpoint=reg_ckpt_path,
                        test_path=args.test_path), ensure_ascii=False, indent=2), encoding='utf-8')
    reg_model = utils.register_model('nearest')
    reg_model.to(args.device)
    reg_model_bilin = utils.register_model('bilinear')
    reg_model_bilin.to(args.device)

    '''
    If continue from previous training
    '''
    # if cont_training:
    #     epoch_start = 25
    #     best_model = torch.load(model_path)
    #     print('Model: {} loaded!'.format(model_path))
    #     model.load_state_dict(best_model)

    from types import SimpleNamespace
    data_args = SimpleNamespace(
        train_path=args.train_path,
        test_path=args.test_path,       #
        src_path=args.src_path,
        sseg_path=args.sseg_path,
    )
    val_set = datasets.InferDataset(data_args)
    if len(val_set) == 0:
        raise RuntimeError('  RUN_CONFIG["test_path"]: %s' % args.test_path)
    if args.max_cases and args.max_cases > 0 and args.max_cases < len(val_set):
        from torch.utils.data import Subset
        eval_index = list(range(int(args.max_cases)))
        val_eval_set = Subset(val_set, eval_index)
        print('  %d   %d  ' % (args.max_cases, len(val_set)))
    else:
        eval_index = list(range(len(val_set)))
        val_eval_set = val_set
    val_loader = DataLoader(
        val_eval_set, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=(args.device.type == 'cuda'), drop_last=False)

    dice_list0, dice_list1, dice_list2, ncc_list0, ncc_list1, ncc_list2 = [], [], [], [], [], []
    print('Evaluation Starts')
    '''
    Training
    '''
    rreg_all, rgrad_all, rcycle_all = [], [], []
    rdice_RegSeg_all = []
    rtotalloss_all = []

    sdice_RegSeg_all = []

    batch_loss = []
    step = 0
    # for iter, (data) in enumerate(train_loader):
    #     step = step+1
    #     # images, labels = images.to(self.args.device), labels.to(self.args.device)
    #     data = [torch.as_tensor(t, dtype=torch.float32, device=args.device) for t in data]
    #     y = data[1].to(args.device)
    #     x_seg = data[2].to(args.device)

    #     spatial_aug = aug.SpatialTransform(do_rotation=True,
    #                     angle_x=(-np.pi / 18, np.pi / 18),
    #                     angle_y=(-np.pi / 18, np.pi / 18),
    #                     angle_z=(-np.pi / 18, np.pi / 18),
    #                     do_scale=True,
    #                     scale=(0.9, 1.1))

    #     code_spa = spatial_aug.rand_coords(x.shape[2:])
    #     x = spatial_aug.augment_spatial(x, code_spa)
    #     x_seg = spatial_aug.augment_spatial(x_seg, code_spa, mode='nearest')
    #     y = spatial_aug.augment_spatial(y, code_spa)

    #     model.train()

    #     x_y, flow, x_logits, y_logits = model(x, y)
    #     # y_x, flow_y_x = model.Reg_Decoder(y, encoded_y, encoded_x)

    #     loss, ncc, grad, dice_seg = reg_criterion(y, x_y, y_logits, x_seg, flow)

    #     # loss, ncc, grad, dice_seg = self.loss_func(y, x2y, y_logits, x_seg, flow)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if iter % 10 == 0:
    #         print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'Train Reg_Decoder {} Iter {} of {} dice_loss {:.8f} ncc {:.8f} grad_loss {:.8f}'
    #                 .format(epoch, step, len(train_loader), dice_seg.item(), ncc.item(), grad.item()))
    #         # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Train Seg_decoder {} Iter {} of {} dice_seg {:.8f} '
    #         #         .format(epoch, iter, len(train_loader), dice_RegSeg.item()))
    #         print('total loss {:.8f}'.format(loss.item()))
    # writer.add_scalar('toalloss', np.mean(rdice_RegSeg_all), epoch)

    # # copy weight to net_glob
    # # net_glob.load_state_dict(w_glob)

    # writer.add_scalar('toalloss', np.mean(rdice_RegSeg_all), epoch)
    seg_save_dir = os.path.join(args.save_path, 'seg')
    reg_save_dir = os.path.join(args.save_path, 'reg')
    '''
    Validation
    '''
    if args.save_nifti:
        os.makedirs(seg_save_dir, exist_ok=True)
        os.makedirs(reg_save_dir, exist_ok=True)
    per_class_seg, per_class_reg = [], []      #
    t1 = time.time()
    with torch.inference_mode():
        for idx, data in enumerate(val_loader):
            case_name = os.path.basename(val_set.paths[eval_index[idx]])
            case_name = case_name.replace('.nii.gz', '').replace('.nii', '')
            data = [torch.as_tensor(t, dtype=torch.float32, device=args.device) for t in data]
            x = data[0]   #
            y = data[1]
            x_seg = data[2]
            y_seg = data[3].to(dtype=torch.long)
            warped_x, flow, _, _ = reg_network(x, y)
            _, _, _, ylogits = seg_network(x, y)

            def_out = reg_model([x_seg.to(args.device).float(), flow.to(args.device)])
            vals0, _ = utils.dice(x_seg.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
            dice_list0.append(np.average(vals0))
            vals1, _ = utils.dice(def_out.cpu().numpy(), y_seg.cpu().numpy(), nargout=2)
            dice_list1.append(np.average(vals1))
            # print(np.average(vals0), np.average(vals1))

            ncc0 = -losses3D.nas_ncc(y, x)
            # warped
            ncc1 = -losses3D.nas_ncc(y, warped_x)

            ncc_list0.append(ncc0.item())
            ncc_list1.append(ncc1.item())

            full_mask = F.softmax(ylogits, dim=1).float()
            mask = full_mask.argmax(dim=1)
            vals2, _ = utils.dice(mask.cpu().numpy().squeeze(), y_seg.cpu().numpy().squeeze(), nargout=2)
            dice_list2.append(np.average(vals2))

            _gt_np = y_seg.cpu().numpy().squeeze()
            per_class_seg.append(dice_per_class(mask.cpu().numpy().squeeze(), _gt_np, 14))
            per_class_reg.append(dice_per_class(def_out.cpu().numpy().squeeze(), _gt_np, 14))

            if args.save_nifti:
                save_native_labels(mask.squeeze().cpu().numpy(), val_set.paths[eval_index[idx]],
                                   os.path.join(seg_save_dir, case_name + '_seg.nii.gz'))

                save_native_labels(def_out.squeeze().cpu().numpy(), val_set.paths[eval_index[idx]],
                                   os.path.join(reg_save_dir, case_name + '_warpedseg.nii.gz'))
            if idx % 10 == 0:
                print('  [%3d/%3d] %s  seg=%.4f reg=%.4f    %.1fs'
                      % (idx + 1, len(val_loader), case_name,
                         np.average(vals2), np.average(vals1), time.time() - t1))

            # print(np.mean(ncc0.item()), np.mean(ncc1.item()))
        t2 = time.time()
        avg_time = (t2-t1)/len(val_loader)
        result_dir = os.path.join(args.save_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        f_test = open(result_dir + '/result-1.txt', 'a')

        # print('Epoch', epoch)
        # f_test.write('Epoch '+str(epoch)+'\n')
        epoch = 1
        print('------------------------Dice------------------------')
        print(' initial :', np.mean(dice_list0), np.std(dice_list0))
        print('   Reg   :', np.mean(dice_list1), np.std(dice_list1))
        print('   Seg   :', np.mean(dice_list2), np.std(dice_list2))
        f_test.write('------------------------Dice------------------------' + '\n')
        f_test.write(' initial :' + str(np.mean(dice_list0)) + str(np.std(dice_list0)) + '\n')
        f_test.write('   Reg   :' + str(np.mean(dice_list1)) + str(np.std(dice_list1)) +'\n')
        f_test.write('   Seg   :' + str(np.mean(dice_list2)) + str(np.std(dice_list2)) +'\n')

        _n = len(LABEL_VALUES) - 1
        pc_seg = nanmean_rows(per_class_seg, _n)
        pc_reg = nanmean_rows(per_class_reg, _n)
        _names = ['label%d' % v for v in LABEL_VALUES[1:]]
        _lseg = ' '.join('%s=%.4f' % (nm, v) for nm, v in zip(_names, pc_seg))
        _lreg = ' '.join('%s=%.4f' % (nm, v) for nm, v in zip(_names, pc_reg))
        print('---------------------Per-class Dice---------------------')
        print(' Seg:', _lseg)
        print(' Reg:', _lreg)
        _worst = np.argsort(np.nan_to_num(pc_seg, nan=9.9))[:5]
        print(' >>> Seg   5  : ' + ', '.join('%s=%.4f' % (_names[i], pc_seg[i]) for i in _worst))
        f_test.write(' per-class Seg: ' + _lseg + '\n')
        f_test.write(' per-class Reg: ' + _lreg + '\n')
        with open(os.path.join(result_dir, 'eval_per_class.csv'), 'w', encoding='utf-8') as _pc:
            _pc.write('metric,' + ','.join(_names) + '\n')
            _pc.write('seg,' + ','.join('%.6f' % v for v in pc_seg) + '\n')
            _pc.write('reg,' + ','.join('%.6f' % v for v in pc_reg) + '\n')
        print(' : %s' % os.path.join(result_dir, 'eval_per_class.csv'))

        print('------------------------NCC------------------------')
        print(' initial :', np.mean(ncc_list0), np.std(ncc_list0))
        print('  warped :', np.mean(ncc_list1), np.std(ncc_list1))
        f_test.write('------------------------NCC------------------------' + '\n')
        f_test.write('mean:' + str(np.mean(ncc_list0)) + '\n')
        f_test.write('std:' + str(np.std(ncc_list0)) + '\n')
        f_test.write('warped-mean:' + str(np.mean(ncc_list1)) + '\n')
        f_test.write('warped-std:' + str(np.std(ncc_list1)) + '\n')

        r_dice = np.mean(dice_list1)
        s_dice = np.mean(dice_list2)
        f_test.write('Seg checkpoint: %s\nReg checkpoint: %s\n' % (seg_ckpt_path, reg_ckpt_path))
        f_test.write('Reg dice: %s\n' % r_dice)
        f_test.write('Seg dice: %s\n' % s_dice)
        f_test.flush()          #
        f_test.close()

    _logger = sys.stdout
    if hasattr(_logger, 'terminal'):
        sys.stdout = _logger.terminal
        sys.stderr = _logger.terminal
        _logger.close()
    print('=' * 90)
    print(' : %s' % args.save_path)
    print('   : %s' % os.path.join(args.save_path, 'logfile_nopretrain.log'))
    print('   : %s' % os.path.join(args.save_path, 'result-1.txt'))
    print('   : %s' % os.path.join(args.save_path, 'eval_per_class.csv'))
    if args.save_nifti:
        print('  Seg NIfTI: %s' % seg_save_dir)
        print('  Reg NIfTI: %s' % reg_save_dir)
    print('=' * 90)

# def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

if __name__ == '__main__':
    main()
