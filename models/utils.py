
import os
from argparse import Namespace as Args
from typing import List

import numpy as np
import torch
import yaml

MODEL_PKA = {
    # Table 2: we use it because it has a reference.
    'ARG': 13.5,
    'GLU': 4.4,
    'LYS': 10.4,
    'CYS': 8.3,
    'HIS': 6.8,
    'ASP': 4.0,
    'TYR': 9.6,
    'CTR': 3.0,
    'NTR': 8.0,
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config(object):
    def __init__(self, config_file):
        self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    def __getattr__(self, name):
        return self.config[name]

    def __str__(self):
        return str(self.config)


def gen_ckpt_path(args: Args, attrs: list):
    """ Generate checkpoint path based on args and attrs """
    p = ''
    for attr in attrs:
        p += f'{attr}={getattr(args, attr)}_'
    return p[:-1]


def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def num_params(model: torch.nn.Module):
    s = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = s / 1024 / 1024
    return f'{s} ({mb:.2f} MB)'


def is_wandb_running():
    """ Check if wandb is running """
    return "WANDB_SWEEP_ID" in os.environ


def get_loss_func(name):
    if name == 'mse':
        return torch.nn.MSELoss()
    if name == 'mse_w':
        return torch.nn.MSELoss(reduction='none')
    elif name == 'mae':
        return torch.nn.L1Loss()
    elif name == 'smooth_l1':
        return torch.nn.SmoothL1Loss(beta=0.5)
    elif name == 'mse_y':
        return lambda pred, target: torch.mean(
            torch.nn.MSELoss(reduction='none')(pred, target) * target.abs())
    elif name == 'mse_y2':
        return lambda pred, target: torch.mean(
            torch.nn.MSELoss(reduction='none')(pred, target) * target**2)
    elif name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown loss function: {name}')


class pKaMetric(object):
    def __init__(self):
        self.preds, self.targets, self.resnames, \
            self.buried, self.exposed = [], [], [], [], []

    def clear(self):
        self.preds, self.targets, self.resnames, \
            self.buried, self.exposed = [], [], [], [], []

    def add(self, pred: torch.Tensor, target: torch.Tensor, resnames: List[str], buried: torch.Tensor, exposed: torch.Tensor):
        self.preds.append(pred.cpu().numpy())
        self.targets.append(target.cpu().numpy())
        self.resnames.extend(resnames)
        self.buried.extend(buried)
        self.exposed.extend(exposed)

    @ staticmethod
    def get_indices(all_lists, res_names):
        indices = []
        for i, l in enumerate(all_lists):
            if l in res_names:
                indices.append(i)
        return np.array(indices)

    def compute(self):
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        burieds = np.array(self.buried)
        exposeds = np.array(self.exposed)

        # calculate metrics
        major_idx = self.get_indices(
            self.resnames, ['ASP', 'GLU', 'HIS', 'LYS'])
        major_rmse = calc_rmse(targets[major_idx], preds[major_idx])
        major_mae = calc_mae(targets[major_idx], preds[major_idx])
        rmse = calc_rmse(targets, preds)
        mae = calc_mae(targets, preds)
        metrics = {
            'val_major_rmse': major_rmse,
            'val_major_mae': major_mae,
            'val_rmse': rmse,
            'val_mae': mae
        }
        for res_name in MODEL_PKA:
            idx = self.get_indices(self.resnames, [res_name])
            if len(idx) == 0:
                continue
            rmse = calc_rmse(targets[idx], preds[idx])
            mae = calc_mae(targets[idx], preds[idx])
            metrics[f'val_{res_name}_rmse'] = rmse
            metrics[f'val_{res_name}_mae'] = mae
        shift = targets - np.array([MODEL_PKA[k] for k in self.resnames])
        for a, b in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 100)]:
            idx = np.where((shift >= a) & (shift < b))[0]
            if len(idx) == 0:
                continue
            rmse = calc_rmse(targets[idx], preds[idx])
            mae = calc_mae(targets[idx], preds[idx])
            metrics[f'val_{a}_{b}_rmse'] = rmse
            metrics[f'val_{a}_{b}_mae'] = mae
        # compute buried metrics
        rmse = calc_rmse(targets[burieds], preds[burieds])
        mae = calc_mae(targets[burieds], preds[burieds])
        metrics['val_buried_rmse'] = rmse
        metrics['val_buried_mae'] = mae
        rmse = calc_rmse(targets[exposeds], preds[exposeds])
        mae = calc_mae(targets[exposeds], preds[exposeds])
        metrics['val_exposed_rmse'] = rmse
        metrics['val_exposed_mae'] = mae
        metrics = dict(sorted(metrics.items()))
        return metrics


class pIMetric(object):
    def __init__(self):
        self.preds, self.targets = [], []

    def clear(self):
        self.preds, self.targets = [], []

    def add(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds.append(pred.cpu().numpy())
        self.targets.append(target.cpu().numpy())

    def compute(self):
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        rmse = calc_rmse(targets, preds)
        mae = calc_mae(targets, preds)
        R2 = 1 - np.sum((targets - preds)**2) / \
            np.sum((targets - np.mean(targets))**2)
        return {'val_rmse': rmse, 'val_mae': mae, 'val_R2': R2}


class SSMetric(object):
    def __init__(self):
        self.preds, self.targets = [], []

    def clear(self):
        self.preds, self.targets = [], []

    def add(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds.append(pred.cpu().numpy())
        self.targets.append(target.cpu().numpy())

    def compute(self):
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        acc = np.mean(np.argmax(preds, axis=1) == targets)
        return {'val_acc': acc}


class HendersonHasselbalch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pka: torch.Tensor, ph, charge: torch.Tensor):
        return torch.mean(charge / (1 + 10 ** (charge * (ph - pka))), dim=-1)


def solve_hh(pka: torch.Tensor,  charge: torch.Tensor, eps=1e-4):
    min_ph = pka.min()
    max_ph = pka.max()
    # bisection method
    hh = HendersonHasselbalch()
    while max_ph - min_ph > eps:
        mid_ph = (min_ph + max_ph) / 2
        result = hh(pka, mid_ph, charge)
        if result > 0:
            min_ph = mid_ph
        else:
            max_ph = mid_ph

    return mid_ph
