from argparse import Namespace as Args
from collections import namedtuple
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from models.plm import get_model
from models.utils import is_wandb_running

AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
rAA = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


MODEL_PKA = {
    # Table 1: from PropKa 3: https://github.com/jensengroup/propka-3.0/blob/master/Source/parameters.py
    'ARG': 12.5,
    'GLU': 4.5,
    'LYS': 10.5,
    'CYS': 9.0,
    'HIS': 6.5,
    'ASP': 3.8,
    'TYR': 10.0,
    'CTR': 3.2,
    'NTR': 8.0,
    # Table 2: we use it because it has a reference.
    # 'ARG': 13.5,
    # 'GLU': 4.4,
    # 'LYS': 10.4,
    # 'CYS': 8.3,
    # 'HIS': 6.8,
    # 'ASP': 4.0,
    # 'TYR': 9.6,
    # 'CTR': 3.0,
    # 'NTR': 8.0,
}


CHARGE = {
    'ARG': 1,
    'HIS': 1,
    'LYS': 1,
    'CYS': -1,
    'TYR': -1,
    'GLU': -1,
    'ASP': -1,
    'CTR': -1,
    'NTR': 1,
}

res_to_idx = {
    'ASP': 0, 'GLU': 1, 'HIS': 2, 'LYS': 3, 'CYS': 4, 'TYR': 5,
    'CTR': 6, 'NTR': 7, 'ARG': 8,
}
idx_to_res = {v: k for k, v in res_to_idx.items()}


pKa_Data = namedtuple(
    'pKa_Data', ['pKa', 'res_id', 'seq_id', 'res_name', 'seq', 'esm_feats', 'ss', 'buried', 'exposed'], defaults=[None]*9)
pI_Data = namedtuple(
    'pI_Data', ['pI', 'seq', 'esm_feats', 'mask'], defaults=[None]*4)
SS_Data = namedtuple(
    'SS_Data', ['seq_id', 'seq', 'ss', 'esm_feats'], defaults=[None]*4)
aa_str = 'ACDEFGHIKLMNPQRSTVWYX'

aa_to_idx = {aa: i for i, aa in enumerate(aa_str)}
idx_to_aa = {i: aa for i, aa in enumerate(aa_str)}

aa_pka = 'RHKDECY'
q8_str = 'GHIBESTC'


class pKaDataloader(LightningDataModule):
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args

        self.pka_train = pKaDataset(
            args.pka_train_csv, args.pka_plm)
        self.pka_test = pKaDataset(
            args.pka_test_csv, args.pka_plm)

        self.steps_per_epoch = len(self.train_dataloader())

        if args.kfold == 1:
            self.pka_val = self.pka_test

        self.kfold = None

    def setup_kfold(self, i):
        if self.kfold is None:
            self.kfold = KFold(
                n_splits=self.args.kfold,
                shuffle=True, random_state=self.args.seed)
            self.old_pka_train = self.pka_train

        train_idx, val_idx = list(self.kfold.split(self.old_pka_train))[i]
        self.pka_val = Subset(self.old_pka_train, val_idx)
        self.pka_train = Subset(self.old_pka_train, train_idx)

    def train_dataloader(self):
        return DataLoader(
            self.pka_train,
            batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.pka_val, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(
            self.pka_test, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # xs, seqs, resids, restypes, pkas, buried
        return [(
            d.esm_feats[..., 1],
            d.seq,
            d.res_id,
            res_to_idx[d.res_name],
            d.pKa,
            d.buried,
            d.exposed,
        ) for d in batch]


class pKaDataset(InMemoryDataset):
    def __init__(self, pka_csv: str, pka_plm: str):
        self.pka_csv = pka_csv
        self.pka_plm = pka_plm
        super().__init__(root='data')
        self.data_list = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{Path(self.pka_csv).stem}_{self.pka_plm.replace("/", "_")}.pt']

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

    def process(self):
        pka_df = pd.read_csv(self.pka_csv)

        if self.pka_plm != 'none':
            pka_plm_func = get_model(self.pka_plm, 'cuda')
        self.data_list = []
        for i, row in tqdm(pka_df.iterrows(), total=len(pka_df), leave=False):
            pka, resid, seqid, seq, buried, exposed = row
            if self.pka_plm == 'none':
                f_pka = torch.zeros(1, 2)
            else:
                f_pka = pka_plm_func([seq]).half().to('cpu')

            self.data_list.append(pKa_Data(
                pKa=pka,
                res_id=resid,
                res_name=seqid.split('_')[-1],
                seq_id=seqid,
                seq=seq,
                esm_feats=f_pka,
                buried=buried,
                exposed=exposed,
            ))
        torch.save(self.data_list, self.processed_paths[0])


class pIDataloader(LightningDataModule):
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args

        self.pI_train = pIDataset(args.pi_train_csv)
        self.pI_test = pIDataset(args.pi_test_csv)

        self.steps_per_epoch = len(self.train_dataloader())

        if args.kfold == 1:
            self.pI_val = self.pI_test

        self.kfold = None

    def setup_kfold(self, i):
        if self.kfold is None:
            self.kfold = KFold(
                n_splits=self.args.kfold,
                shuffle=True, random_state=self.args.seed)
            self.old_pI_train = self.pI_train

        train_idx, val_idx = list(self.kfold.split(self.old_pI_train))[i]
        self.pI_val = Subset(self.old_pI_train, val_idx)
        self.pI_train = Subset(self.old_pI_train, train_idx)

    def train_dataloader(self):
        return DataLoader(
            self.pI_train,
            batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.pI_val, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(
            self.pI_test, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        return [(
            d.seq,
            d.mask,
            d.pI) for d in batch]


class pIDataset(InMemoryDataset):
    def __init__(self, csvfile: str, plm_name: str = ''):
        self.csvfile = csvfile
        self.plm_name = plm_name
        super().__init__(root='data')
        logger.info(f'Loading {self.processed_paths[0]}')
        self.data_list = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{Path(self.csvfile).stem}.pt']

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

    def process(self):
        df = pd.read_csv(self.csvfile)
        self.data_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), leave=False):
            pI, seq = row
            mask = torch.zeros(len(seq), dtype=torch.bool)
            for j, aa in enumerate(seq):
                mask[j] = aa in aa_pka
            mask[0], mask[-1] = True, True
            self.data_list.append(pI_Data(
                pI=pI,
                seq=seq,
                mask=mask,
            ))
        torch.save(self.data_list, self.processed_paths[0])
