from argparse import Namespace as Args
from typing import List

import lightning as pl
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from models.dataset import MODEL_PKA, aa_to_idx, idx_to_res
from models.module import get_rnn_network
from models.plm import EsmModelInfo
from models.utils import get_loss_func, pIMetric, pKaMetric


class pIPeptide(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.res_emb_pipep = torch.nn.Embedding(21, args.pipep_dim)
        self.rnn_pipep = get_rnn_network(
            rnn_type=args.rnn_type,
            in_dim=args.pipep_dim,
            out_dim=args.pipep_dim,
            num_layers=args.pipep_layers,
        )

        self.out_pipep = torch.nn.Linear(args.pipep_dim, 1)

    def forward(self, x: List[torch.Tensor], masks: List[torch.Tensor]):
        """_summary_

        Args:
            x (List[torch.Tensor]): List of PLM features
            masks (List[torch.Tensor]): List of masks for titrable residues
        """
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        x = self.res_emb_pipep(x)
        x = self.rnn_pipep(x)
        x = torch.stack([
            torch.mean(xi[mask], dim=0)
            for xi, mask in zip(x, masks)])
        x = self.out_pipep(x).squeeze(-1)
        return x


class pl_pIPeptide(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.model = pIPeptide(args)
        self.loss_func = get_loss_func(args.loss)
        self.automatic_optimization = False
        self.metric = pIMetric()

    def forward(self, *args):
        return self.model(*args)

    def _step(self, batch):
        seqs, masks, pIs = zip(*batch)
        pIs = torch.tensor(pIs, dtype=torch.float32, device=self.device)
        xs = [torch.tensor(
            [aa_to_idx.get(aa, 20) for aa in seq], dtype=torch.long, device=self.device)
            for seq in seqs]
        pred = self.model(xs, masks)
        loss = self.loss_func(pred, pIs)
        return loss, pred, pIs

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step(batch)
        self.log('train/pI_loss', loss, prog_bar=True)

        self.manual_backward(loss)
        self.optimizers().step()
        if self.args.lr_sch:
            self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, pred, target = self._step(batch)
            self.metric.add(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metric.compute(), prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.metric.clear()

    def configure_optimizers(self):
        opt = (
            torch.optim.AdamW
            if self.args.optimizer == 'adamw' else torch.optim.Adam
            if self.args.optimizer == 'adam' else torch.optim.SGD)(
            self.parameters(),
            lr=self.args.lr, weight_decay=self.args.wd)
        if self.args.lr_sch:
            sch = CosineLRScheduler(opt, t_initial=self.args.t_initial, lr_min=1e-6,
                                    warmup_t=round(self.args.wamrup_ratio * self.args.t_initial))
            return [opt], [sch]
        else:
            return opt


class pIProtein(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        self.res_emb_piprot = torch.nn.Embedding(21, args.piprot_dim)

        self.rnn_piprot = get_rnn_network(
            rnn_type=args.rnn_type,
            in_dim=args.piprot_dim,
            out_dim=args.piprot_dim,
            num_layers=args.piprot_layers,
        )

        self.out_piprot = torch.nn.Linear(args.piprot_dim, 1)

    def forward(self, x: List[torch.Tensor], masks: List[torch.Tensor]):
        """_summary_

        Args:
            x (List[torch.Tensor]): List of PLM features
            masks (List[torch.Tensor]):
        """
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        x = self.res_emb_piprot(x)
        x = self.rnn_piprot(x)
        x = torch.stack([
            torch.mean(xi[mask], dim=0)
            for xi, mask in zip(x, masks)])
        x = self.out_piprot(x).squeeze(-1)
        return x


class pl_pIProtein(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.model = pIProtein(args)
        self.loss_func = get_loss_func(args.loss)
        self.automatic_optimization = False
        self.metric = pIMetric()

    def forward(self, *args):
        return self.model(*args)

    def _step(self, batch):
        seqs, masks, pIs = zip(*batch)
        pIs = torch.tensor(pIs, dtype=torch.float32, device=self.device)
        xs = [torch.tensor(
            [aa_to_idx.get(aa, 20) for aa in seq], dtype=torch.long, device=self.device)
            for seq in seqs]
        pred = self.model(xs, masks)
        loss = self.loss_func(pred, pIs)
        return loss, pred, pIs

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step(batch)
        self.log('train/pI_loss', loss, prog_bar=True)

        self.manual_backward(loss)
        self.optimizers().step()
        if self.args.lr_sch:
            self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, pred, target = self._step(batch)
            self.metric.add(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metric.compute(), prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.metric.clear()

    def configure_optimizers(self):
        opt = (
            torch.optim.AdamW
            if self.args.optimizer == 'adamw' else torch.optim.Adam
            if self.args.optimizer == 'adam' else torch.optim.SGD)(
            self.parameters(),
            lr=self.args.lr, weight_decay=self.args.wd)
        if self.args.lr_sch:
            sch = CosineLRScheduler(opt, t_initial=self.args.t_initial, lr_min=1e-6,
                                    warmup_t=round(self.args.wamrup_ratio * self.args.t_initial))
            return [opt], [sch]
        else:
            return opt


class pKALM(torch.nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.res_emb_pipep = torch.nn.Embedding(21, args.pipep_dim)
        self.rnn_pipep = get_rnn_network(
            rnn_type=args.rnn_type,
            in_dim=args.pipep_dim,
            out_dim=args.pipep_dim,
            num_layers=args.pipep_layers,
        )

        self.res_emb_piprot = torch.nn.Embedding(21, args.piprot_dim)
        self.rnn_piprot = get_rnn_network(
            rnn_type=args.rnn_type,
            in_dim=args.piprot_dim,
            out_dim=args.piprot_dim,
            num_layers=args.piprot_layers,
        )

        self.args = args
        if args.pka_plm == 'none':
            self.res_emb = torch.nn.Embedding(21, args.pka_dim)
            in_dim = args.pka_dim
        else:
            in_dim = EsmModelInfo(args.pka_plm)['dim']
        self.rnn_pka = get_rnn_network(
            rnn_type=args.rnn_type,
            in_dim=in_dim,
            out_dim=args.pka_dim,
            num_layers=args.pka_layers,
        )

        dims = [args.pka_dim]
        if args.pipep_enable:
            dims.append(args.pipep_dim)
        if args.piprot_enable:
            dims.append(args.piprot_dim)
        merge_dim = sum(dims)

        self.typ_emb = torch.nn.Embedding(9, merge_dim)
        self.pka_out = torch.nn.Linear(merge_dim, 1)

    def forward(self, xs: List[torch.Tensor], seqs: List[str], resids=None, restypes=None):
        """_summary_

        Args:
            feats (Dict): Features from protein language model
            resids (_type_, optional): _description_. Defaults to None.
            res_types (_type_, optional): _description_. Defaults to None.
        """
        tokens = [torch.tensor([
            aa_to_idx.get(aa, 20) for aa in seq], dtype=torch.long).to('cuda') for seq in seqs]
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True)

        if self.args.pka_plm == 'none':
            x = self.res_emb(tokens)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                xs, batch_first=True)
            x = self.rnn_pka(x)

        x = [x]
        if self.args.pipep_enable:
            feat1 = self.rnn_pipep(self.res_emb_pipep(tokens)).detach()
            x.append(feat1)
        if self.args.piprot_enable:
            feat2 = self.rnn_piprot(self.res_emb_piprot(tokens)).detach()
            x.append(feat2)

        x = torch.cat(x, dim=-1)

        x = x[torch.arange(len(resids)), resids]
        x += self.typ_emb(restypes)
        x = self.pka_out(x).squeeze(-1)
        return x


class pl_pKALM(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.model = pKALM(args)
        self.loss_func = get_loss_func(args.loss)
        self.automatic_optimization = False
        self.metric = pKaMetric()

    def forward(self, *args):
        return self.model(*args)

    def _step(self, batch):
        xs, seqs, resids, restypes, pkas, buried, exposed = zip(*batch)
        restypes = torch.tensor(
            restypes, dtype=torch.long, device=self.device)
        pkas = torch.tensor(pkas, dtype=torch.float32, device=self.device)
        pred = self.model(
            xs=xs, seqs=seqs,
            resids=resids, restypes=restypes)
        model_pkas = torch.tensor([
            MODEL_PKA[idx_to_res[rt.item()]] for rt in restypes],
            dtype=torch.float32, device=self.device)
        pred = pred + model_pkas
        loss = self.loss_func(pred, pkas)
        return loss, pred, pkas, restypes, buried, exposed

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step(batch)
        self.log('train/pKa_loss', loss, prog_bar=True)

        self.manual_backward(loss)
        self.optimizers().step()
        if self.args.lr_sch:
            self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, pred, target, restypes, buried, exposed = self._step(batch)
            self.metric.add(
                pred, target, [idx_to_res[rt.item()] for rt in restypes],
                buried, exposed)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metric.compute(), prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.metric.clear()

    def configure_optimizers(self):
        opt = (
            torch.optim.AdamW
            if self.args.optimizer == 'adamw' else torch.optim.Adam
            if self.args.optimizer == 'adam' else torch.optim.SGD)(
            self.parameters(),
            lr=self.args.lr, weight_decay=self.args.wd)
        if self.args.lr_sch:
            sch = CosineLRScheduler(opt, t_initial=self.args.t_initial, lr_min=1e-6,
                                    warmup_t=round(self.args.wamrup_ratio * self.args.t_initial))
            return [opt], [sch]
        else:
            return opt
