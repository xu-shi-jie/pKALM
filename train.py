import subprocess
import time
import warnings

import lightning as pl
import numpy as np
import pandas as pd
import torch

import wandb
from models.dataset import pIDataloader, pKaDataloader
from models.model import pl_pIPeptide, pl_pIProtein, pl_pKALM
from models.utils import (Config, gen_ckpt_path, is_wandb_running,
                          seed_everything)

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)


def train(config_file, tune=False):
    args = Config(config_file)
    if tune:
        wandb.init(project=args.task)
        args.config.update(dict(wandb.config))

    seed_everything(args.seed)

    if args.task == 'PLM-PI-PEP':
        dm = pIDataloader(args)
    elif args.task == 'PLM-PI-PROT':
        dm = pIDataloader(args)
    elif args.task == 'PLM-PKA':
        dm = pKaDataloader(args)

    all_metrics = []
    for i in range(args.kfold):
        print(f'Fold {i+1}' + '\n' + '='*40)

        if args.task == 'PLM-PI-PEP':
            model = pl_pIPeptide(args)
            callbacks = [
                pl.pytorch.callbacks.ModelCheckpoint(
                    dirpath=args.ckpt_dir,
                    filename=gen_ckpt_path(
                        args, ['batch_size', 'pipep_dim', 'pipep_layers']) + f'-{{epoch}}',
                ),
            ]
        elif args.task == 'PLM-PI-PROT':
            model = pl_pIProtein(args)
            callbacks = [
                pl.pytorch.callbacks.ModelCheckpoint(
                    dirpath=args.ckpt_dir,
                    filename=gen_ckpt_path(
                        args, ['batch_size', 'piprot_dim', 'piprot_layers']) + f'-{{epoch}}',
                ),
            ]
        elif args.task == 'PLM-PKA':
            model = pl_pKALM(args)
            if args.ckpt_pipep:
                model.load_state_dict(torch.load(
                    args.ckpt_pipep), strict=False)
            if args.ckpt_piprot:
                model.load_state_dict(torch.load(
                    args.ckpt_piprot), strict=False)
            plm_name = args.pka_plm.replace('/', '_')
            callbacks = [
                pl.pytorch.callbacks.ModelCheckpoint(
                    dirpath=args.ckpt_dir,
                    filename=gen_ckpt_path(
                        args, ['batch_size', 'pka_dim', 'pka_layers']) + f'-{plm_name}-{{epoch}}',
                ),
            ]

        if args.kfold > 1:  # enable cross-validation
            dm.setup_kfold(i)
        args.t_initial = args.epochs * dm.steps_per_epoch

        precision = '16-mixed' if args.amp else '32-true'

        trainer = pl.Trainer(
            accelerator='gpu', devices=1, max_epochs=args.epochs,
            precision=precision, callbacks=callbacks,
        )
        trainer.fit(model, dm)

        trainer.validate(
            model, ckpt_path='best',
            dataloaders=dm.val_dataloader())
        time.sleep(1)
        val_test_metrics = trainer.callback_metrics.copy()
        trainer.validate(
            model, ckpt_path='best',
            dataloaders=dm.test_dataloader())
        time.sleep(1)

        val_test_metrics.update([
            (k.replace('val_', 'test_'), v) for k, v in trainer.callback_metrics.items()])
        all_metrics.append(val_test_metrics)

    d = {}
    for k in all_metrics[0]:
        d[k] = np.mean([m[k] for m in all_metrics if k in m])
    val_test_metrics = d

    if tune:
        wandb.log(val_test_metrics)
    elif args.task == 'PLM-PKA':
        # write to file
        # with open('metrics.txt', 'a') as f:
        #     f.write(f'Task: {args.task}\n')
        #     for k, v in val_test_metrics.items():
        #         f.write(f'{k}: {v}\n')

        subprocess.run(['python', 'evaluate.py'])
        df = pd.read_csv('comparison/test_rmse.csv')
        df.loc[len(df)] = [
            'pKALM',
            val_test_metrics['test_ASP_rmse'],
            val_test_metrics['test_GLU_rmse'],
            val_test_metrics['test_HIS_rmse'],
            val_test_metrics['test_TYR_rmse'],
            val_test_metrics['test_CYS_rmse'],
            val_test_metrics['test_LYS_rmse'],
            val_test_metrics['test_CTR_rmse'],
            val_test_metrics['test_NTR_rmse'],
            val_test_metrics['test_rmse'],
            val_test_metrics['test_major_rmse'],
            val_test_metrics['test_0_0.5_rmse'],
            val_test_metrics['test_0.5_1_rmse'],
            val_test_metrics['test_1_1.5_rmse'],
            val_test_metrics['test_1.5_2_rmse'],
            val_test_metrics['test_2_100_rmse'],
            val_test_metrics['test_buried_rmse'],
            val_test_metrics['test_exposed_rmse'],
            152,
        ]
        df.to_csv('comparison/test_rmse.csv', index=False)
        df = pd.read_csv('comparison/test_mae.csv')
        df.loc[len(df)] = [
            'pKALM',
            val_test_metrics['test_ASP_mae'],
            val_test_metrics['test_GLU_mae'],
            val_test_metrics['test_HIS_mae'],
            val_test_metrics['test_TYR_mae'],
            val_test_metrics['test_CYS_mae'],
            val_test_metrics['test_LYS_mae'],
            val_test_metrics['test_CTR_mae'],
            val_test_metrics['test_NTR_mae'],
            val_test_metrics['test_mae'],
            val_test_metrics['test_major_mae'],
            val_test_metrics['test_0_0.5_mae'],
            val_test_metrics['test_0.5_1_mae'],
            val_test_metrics['test_1_1.5_mae'],
            val_test_metrics['test_1.5_2_mae'],
            val_test_metrics['test_2_100_mae'],
            val_test_metrics['test_buried_mae'],
            val_test_metrics['test_exposed_mae'],
            152,
        ]
        df.to_csv('comparison/test_mae.csv', index=False)


if __name__ == '__main__':
    train('configs/train_pka.yaml', tune=is_wandb_running())
    # train('configs/train_pep_pi.yaml', tune=is_wandb_running())
    # train('configs/train_prot_pi.yaml', tune=is_wandb_running())
