import argparse
import time
from argparse import Namespace as Args
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from models.dataset import MODEL_PKA, idx_to_res, rAA, res_to_idx
from models.model import pKALM
from models.plm import get_model
from models.utils import Config


def print_help():
    ver = '1.0'
    print('-' * 80)
    print(
        f'                 pKALM v{ver}, Developed by Shijie Xu, 2024                 \n')
    print(f'''If you use this tool, please cite paper:
    <> \n''')
    print('''Usuage example:
    python predict.py -i <inputs.csv> -o <out_dir>\n''')
    print('''<input.csv> file should contain three columns named "idx", "res", "seq":
    1. Index of residues in the sequence (starting from 1)
    2. Type of the residue (ASP, GLU, HIS, LYS, CYS, TYR, NTR, CTR)
    3. Sequence of the protein\n''')
    print('''The output files will be saved to
    ./<out_dir>/outputs.csv\n''')
    print('-' * 80)


def parse_inputs(df: pd.DataFrame):
    sequences, resids, res_types = [], [], []
    for i, row in df.iterrows():
        idx, res, seq = row['idx'], row['res'], row['seq']
        rn = rAA[seq[idx-1]]
        if res not in ['CTR', 'NTR']:
            assert res == rn, f'{res} != {rn}, {seq, idx}'
        sequences.append(seq)
        resids.append(idx-1)
        res_types.append(res_to_idx[res])
    return sequences, torch.tensor(resids).to('cuda'), torch.tensor(res_types).to('cuda')


def load_weights(model: pKALM, args: Args, dev) -> pKALM:
    w = torch.load(args.ckpt_model)['state_dict']
    # load pka model
    model.rnn_pka.load_state_dict(
        {k.replace('model.rnn_pka.', ''): v for k, v in w.items() if 'model.rnn_pka' in k})
    model.typ_emb.load_state_dict(
        {k.replace('model.typ_emb.', ''): v for k, v in w.items() if 'model.typ_emb' in k})
    model.pka_out.load_state_dict(
        {k.replace('model.pka_out.', ''): v for k, v in w.items() if 'model.pka_out' in k})
    # load pipep model
    w = torch.load(args.ckpt_pipep)['state_dict']
    model.rnn_pipep.load_state_dict(
        {k.replace('model.rnn_pipep.', ''): v for k, v in w.items() if 'model.rnn_pipep' in k})
    model.res_emb_pipep.load_state_dict(
        {k.replace('model.res_emb_pipep.', ''): v for k, v in w.items() if 'model.res_emb_pipep' in k})
    # load piprot model
    w = torch.load(args.ckpt_piprot)['state_dict']
    model.rnn_piprot.load_state_dict(
        {k.replace('model.rnn_piprot.', ''): v for k, v in w.items() if 'model.rnn_piprot' in k})
    model.res_emb_piprot.load_state_dict(
        {k.replace('model.res_emb_piprot.', ''): v for k, v in w.items() if 'model.res_emb_piprot' in k})
    return model.to(dev).eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input.csv')
    parser.add_argument('-o', '--out_dir', type=str, default='.')
    args = parser.parse_args()
    print_help()

    model_args = Config('configs/predict.yaml')
    model = pKALM(model_args)
    model = load_weights(model, model_args, 'cuda')
    model.eval()
    logger.info('Model loaded successfully')
    plm_func = get_model(model_args.pka_plm, dev='cuda')
    logger.info('PLM model loaded successfully')

    # load data
    df = pd.read_csv(args.input)
    sequences, resids, restypes = parse_inputs(df)
    preds = []
    logger.info(f"Encoding sequences by {model_args.pka_plm}")
    xs = []
    for seq, resid, residx in tqdm(zip(sequences, resids, restypes), total=len(sequences)):
        xs.append(plm_func([seq])[..., 1].to('cuda'))
    t0 = time.time()
    logger.info("Predicting pKa values")
    for x, seq, resid, residx in tqdm(zip(xs, sequences, resids, restypes), total=len(sequences)):
        pred = model([x], [seq], resid.unsqueeze(0), residx.unsqueeze(0))
        preds.append(pred)
    pred = torch.cat(preds)
    t1 = time.time()
    model_pka = torch.tensor([
        MODEL_PKA[idx_to_res[i.item()]] for i in restypes], device='cuda')
    t2 = time.time()
    df['shift'] = pred.cpu().detach().numpy()
    df['pred'] = (model_pka + pred).cpu().detach().numpy()

    logger.info(f'Predictions saved to {Path(args.out_dir) / "outputs.csv"}')
    logger.info(
        f'Encoding time: {t1-t0:.2f}s, avg: {(t1-t0)/len(sequences):.2f}s')
    logger.info(
        f'Prediction time: {t2-t1:.2f}s, avg: {(t2-t1)/len(sequences):.2f}s')
    df.to_csv(f'{args.out_dir}/outputs.csv', header=False, index=False)
