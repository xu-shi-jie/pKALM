import argparse
import time
from argparse import Namespace as Args
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO
from loguru import logger
from tqdm import tqdm

from models.dataset import MODEL_PKA, aa_to_idx, idx_to_res, rAA, res_to_idx
from models.model import pKALM
from models.plm import get_model
from models.utils import Config


def print_help():
    ver = "1.0"
    print("-" * 80)
    print(
        f"             pKALM v{ver} (batch version), Developed by Shijie Xu, 2024             \n"
    )
    print(f"""If you use this tool, please cite paper: <> \n""")
    print(
        """Usuage example:
    python predict_batch.py -i <fasta file> -o <out_dir>\n"""
    )
    print(
        """The output files will be saved to
    ./<out_dir>/outputs.csv\n"""
    )
    print("-" * 80)


def load_weights(model: pKALM, args: Args, dev) -> pKALM:
    w = torch.load(args.ckpt_model)["state_dict"]
    # load pka model
    model.rnn_pka.load_state_dict(
        {
            k.replace("model.rnn_pka.", ""): v
            for k, v in w.items()
            if "model.rnn_pka" in k
        }
    )
    model.typ_emb.load_state_dict(
        {
            k.replace("model.typ_emb.", ""): v
            for k, v in w.items()
            if "model.typ_emb" in k
        }
    )
    model.pka_out.load_state_dict(
        {
            k.replace("model.pka_out.", ""): v
            for k, v in w.items()
            if "model.pka_out" in k
        }
    )
    # load pipep model
    w = torch.load(args.ckpt_pipep)["state_dict"]
    model.rnn_pipep.load_state_dict(
        {
            k.replace("model.rnn_pipep.", ""): v
            for k, v in w.items()
            if "model.rnn_pipep" in k
        }
    )
    model.res_emb_pipep.load_state_dict(
        {
            k.replace("model.res_emb_pipep.", ""): v
            for k, v in w.items()
            if "model.res_emb_pipep" in k
        }
    )
    # load piprot model
    w = torch.load(args.ckpt_piprot)["state_dict"]
    model.rnn_piprot.load_state_dict(
        {
            k.replace("model.rnn_piprot.", ""): v
            for k, v in w.items()
            if "model.rnn_piprot" in k
        }
    )
    model.res_emb_piprot.load_state_dict(
        {
            k.replace("model.res_emb_piprot.", ""): v
            for k, v in w.items()
            if "model.res_emb_piprot" in k
        }
    )
    return model.to(dev).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="data/UP000005640_9606.fasta"
    )
    parser.add_argument("-o", "--out_dir", type=str, default="humanproteome/")
    args = parser.parse_args()
    print_help()

    if not Path(args.out_dir).exists():
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Output directory created at {args.out_dir} because it does not exist"
        )

    model_args = Config("configs/predict.yaml")
    model = pKALM(model_args)
    model = load_weights(model, model_args, "cuda")
    model.eval()
    logger.info("Model loaded successfully")
    plm_func = get_model(model_args.pka_plm, dev="cuda")
    logger.info("PLM model loaded successfully")

    # load data
    fasta_file = SeqIO.parse(args.input, "fasta")
    for r in tqdm(fasta_file, desc="Predicting"):
        seqid, seq = r.id, str(r.seq)
        # convert seqid to an available filename
        file_id = seqid.replace("/", "_").replace("\\", "_").replace(":", "_")
        if Path(f"{args.out_dir}/{file_id}.csv").exists():
            continue
        resids, restypes = [], []
        for i, c in enumerate(seq):
            rn = rAA.get(c, "UNK")
            if rn in MODEL_PKA:
                resids.append(i)
                restypes.append(res_to_idx[rn])
            if i == 0:
                resids.append(i)
                restypes.append(res_to_idx["NTR"])
            if i == len(seq) - 1:
                resids.append(i)
                restypes.append(res_to_idx["CTR"])
        with torch.no_grad():
            pred = model(
                [plm_func([seq])[..., 1].to("cuda")],
                [seq],
                torch.tensor(resids).to("cuda").unsqueeze(0),
                torch.tensor(restypes).to("cuda").unsqueeze(0),
            )

        model_pka = torch.tensor(
            [MODEL_PKA[idx_to_res[restype]] for restype in restypes], device="cuda"
        )
        df = pd.DataFrame(
            {
                "seqid": [seqid] * len(resids),
                "resid": resids,
                "residue": [idx_to_res[i] for i in restypes],
                "shift": pred.squeeze(0).cpu().detach().numpy(),
                "pred": (model_pka + pred).squeeze(0).cpu().detach().numpy(),
            }
        )
        df.to_csv(f"{args.out_dir}/{file_id}.csv", header=False, index=False)
    logger.info(f"Predictions saved to {args.out_dir}")
