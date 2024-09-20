# Accurate and Rapid Prediction of Protein pKa: Protein Language Models Reveal the Sequence-pKa Relationship

This is the official implementation of the paper ["Accurate and Rapid Prediction of Protein pKa: Protein Language Models Reveal the Sequence-pKa Relationship"](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1).

## Freely available server
Our pKALM server will be soon available: [Access pKALM](https://onodalab.ees.hokudai.ac.jp/pkalm)

## Installation


```bash
conda create -n pkalm python=3.12
conda activate pkalm
pip install -r requirements.txt
```

## Usage

Use `predict_batch.py` to predict pKa values using a FASTA file:
- `--input`: a FASTA-formatted file containing protein sequences.
- `--out_dir`: the output directory.

Use `predict.py` to predict pKa values using an input file:
- `--input`: an CSV-formatted file containing `idx`, `res`, and `seq` columns.
- `--out_dir`: the output directory.


## Datasets
- `data/seq_train.csv` and `data/seq_test.csv`: The training and testing datasets used in the paper.
- `data/PKAD2_DOWNLOAD.xlsx`: The raw PKAD-2 dataset.
- `data/process_PKAD2_DOWNLOAD_rev.xlsx`: The processed PKAD-2 dataset.
- `data/IPC2_peptide_25.csv` and `data/IPC2_peptide_75.csv`: The peptide IPC2 datasets for testing and training.
- `data/IPC2_protein_25.csv` and `data/IPC2_protein_75.csv`: The protein IPC2 datasets for testing and training.
- `data/UP000005640_9606.fasta`: The human proteome dataset for benchmarking speeds.

## Help

If you have any questions, please contact me at `shijie.xu@ees.hokudai.ac.jp`.