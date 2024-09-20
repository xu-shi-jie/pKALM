import shutil
from pathlib import Path

import pandas as pd

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['variant'] = df.apply(
    lambda x: f'{x["PDB ID"].lower()}_{x["Mutant Pos"]}_{x["Mutant Chain"]}', axis=1)

unique_pdbids = df['variant'].unique()

# shutil.rmtree('comparison/propka/pdbs', ignore_errors=True)
Path('comparison/propka/pdbs').mkdir(parents=True, exist_ok=True)

raw_pdbs = Path(r'data/fixed_pdbs')
for pdbid in unique_pdbids:
    # copy pdb file
    pdb_file = raw_pdbs / f'{pdbid}.pdb'
    shutil.copy(pdb_file, r'comparison/propka/pdbs')
