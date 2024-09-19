import gzip
import random
import shutil
import subprocess
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm

if __name__ == '__main__':
    files = Path('../database/wwpdb/').glob('**/pdb*.ent.gz')
    files = list(files)
    random.shuffle(files)
    files = files[:100]

    Path('speed/samples').mkdir(exist_ok=True, parents=True)
    for file in tqdm(files):
        with gzip.open(file, 'rt') as f:
            with open(f'speed/samples/{file.stem.split(".")[0]}.pdb', 'w') as out:
                out.write(f.read())
