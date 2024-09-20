import io
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests
from biotite.structure.io.pdb import PDBFile, get_structure
from tqdm import tqdm

# DeepKa
# data = {
#     ('6WXK', 49, 3.7),
#     ('1NAY', 56, 2.9),
#     ('4D63', 138, 4.4),
#     ('6NL9', 56, 4.2),
#     ('1G3B', 230, 6.5),
#     ('1ID2', 106, 5.4),
#     ('1W9Z', 257, 8.7),
#     ('5P4B', 330, 9.8),
#     ('2H2Z', 306, 9.5),
#     ('4WUV', 280, 10.0),
#     ('3FSB', 260, 10.2),
#     ('4J79', 300, 11.8),
#     ('3S76', 258, 11.3),
#     ('4K8P', 332, 12.5),
#     ('5U8F', 358, 14.5),
#     ('1KL1', 405, 14.7),
#     ('4MU9', 388, 17.3),
#     ('1GKM', 509, 16.5),
#     ('4ZAA', 496, 19.2),
#     ('1PEV', 610, 22.3),
# }
# total_residues, total_time = 0, 0
# for pdbid, _, t in tqdm(data):
#     file = requests.get(f'https://files.rcsb.org/download/{pdbid}.pdb').content
#     # save string to memory file
#     mem_file = io.StringIO(file.decode('utf-8'))
#     atoms = get_structure(PDBFile.read(mem_file))[0]
#     ava_residue = ['ASP', 'GLU', 'HIS', 'LYS']
#     ca_atoms = atoms[atoms.atom_name == 'CA']
#     total_residues += len([1 for atom in ca_atoms if atom.res_name in ava_residue])
#     total_time += t
# print(f'DeepKa: {total_residues} residues, {total_time:.4f} seconds')
# print(f'Average speed: {total_time / total_residues:.4f} seconds')

# PKAI
# remove *.pka
# for file in Path('speed/samples').glob('*.pka'):
#     file.unlink()
# start = time.time()
# subprocess.run(['bash', 'speed/pkai/2.run_pkai.sh'])
# escape = time.time() - start
# print(f'PKAI: {escape:.2f} seconds')
# total_count = 0
# for pka_file in Path('speed/samples').glob('*.pka'):
#     total_count += sum([1 for line in pka_file.open() if line.strip()])
# print("Total residues:", total_count)
# print("Average PKA time:", escape / total_count)

# PKAI+
# remove *.pka
# for file in Path('speed/samples').glob('*.pka'):
#     file.unlink()
# start = time.time()
# subprocess.run(['bash', 'speed/pkai+/2.run_pkai+.sh'])
# escape = time.time() - start
# print(f'PKAI: {escape:.2f} seconds')
# total_count = 0
# for pka_file in Path('speed/samples').glob('*.pka'):
#     total_count += sum([1 for line in pka_file.open() if line.strip()])
# print("Total residues:", total_count)
# print("Average PKA time:", escape / total_count)


# propka
# for file in Path('speed/samples').glob('*.pka'):
#     file.unlink()
# start = time.time()
# subprocess.run(['bash', 'speed/propka/2.run_propka.sh'])
# escape = time.time() - start
# print(f'PROPKA: {escape:.2f} seconds')
# total_count = 0
# r_pka = r'([A-Z]{3})[ ]{,3}(\d+) ([A-Z])[ ]{,3}(\d+\.\d+)'
# for pka_file in Path('speed/samples').glob('*.pka'):
#     total_count += len(re.findall(r_pka, pka_file.read_text()))
# print("Total residues:", total_count)
# print("Average PKA time:", escape / total_count)

# pypka
for file in Path('speed/samples').glob('*.pka'):
    file.unlink()
start = time.time()
subprocess.run(['bash', 'speed/pypka/2.run_pypka.sh'])
escape = time.time() - start
print(f'PYPKA: {escape:.2f} seconds')
total_count = 0
for pka_file in Path('speed/samples').glob('*.pka'):
    total_count += sum([1 for line in pka_file.open() if line.strip()])
print("Total residues:", total_count)
print("Average PKA time:", escape / total_count)

