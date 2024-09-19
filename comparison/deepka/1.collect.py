import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

pdbids = []
for file in Path('comparison/deepka/predictions').glob('*.csv'):
    pdbids.append(file.stem)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['variant'] = df.apply(
    lambda x: f'{x["PDB ID"].lower()}_{x["Mutant Pos"]}_{x["Mutant Chain"]}', axis=1)
unpredicted_ids = df[~df['variant'].isin(pdbids)]['variant'].unique()
print('Unpredicted ids:', sorted(unpredicted_ids))

url = r'http://159.75.31.213:8121/DeepKaServer/predict'
headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Connection': 'keep-alive',
    'Content-Length': '1519094',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Host': '159.75.31.213:8121',
    'Origin': 'http://www.computbiophys.com',
    'Referer': 'http://www.computbiophys.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
}

data = {
    'model': 'deepka2',
    'input_type': 'PDB file',
    'pdb_id': '',
    'pdb_file_name': '',
    'pdb_file_content': '',
}
Path('comparison/deepka/predictions').mkdir(exist_ok=True, parents=True)
for vari in (pbar := tqdm(unpredicted_ids)):
    pbar.set_description(f'Submitting {vari}')
    data['pdb_file_name'] = vari
    data['pdb_file_content'] = open(
        f'data/fixed_pdbs/{vari}.pdb').read()
    r = requests.post(url, headers=headers, data=json.dumps(data))
    if r.status_code != 200:
        print(r.text)
        continue
    d = json.loads(r.text)
    with open(f'comparison/deepka/predictions/{vari}.csv', 'w') as f:
        f.write(d['result_content'])
