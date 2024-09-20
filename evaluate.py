import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from models.dataset import MODEL_PKA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcsv', type=str, default='data/test.csv')
    args = parser.parse_args()

    logger.info('Reading test.csv...')
    df = pd.read_csv(args.testcsv)
    pkad_dict = {}
    is_buried, is_exposed = {}, {}
    for i, row in df.iterrows():
        pdbid, res_id, res_name, pka, mut_pos, mut_chain, chain = \
            row['PDB ID'], row['Res ID'], row['Res Name'], \
            row['pKa'], row['Mutant Pos'], row['Mutant Chain'], \
            row['Chain']
        key = (
            f'{pdbid.lower()}_{mut_pos}_{mut_chain}',
            chain, int(res_id), res_name)
        pkad_dict[key] = pka
        is_buried[key] = row['%SASA'] <= 50
        is_exposed[key] = row['%SASA'] > 50

    def get_pka(path):
        content = open(path).read()
        return re.findall(r_pka, content)

    def compute_rmse_mae(pred_dict, res_names=None, pka_range=(-100, 100)):
        res_names = res_names if res_names else [
            'ASP', 'GLU', 'HIS',
            'TYR', 'CYS', 'LYS', 'CTR', 'NTR']
        arr = []
        for key, pka in pkad_dict.items():
            shift = pka - MODEL_PKA.get(key[3], 0)
            if shift < pka_range[0] or shift >= pka_range[1]:
                continue
            if key in pred_dict and key[3] in res_names:  # res_name
                if 0 < pred_dict[key] < 12:  # remove outliers
                    arr.append([pka, pred_dict[key]])
            else:
                # modify chain to A
                _key = (key[0], 'A', key[2], key[3])
                if _key in pred_dict and _key[3] in res_names:
                    if 0 < pred_dict[_key] < 12:
                        arr.append([pka, pred_dict[_key]])
                # else:
                #     logger.warning(f'{key} failed to predict')
        if len(arr) == 0:
            return np.nan, np.nan, (0, 0)

        arr = np.array(arr)
        rmse = np.sqrt(np.mean((arr[:, 0] - arr[:, 1])**2))
        mae = np.mean(np.abs(arr[:, 0] - arr[:, 1]))
        return rmse, mae, arr.shape

    logger.info("Reading pypka output...")
    pypka_dict = {}
    for file in Path('comparison/pypka/pdbs').glob('*.pka'):
        pdb_id = file.stem
        for l in file.read_text().splitlines():
            res_id, res_name, pka, chain = l.strip().split()
            if pka != 'None':
                pypka_dict[(pdb_id, chain, int(res_id), res_name)] = float(pka)

    logger.info("Reading propka output...")
    r_pka = r'([A-Z]{3})[ ]{,3}(\d+) ([A-Z])[ ]{,3}(\d+\.\d+)'
    propka_dict = {}
    for file in Path('comparison/propka/pdbs').glob('*.pka'):
        for ResName, ResID, Chain, pKa in get_pka(file):
            propka_dict[(file.stem, Chain, int(ResID), ResName)] = float(pKa)

    logger.info("Reading pkai output...")
    pkai_dict = {}
    for file in Path('comparison/pkai/pdbs').glob('*.pka'):
        pdb_id = file.stem
        for l in file.read_text().splitlines():
            chain, res_id, res_name, pka = l.strip().split()
            if pka != 'None':
                pkai_dict[(pdb_id, chain, int(res_id), res_name)] = float(pka)

    logger.info("Reading pkai+ output...")
    pkaip_dict = {}
    for file in Path('comparison/pkai+/pdbs').glob('*.pka'):
        pdb_id = file.stem
        for l in file.read_text().splitlines():
            chain, res_id, res_name, pka = l.strip().split()
            if pka != 'None':
                pkaip_dict[(pdb_id, chain, int(res_id), res_name)] = float(pka)

    logger.info("Reading deepka output...")
    deepka_dict = {}
    for file in Path('comparison/deepka/predictions').glob('*.csv'):
        tmp_df = pd.read_csv(file)
        for i, row in tmp_df.iterrows():
            deepka_dict[(
                file.stem, row['Chain'], row['Res ID'],
                row['Res Name'])] = row['Predict pKa']

    def output_compare_residue():
        df_rmse, df_mae = [], []

        for name, d in (zip([
            'PypKa', 'PropKa', 'PKAI', 'PKAI+', 'DeepKa', 'EquipKa'
        ], [pypka_dict, propka_dict, pkai_dict, pkaip_dict, deepka_dict])):
            record_rmse, record_mae, record_count = [], [], []
            for res_name in ['ASP', 'GLU', 'HIS', 'TYR', 'CYS', 'LYS', 'CTR', 'NTR']:
                logger.info(f'Computing {name} for {res_name}...')
                try:
                    rmse, mae, shape = compute_rmse_mae(d, [res_name])
                    logger.warning(f'{name} {res_name} shape: {shape}')
                    record_rmse.append(rmse)
                    record_mae.append(mae)
                    record_count.append(shape[0])
                except:
                    record_rmse.append(np.nan)
                    record_mae.append(np.nan)
                    record_count.append(0)

            # compute total
            rmse, mae, _ = compute_rmse_mae(d)
            record_rmse.append(rmse)
            record_mae.append(mae)
            # compute major
            rmse, mae, _ = compute_rmse_mae(d, ['ASP', 'GLU', 'HIS', 'LYS'])
            record_rmse.append(rmse)
            record_mae.append(mae)
            # compute 0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-
            for pka_range in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 100)]:
                rmse, mae, _ = compute_rmse_mae(d, pka_range=pka_range)
                record_rmse.append(rmse)
                record_mae.append(mae)
            # compute buried and exposed
            buried_rmse, buried_mae, _ = compute_rmse_mae(
                {k: v for k, v in d.items() if is_buried.get(k, False)})
            exposed_rmse, exposed_mae, _ = compute_rmse_mae(
                {k: v for k, v in d.items() if is_exposed.get(k, False)})

            df_rmse.append([
                name, *record_rmse,
                buried_rmse, exposed_rmse,
                sum(record_count)])
            df_mae.append([
                name, *record_mae,
                buried_mae, exposed_mae,
                sum(record_count)])

        df_rmse = pd.DataFrame(
            df_rmse, columns=[
                'Method', 'ASP', 'GLU', 'HIS', 'TYR', 'CYS', 'LYS', 'CTR', 'NTR', 'Total', 'Major', '0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-', 'Buried', 'Exposed', 'Count'])
        df_mae = pd.DataFrame(
            df_mae, columns=[
                'Method', 'ASP', 'GLU', 'HIS', 'TYR', 'CYS', 'LYS', 'CTR', 'NTR', 'Total', 'Major', '0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-', 'Buried', 'Exposed', 'Count'])

        df_rmse.to_csv(
            f'comparison/{Path(args.testcsv).stem}_rmse.csv', index=False)
        df_mae.to_csv(
            f'comparison/{Path(args.testcsv).stem}_mae.csv', index=False)

    output_compare_residue()
