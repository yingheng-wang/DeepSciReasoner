import os
import pickle
import sys
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

if not sys.warnoptions:
    warnings.simplefilter("ignore")

DATA_ROOT = './DATA'

# --data_src -> benchmark directory under DATA/. Each benchmark directory holds
# manifest.csv (sample IDs in the first column), split.pkl (fixed train/val/test
# row indices into manifest.csv) and graphs/<id>.chkpt (prepared crystal graphs).
BENCHMARKS = {
    'binned_dos_128': 'edos_caltech',
    'binned_dos_256': 'edos_tri',
    'ph_dos_51': 'phdos',
    'xanes_128_sigma0.5': 'xanes_full',
    'xanes_O_128_sigma0.5': 'xanes_o',
    'raman_128_sigma10': 'raman',
}


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_benchmark(data_src, root_dir=DATA_ROOT):
    """Return the manifest DataFrame and the fixed (train, val, test) row indices."""
    bench_dir = os.path.join(root_dir, BENCHMARKS[data_src])
    manifest = os.path.join(bench_dir, 'manifest.csv')
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f'{manifest} not found. Download the data first: '
            f'bash scripts/download_data.sh {BENCHMARKS[data_src]}')
    pd_data = pd.read_csv(manifest)
    with open(os.path.join(bench_dir, 'split.pkl'), 'rb') as f:
        train_idx, val_idx, test_idx = pickle.load(f)
    return pd_data, train_idx, val_idx, test_idx


class CIF_Lister(Dataset):
    def __init__(self, crystals_ids, full_dataset, df=None):
        self.crystals_ids = crystals_ids
        self.full_dataset = full_dataset
        self.material_ids = df.iloc[crystals_ids].values[:, 0].squeeze()

    def __len__(self):
        return len(self.crystals_ids)

    def __getitem__(self, idx):
        i = self.crystals_ids[idx]
        material = self.full_dataset[i]

        n_features = material[0][0]
        e_features = material[0][1]
        e_features = e_features.view(-1, 41)
        a_matrix = material[0][2]

        groups = material[1]
        enc_compo = material[2]
        coordinates = material[3]
        y = material[4]
        cif_id = material[5]
        # XANES branch returns absorber atomic number; other sources return 0 sentinel.
        absorber_z = material[7] if len(material) > 7 else 0
        absorber_z_tensor = torch.tensor([int(absorber_z)], dtype=torch.long)

        graph_crystal = Data(x=n_features, y=y, edge_attr=e_features, edge_index=a_matrix, global_feature=enc_compo, \
                             cluster=groups, num_atoms=torch.tensor([len(n_features)]).float(), coords=coordinates,
                             the_idx=torch.tensor([float(i)]), cif_id=cif_id, absorber_z=absorber_z_tensor)

        return graph_crystal


class CIF_Dataset(Dataset):
    """Prepared crystal graphs of one benchmark, indexed by manifest row."""

    def __init__(self, args, pd_data=None, root_dir=DATA_ROOT):
        self.args = args
        self.pd_data = pd_data
        self.graph_dir = os.path.join(root_dir, BENCHMARKS[args.data_src], 'graphs')

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):
        cif_id = self.pd_data.iloc[idx, 0]
        graph_path = os.path.join(self.graph_dir, str(cif_id) + '.chkpt')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f'{graph_path} not found. Download the data first: '
                f'bash scripts/download_data.sh {BENCHMARKS[self.args.data_src]}')

        # The prepared graphs contain pymatgen objects, so they are full pickles.
        tmp_dist = torch.load(graph_path, weights_only=False)

        atom_fea = tmp_dist['atom_fea'] # [n_atom, 92]
        nbr_fea = tmp_dist['nbr_fea'] # [n_atom, 12, 41]
        nbr_fea_idx = tmp_dist['nbr_fea_idx'] # [2, n_edge]
        groups = tmp_dist['groups'] # [n_atom]
        enc_compo = tmp_dist['enc_compo'] # [1, 103]
        coordinates = tmp_dist['coordinates'] # [n_atom, 3]
        target = tmp_dist['target'] # [label_dim]
        cif_id = tmp_dist['cif_id']
        atom_id = tmp_dist['atom_id']
        # XANES adds an absorber atomic number; reuse 0 sentinel for non-XANES sources.
        absorber_z = tmp_dist.get('absorber_z', 0) if isinstance(tmp_dist, dict) else 0
        return (atom_fea, nbr_fea, nbr_fea_idx), groups, enc_compo, coordinates, target, cif_id, atom_id, absorber_z
