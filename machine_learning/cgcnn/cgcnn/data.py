import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def get_train_val_test_loader(dataset, batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=0, pin_memory=False,
                              train_size=None, val_size=None, test_size=None):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!
    (CIFData shuffles itself using its `random_seed`.)

    Parameters
    ----------
    dataset: torch_geometric.data.Dataset
      The full dataset to be divided.
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    train_size, val_size, test_size: int
      Absolute sizes. If given, they override the corresponding ratio.

    Returns
    -------
    train_loader: torch_geometric.loader.DataLoader
      DataLoader that randomly samples the training data.
    val_loader: torch_geometric.loader.DataLoader
    (test_loader): torch_geometric.loader.DataLoader
      Returned if return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    if not train_size:
        train_size = int(train_ratio * total_size)
    if not test_size:
        test_size = int(test_ratio * total_size)
    if not val_size:
        val_size = int(val_ratio * total_size)

    indices = list(range(total_size))
    train_idx = indices[:train_size]
    val_idx = indices[total_size - (val_size + test_size):
                      total_size - test_size]
    test_idx = indices[total_size - test_size:]

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory)
    train_loader = DataLoader(dataset[train_idx], shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(dataset[val_idx], shuffle=False, **loader_kwargs)
    if return_test:
        test_loader = DataLoader(dataset[test_idx], shuffle=False,
                                 **loader_kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super().__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    PyTorch Geometric dataset where the crystal structures are stored in the
    form of CIF files. The dataset should have the following directory
    structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    disable_save_torch: bool
        Don't save the processed crystal graphs to `root_dir/cifdata/*.pt`
    random_seed: int
        Random seed for shuffling the dataset

    Each item is a `torch_geometric.data.Data` with:

    x: torch.Tensor shape (n_i, atom_fea_len)
    edge_index: torch.LongTensor shape (2, n_i * M)
        edge_index[0] = neighbor atom, edge_index[1] = central atom
    edge_attr: torch.Tensor shape (n_i * M, nbr_fea_len)
    y: torch.Tensor shape (1, 1)
    cif_id: str
    """

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 disable_save_torch=False, random_seed=42):
        super().__init__(root=None)
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.disable_save_torch = disable_save_torch
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [[x.strip().replace('\ufeff', '')
                                  for x in row] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.torch_data_path = os.path.join(self.root_dir, 'cifdata')

    def len(self):
        return len(self.id_prop_data)

    def get(self, idx):
        cif_id, target = self.id_prop_data[idx]
        cif_id = cif_id.replace('ï»¿', '')

        cache_file = os.path.join(self.torch_data_path, cif_id + '.pt')
        if os.path.exists(cache_file):
            data = torch.load(cache_file, weights_only=False)
        else:
            data = self._build_graph(cif_id)
            if not self.disable_save_torch:
                os.makedirs(self.torch_data_path, exist_ok=True)
                torch.save(data, cache_file)

        # The target is not cached so id_prop.csv can be edited freely
        data.y = torch.tensor([[float(target)]], dtype=torch.float)
        data.cif_id = cif_id
        return data

    def _build_graph(self, cif_id):
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(site.specie.number)
                              for site in crystal])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)

        src, dst, dist = [], [], []
        for center, nbrs in enumerate(all_nbrs):
            nbrs = sorted(nbrs, key=lambda n: n.nn_distance)
            if len(nbrs) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
            nbrs = nbrs[:self.max_num_nbr]
            src += [n.index for n in nbrs]
            dst += [center] * len(nbrs)
            dist += [n.nn_distance for n in nbrs]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(self.gdf.expand(np.array(dist)),
                                 dtype=torch.float)
        return Data(x=torch.tensor(atom_fea, dtype=torch.float),
                    edge_index=edge_index, edge_attr=edge_attr)
