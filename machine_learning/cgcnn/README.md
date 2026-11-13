# CGCNN (PyTorch Geometric version)

This folder contains a port of the [Crystal Graph Convolutional Neural Network](https://github.com/txie-93/cgcnn) from plain PyTorch to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (PyG).

## Installation

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build, install `torch` first following the [PyTorch instructions](https://pytorch.org/get-started/locally/). No extra PyG extensions (`torch_scatter`, `torch_sparse`, ...) are required.

## Usage

`run_train.sh` and `run_predict.sh` are sample scripts to train a model and to make predictions with a trained model, respectively. The CLI is unchanged with respect to the original implementation.

```bash
python main.py --batch-size 16 --n-conv 5 --n-h 1 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --epochs 300 data/QMOF
python predict.py output/model_best.pth.tar data/QMOF
```

The dataset directory must contain `id_prop.csv`, `atom_init.json` and one `.cif` per entry (see `data/example`).

## What changed with respect to the plain-PyTorch version

| Before | Now |
| --- | --- |
| Dense neighbor tensors `nbr_fea (N, M, F)` and `nbr_fea_idx (N, M)` | Sparse graph: `edge_index (2, E)` and `edge_attr (E, F)` |
| Hand-written `collate_pool` + `crystal_atom_idx` list | `torch_geometric.loader.DataLoader` (`batch.batch` vector) |
| `ConvLayer(nn.Module)` with `expand`/`cat`/`view` | `ConvLayer(MessagePassing)` (`message` + `aggr='add'`) |
| Per-crystal python loop for pooling | `global_mean_pool` |
| `CIFData(torch.utils.data.Dataset)`, graph cache in `cifdata/*.pkl` | `CIFData(torch_geometric.data.Dataset)`, graph cache in `cifdata/*.pt` |
| `Normalizer`, `AverageMeter`, ... duplicated in `main.py` and `predict.py` | Shared in `cgcnn/utils.py` |

**Checkpoint compatibility:** parameter names are identical, so checkpoints trained with the original implementation load directly with `load_state_dict`.

**Differences to be aware of:**

- The old implementation padded atoms with fewer than `max_num_nbr` neighbors within the cutoff using fake neighbors (index 0 at distance `radius + 1`). The graph version simply has fewer edges for those atoms. The two are identical whenever every atom has at least `max_num_nbr` neighbors (the usual case for MOFs with `radius=8`).
- Old `cifdata/*.pkl` caches are ignored; graphs are rebuilt once and cached as `cifdata/*.pt`. Use `--clean-torch` to delete the cache if you change `radius` or `max_num_nbr`.

For more details on the original CGCNN see https://github.com/arosen93/cgcnn. In general, we also recommend [MatDeepLearn](https://github.com/vxfung/MatDeepLearn) if you are interested in training graph neural networks given its superior features.
