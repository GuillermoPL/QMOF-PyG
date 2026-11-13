import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool


class ConvLayer(MessagePassing):
    """
    Crystal graph convolution (Xie & Grossman, 2018) as a PyG MessagePassing
    layer.

    The parameter names (fc_full, bn1, bn2) are identical to the original
    pure-PyTorch implementation, so old checkpoints can be loaded as-is.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super().__init__(aggr='add', flow='source_to_target')
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor) -> Tensor:
        """
        N: Total number of atoms in the batch
        E: Total number of edges in the batch

        Parameters
        ----------

        x: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features before convolution
        edge_index: torch.LongTensor shape (2, E)
          edge_index[0] = neighbor atom (source),
          edge_index[1] = central atom (target)
        edge_attr: torch.Tensor shape (E, nbr_fea_len)
          Bond features

        Returns
        -------

        out: torch.Tensor shape (N, atom_fea_len)
          Atom hidden features after convolution
        """
        nbr_sumed = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                   size=(x.size(0), x.size(0)))
        nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(x + nbr_sumed)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # x_i: central atom, x_j: neighbor atom
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        z = self.bn1(self.fc_full(z))
        nbr_filter, nbr_core = z.chunk(2, dim=-1)
        return self.sigmoid(nbr_filter) * self.softplus1(nbr_core)


class CrystalGraphConvNet(nn.Module):
    """
    Crystal graph convolutional neural network for predicting total
    material properties, built on PyTorch Geometric.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        N: Total number of atoms in the batch
        E: Total number of edges in the batch
        B: Number of crystals in the batch

        Parameters
        ----------

        x: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        edge_index: torch.LongTensor shape (2, E)
          Graph connectivity (neighbor -> central atom)
        edge_attr: torch.Tensor shape (E, nbr_fea_len)
          Bond features
        batch: torch.LongTensor shape (N,)
          Crystal index of each atom

        Returns
        -------

        out: torch.Tensor shape (B, 1) (or (B, 2) log-probs if classification)
        """
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        crys_fea = global_mean_pool(x, batch)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out
