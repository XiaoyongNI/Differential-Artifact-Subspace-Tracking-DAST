"""Reused EEGNet and explicit GRU/ChronoNet decoder definitions.

ChronoNet: three parallel (2,4,8)-kernel, stride-2 inception blocks followed
by four densely connected GRUs, following Roy et al., arXiv:1802.00308.
Adaptations: variable channels/window duration, last-state binary readout,
and configurable dropout. This is not a pretrained abnormal-EEG classifier.
"""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .common import import_source


class GRUDecoder(nn.Module):
    def __init__(self, channels, hidden_size=64, layers=2, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size, layers, batch_first=True,
                          dropout=dropout if layers>1 else 0)
        self.readout = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 2))

    def forward(self, x):
        _, hidden = self.gru(x.transpose(1, 2))
        return self.readout(hidden[-1])


class Inception1D(nn.Module):
    def __init__(self, channels, filters):
        super().__init__()
        self.branches = nn.ModuleList(nn.Conv1d(channels, filters, k, stride=2) for k in (2,4,8))

    def forward(self, x):
        # TensorFlow-style SAME padding even when the sequence length is odd.
        outputs = []
        for conv in self.branches:
            padding = max(0, ((x.shape[-1]+1)//2-1)*2 + conv.kernel_size[0]-x.shape[-1])
            outputs.append(F.elu(conv(F.pad(x, (padding//2, padding-padding//2)))))
        return torch.cat(outputs, dim=1)


class ChronoNet(nn.Module):
    def __init__(self, channels, filters=32, hidden_size=32, dropout=0.25):
        super().__init__()
        self.inception = nn.Sequential(Inception1D(channels, filters),
                                        Inception1D(3*filters, filters), Inception1D(3*filters, filters))
        self.recurrent = nn.ModuleList([nn.GRU(3*filters, hidden_size, batch_first=True)] +
            [nn.GRU(i*hidden_size, hidden_size, batch_first=True) for i in (1,2,3)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, 2)

    def forward(self, x):
        features = self.inception(x).transpose(1, 2)
        states = []
        for i, gru in enumerate(self.recurrent):
            sequence, _ = gru(features if i==0 else torch.cat(states, dim=-1))
            states.append(self.dropout(sequence))
        return self.readout(states[-1][:, -1])


@lru_cache(maxsize=None)
def _eegnet_class(source):
    return import_source(source, 'reused_swec_eegnet').EEGNet


def build_model(name, channels, samples, config):
    options = config['training']
    if name == 'EEGNet':
        model = _eegnet_class(config['eegnet_source'])(num_classes=2, **options['eegnet'])
        # Existing EEGNet creates both its spatial convolution and classifier
        # lazily. Materialize BEFORE constructing the optimizer/checkpoint load.
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, channels, samples))
    elif name == 'GRU':
        model = GRUDecoder(channels, **options['gru'])
    elif name == 'ChronoNet':
        model = ChronoNet(channels, **options['chrononet'])
    else:
        raise ValueError(f'Unknown decoder: {name}')
    return model
