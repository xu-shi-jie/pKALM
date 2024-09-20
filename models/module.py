import torch


def MLP(dim_list, dropout=0.0, act=torch.nn.ReLU):
    """ dim_list = [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]"""
    layers = []
    for i in range(1, len(dim_list)):
        layers.append(torch.nn.Linear(dim_list[i - 1], dim_list[i]))
        layers.append(torch.nn.Dropout(dropout))
        if i != len(dim_list) - 1:
            layers.append(act())
    return torch.nn.Sequential(*layers)


class BiLSTM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True, bidirectional=True)
        self.fc = MLP([out_dim * 2, out_dim])

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class BiGRU(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True, bidirectional=True)
        self.fc = MLP([out_dim * 2, out_dim])

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return x


def get_rnn_network(rnn_type, in_dim, out_dim, num_layers, dropout=0.0):
    """ Get RNN network or just MLP network"""
    if rnn_type == 'bilstm':
        return BiLSTM(in_dim, out_dim, num_layers, dropout)
    elif rnn_type == 'bigru':
        return BiGRU(in_dim, out_dim, num_layers, dropout)
    elif rnn_type == 'linear':
        return MLP([in_dim, out_dim])
    else:
        raise ValueError('Invalid RNN type')


# https://github.com/gngdb/pytorch-pca/blob/main/pca.py
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mean_ = torch.zeros(1, in_dim)
        self.components_ = torch.zeros(out_dim, in_dim)

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.out_dim is not None:
            d = min(self.out_dim, d)
        # self.register_buffer("mean_", X.mean(0, keepdim=True))
        self.mean_ = X.mean(0, keepdim=True)
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        # self.register_buffer("components_", Vt[:d])
        self.components_ = Vt[:d]
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


class DoubleConv1d(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, out_dim, 3, padding=1),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(out_dim, out_dim, 3, padding=1),
            torch.nn.BatchNorm1d(out_dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down1d(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.MaxPool1d(2),
            DoubleConv1d(in_dim, out_dim),
        )

    def forward(self, x):
        return self.conv(x)


class Up1d(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = torch.nn.ConvTranspose1d(
                in_dim//2, in_dim//2, kernel_size=2, stride=2)
        self.conv = DoubleConv1d(in_dim, out_dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = abs(x1.size()[-1] - x2.size()[-1])
        if x1.size()[-1] > x2.size()[-1]:
            x2 = torch.nn.functional.pad(x2, [diff//2, diff-diff//2])
        else:
            x1 = torch.nn.functional.pad(x1, [diff//2, diff-diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(torch.nn.Module):
    """ UNet for 1D signal"""

    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        for i in range(num_layers):
            self.downs.append(Down1d(in_dim, out_dim))
            in_dim = out_dim
            out_dim *= 2
        out_dim //= 2
        self.downs.append(DoubleConv1d(out_dim, out_dim))
        in_dim = out_dim*2
        out_dim //= 2
        for i in range(num_layers-1):
            self.ups.append(Up1d(in_dim, out_dim))
            in_dim = out_dim*2
            out_dim //= 2
        self.ups.append(Up1d(in_dim, out_dim))
        self.out_conv = torch.nn.Conv1d(out_dim, out_dim*2, 1)

    def forward(self, x):
        seq_len = x.size()[-1]
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        skips = skips[:-1]
        for up in self.ups:
            x = up(x, skips.pop())

        diff = seq_len - x.size()[-1]
        if diff > 0:
            x = torch.nn.functional.pad(x, [diff//2, diff-diff//2])
        return self.out_conv(x)
