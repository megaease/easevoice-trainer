import torch
import torch.nn as nn
from .modules import TFC_TDF
from pytorch_lightning import LightningModule

dim_s = 4


class AbstractMDXNet(LightningModule):
    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length, overlap):
        super().__init__()
        self.target_name = target_name
        self.lr = lr
        self.optimizer = optimizer
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, dim_c, self.n_bins - self.dim_f, self.dim_t]), requires_grad=False)

    def get_optimizer(self):
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), self.lr)

        if self.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), self.lr)


class ConvTDFNet(AbstractMDXNet):
    def __init__(self, target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length,
                 num_blocks, l, g, k, bn, bias, overlap):

        super(ConvTDFNet, self).__init__(
            target_name, lr, optimizer, dim_c, dim_f, dim_t, n_fft, hop_length, overlap)
        # self.save_hyperparameters()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        if optimizer == 'rmsprop':
            norm = nn.BatchNorm2d

        if optimizer == 'adamw':
            norm = lambda input: nn.GroupNorm(2, input)

        self.n = num_blocks // 2
        scale = (2, 2)

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c, out_channels=g, kernel_size=(1, 1)),
            norm(g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(self.n):
            self.encoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias, norm=norm))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    norm(c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block = TFC_TDF(c, l, f, k, bn, bias=bias, norm=norm)

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    norm(c - g),
                    nn.ReLU()
                )
            )
            f = f * 2
            c -= g

            self.decoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias, norm=norm))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c, kernel_size=(1, 1)),
        )

    def forward(self, x):

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.bottleneck_block(x)

        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x


class Mixer(nn.Module):
    def __init__(self, device, mixer_path):
        super(Mixer, self).__init__()

        self.linear = nn.Linear((dim_s + 1) * 2, dim_s * 2, bias=False)

        self.load_state_dict(
            torch.load(mixer_path, map_location=device)
        )

    def forward(self, x):
        x = x.reshape(1, (dim_s + 1) * 2, -1).transpose(-1, -2)
        x = self.linear(x)
        return x.transpose(-1, -2).reshape(dim_s, 2, -1)


class ConvTDFNetTrim:
    def __init__(
            self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(ConvTDFNetTrim, self).__init__()

        self.dim_f = dim_f
        self.dim_t = 2 ** dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )
        self.target_name = target_name
        self.blender = "blender" in model_name

        self.dim_c = 4
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])
