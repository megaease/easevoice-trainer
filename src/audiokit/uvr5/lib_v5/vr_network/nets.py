import torch
import torch.nn.functional as F
from torch import nn

import src.audiokit.uvr5.lib_v5.vr_network.layers as layers


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, enlarge, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations, enlarge=enlarge)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft, params):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(params[0][0], params[0][1], params[0][2])
        self.stg1_high_band_net = BaseASPPNet(params[1][0], params[1][1], params[1][2])

        self.stg2_bridge = layers.Conv2DBNActiv(params[2][0], params[2][1], params[2][2], params[2][3], params[2][4])
        self.stg2_full_band_net = BaseASPPNet(params[3][0], params[3][1], params[3][2])

        self.stg3_bridge = layers.Conv2DBNActiv(params[4][0], params[4][1], params[4][2], params[4][3], params[4][4])
        self.stg3_full_band_net = BaseASPPNet(params[5][0], params[5][1], params[5][2])

        self.out = nn.Conv2d(params[6][0], params[6][1], params[6][2], bias=params[6][3])
        self.aux1_out = nn.Conv2d(params[7][0], params[7][1], params[7][2], bias=params[7][3])
        self.aux2_out = nn.Conv2d(params[8][0], params[8][1], params[8][2], bias=params[8][3])

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x, aggressiveness=None):
        mix = x.detach()
        x = x.clone()

        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"]:] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"]:],
                    1 + aggressiveness["value"],
                )

            return mask * mix

    def predict(self, x_mag, aggressiveness=None):
        h = self.forward(x_mag, aggressiveness)

        if self.offset > 0:
            h = h[:, :, :, self.offset: -self.offset]
            assert h.size()[3] > 0

        return h


def get_nets_model(n_fft, size=61968) -> CascadedASPPNet:
    params = {
        33966: [
            [2, 16, True],
            [2, 16, True],
            [18, 8, 1, 1, 0],
            [8, 16, True],
            [34, 16, 1, 1, 0],
            [16, 32, True],
            [32, 2, 1, False],
            [16, 2, 1, False],
            [16, 2, 1, False],
        ],
        61968: [
            [2, 32, False],
            [2, 32, False],
            [34, 16, 1, 1, 0],
            [16, 32, False],
            [66, 32, 1, 1, 0],
            [32, 64, False],
            [64, 2, 1, False],
            [32, 2, 1, False],
            [32, 2, 1, False],
        ],
        123812: [
            [2, 32, False],
            [2, 32, False],
            [34, 16, 1, 1, 0],
            [16, 32, False],
            [66, 32, 1, 1, 0],
            [32, 64, False],
            [64, 2, 1, False],
            [32, 2, 1, False],
            [32, 2, 1, False],
        ],
        123821: [
            [2, 32, False],
            [2, 32, False],
            [34, 16, 1, 1, 0],
            [16, 32, False],
            [66, 32, 1, 1, 0],
            [32, 64, False],
            [64, 2, 1, False],
            [32, 2, 1, False],
            [32, 2, 1, False],
        ],
        537227: [
            [2, 64, True],
            [2, 64, True],
            [66, 32, 1, 1, 0],
            [32, 64, True],
            [130, 64, 1, 1, 0],
            [64, 128, True],
            [128, 2, 1, False],
            [64, 2, 1, False],
            [64, 2, 1, False],
        ],
        537238: [
            [2, 64, True],
            [2, 64, True],
            [66, 32, 1, 1, 0],
            [32, 64, True],
            [130, 64, 1, 1, 0],
            [64, 128, True],
            [128, 2, 1, False],
            [64, 2, 1, False],
            [64, 2, 1, False],
        ],
        16983: [
            [2, 16, False],
            [2, 16, False],
            [18, 8, 1, 1, 0],
            [8, 16, False],
            [34, 16, 1, 1, 0],
            [16, 32, False],
            [32, 2, 1, False],
            [16, 2, 1, False],
            [16, 2, 1, False],
        ]
    }
    param = params[size]
    return CascadedASPPNet(n_fft, param)
