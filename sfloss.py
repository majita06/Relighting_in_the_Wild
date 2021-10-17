import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_LoG_kernel(size, sigma, device):
    lin = torch.linspace(-(size - 1) // 2, size // 2, size, device=device)
    [x, y] = torch.meshgrid(lin, lin)
    ss = sigma ** 2
    xx = x * x
    yy = y * y
    g_div_ss = torch.exp(-(xx + yy) / (2. * ss)) / (2. * np.pi * (ss ** 2))
    a = (xx + yy - 2. * ss) * g_div_ss

    # Normalize.
    a = a - a.sum() / size ** 2
    return a

def get_LoG_filter(num_channels, sigma, device='cuda'):
    kernel_size = int(torch.ceil(8. * sigma))
    # Make into odd number.
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = get_LoG_kernel(kernel_size, sigma, device)
    # [kH, kW] => [OutChannels, (InChannels / groups) => 1, kH, kW].
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(num_channels, 1, 1, 1)

    # Create filter.
    filter = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                             kernel_size=kernel_size, groups=num_channels, bias=False,
                             padding=kernel_size // 2, padding_mode='reflect')
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    return filter


class SpatialFrequencyLoss(nn.Module):
    def __init__(self, num_channels, device='cuda', debug = False):
        super(SpatialFrequencyLoss, self).__init__()
        self.debug = debug

        self.sigmas = torch.tensor([0.6, 1.2, 2.4, 4.8, 9.6, 19.2]).to(device)
        self.w_sfl = torch.tensor([600, 500, 400, 20, 10, 10]).to(device)
        self.num_filters = len(self.sigmas)

        self.filters = []
        for x in range(self.num_filters):
            filter = get_LoG_filter(num_channels, self.sigmas[x], device)
            self.filters.append(filter)
    def forward(self, input, target):
        loss = 0.
        for x in range(self.num_filters):
            input_LoG = self.filters[x](input)
            target_LoG = self.filters[x](target)
            loss += self.w_sfl[x] * F.mse_loss(input_LoG, target_LoG)
        return loss
