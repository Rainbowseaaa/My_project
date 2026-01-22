# 瑞利索墨菲衍射传播计算
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class rsProp:
    def __init__(self, config, device, coordinate):
        self.wl = config['lambda']
        self.k = torch.pi * 2 / self.wl
        self.rij, self.cos_factor = self.factor(coordinate.x, coordinate.y, config['distance'])

    def factor(self, x, y, d):
        x0 = x.reshape(-1, 1)
        deltax = x0 - x0.T
        y0 = y.reshape(-1, 1)
        deltay = y0 - y0.T
        rij = torch.sqrt(deltax ** 2 + deltay ** 2 + d ** 2)
        cos_factor = d / rij
        return rij, cos_factor

    def prop(self, Ein, device='cpu'):
        shape = Ein.shape
        Ein = (Ein + 0j).to(device).reshape(-1, 1)
        Eout = 1/1j/self.wl * torch.matmul((torch.exp(1j * self.k * self.rij) / self.rij * self.cos_factor), Ein)
        return Eout.reshape(shape)
