import torch
import torch.nn as nn


class Backbone3D(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.MODULES = []
        self.