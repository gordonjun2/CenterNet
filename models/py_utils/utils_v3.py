import torch
import torch.nn as nn

import math

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_up_layer(modules, dim):
    growth_rate = dim / 2
    #nb_layers = 1           # Hyperparameters, but treat as a constant first
    reduction = 0.5         # Hyperparameters, but treat as a constant first
    dropRate = 0            # Hyperparameters, but treat as a constant first
    depth = 10
    in_planes = 2 * growth_rate
    n = (depth - 4) / 3
    n = int(n)
    layers = []

    for _ in range(modules):
        for i in range(n):
            layers.append(DenseBlock(nb_layers = i, in_planes = in_planes, growth_rate = growth_rate, block = BasicBlock, dropRate = dropRate))  
        in_planes = int(in_planes+n*growth_rate)  
        layers.append(TransitionBlock(in_planes = in_planes, out_planes = int(math.floor(in_planes*reduction)), dropRate = dropRate))
        in_planes = int(math.floor(in_planes*reduction))
    
    return nn.Sequential(*layers)

def make_hg_layer(kernel, mod, dim0, dim1, nDenselayer):
    #nDenselayer = 2             # Hyperparameters, but treat as a constant first
    growthRate = dim0 // 2        # Hyperparameters, but treat as a constant first
    nChannels = int(growthRate * 2)
    layers = []
    nChannels_def = nChannels
    nChannels_ = nChannels

    for _ in range(mod):
        for i in range(nDenselayer):
            layers.append(RDB(nChannels = nChannels_, nDenselayer = nDenselayer, growthRate = growthRate))
            nChannels_ += growthRate
        layers.append(nn.Conv2d(nChannels_ - growthRate, nChannels_def, kernel_size=1, padding=0, bias=False))
        layers.append(nn.Conv2d(nChannels_def, dim1, kernel_size=1, padding=0, bias=True))
        layers.append(nn.Conv2d(dim1, dim1, kernel_size=3, padding=1, bias=True))         # global feature fusion (GFF)
        nChannels_ = dim1

    return nn.Sequential(*layers) 

# def make_hg_layer_revr(kernel, mod, dim0, dim1, nDenselayer):
#     #nDenselayer = 2             # Hyperparameters, but treat as a constant first
#     growthRate = dim0 // 2        # Hyperparameters, but treat as a constant first
#     nChannels = int(growthRate * 2)
#     layers = []
#     nChannels_def = nChannels
#     nChannels_ = nChannels
#     print(dim0)
#     print(dim1)

#     for _ in range(mod):
#         for i in range(nDenselayer):
#             layers.append(RDB(nChannels = nChannels_, nDenselayer = nDenselayer, growthRate = nChannels_))
#             nChannels_ += growthRate
#             print("hello")
#         layers.append(nn.Conv2d(nChannels_ - growthRate, nChannels_def, kernel_size=1, padding=0, bias=False))
#         layers.append(nn.Conv2d(nChannels_def, dim1, kernel_size=1, padding=0, bias=True))
#         layers.append(nn.Conv2d(dim1, dim1, kernel_size=3, padding=1, bias=True))         # global feature fusion (GFF)

#     #print(layers)

#     return nn.Sequential(*layers) 

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(int(nChannels), int(growthRate), kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        out = nn.functional.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        self.layer = self._make_layer(nChannels, nDenselayer, growthRate)
    def _make_layer(self, nChannels, nDenselayer, growthRate):
        #modules = []
        #nChannels_def = nChannels
        #nChannels_ = nChannels
        #for i in range(nDenselayer):    
        module = make_dense(nChannels, growthRate)

        #modules.append(nn.Conv2d(nChannels_, nChannels_def, kernel_size=1, padding=0, bias=False))
        #self.dense_layers = nn.Sequential(*modules)    
        #self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        return module

    def forward(self, x):
        #out = self.dense_layers(x)
        #out = self.conv_1x1(out)
        #out = out + x
        return self.layer(x)