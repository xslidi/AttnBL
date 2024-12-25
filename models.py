import torch
import torch.nn as nn
from layers import *


class Coefficients(nn.Module):
    def __init__(self, params, c_in=3, nf=16):
        super(Coefficients, self).__init__()
        self.params = params
        self.relu = nn.ReLU()

        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, nf,  kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat2 = conv_layer(nf, nf*2, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.merge = PatchEmbed(2, nf*2, nf*4, norm_layer=nn.LayerNorm)
        self.merge1 = PatchEmbed(2, nf*4, nf*8, norm_layer=nn.LayerNorm)
        self.transformer = Transformer(nf*8, 1, 4, nf*8, input_resolution=16, position_bias=False)
        self.transformer1 = Transformer(nf*4, 1, 8, nf*4, input_resolution=32, position_bias=False)
        self.transformer2 = Transformer(nf*2, 1, 8, nf*2, input_resolution=64, position_bias=False, window_size=16)
        # ===========================predicton===========================
        self.pred = conv_layer(nf*8, 144, kernel_size=1,padding=0, activation=None) # 128 -> 144

    def forward(self, x):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x) # N, C=16,  H=128, W=128
        x = self.splat2(x) # N, C=32, H=64,  W=64
        x = self.transformer2(x)
        x = self.merge(x)  # N, C=64, H=32,  W=32
        x = self.transformer1(x)
        x = self.merge1(x) # N, C=128, H=16,  W=16
        x = self.transformer(x)
        # ===========================Prediction===========================
        x = self.pred(x) # N, C=128, H=16, W=16
        x = x.view(N, 12, 12, 16, 16) # N, C=12, D=12, H=16, W=16

        return x

class Coefficients_sim(nn.Module):
    def __init__(self, params, c_in=3, nf=16):
        super(Coefficients_sim, self).__init__()
        self.params = params
        self.relu = nn.ReLU()
        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, nf,  kernel_size=7, stride=2, padding=3, activation=nn.LeakyReLU, batch_norm=False)
        self.splat2 = conv_layer(nf, nf*2, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU, batch_norm=params['batch_norm'])
        self.splat21 = conv_layer(nf*2, nf*2, kernel_size=3, padding=1, activation=nn.LeakyReLU, batch_norm=params['batch_norm'])
        self.splat3 = conv_layer(nf*2, nf*4, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU, batch_norm=params['batch_norm'])
        self.splat31 = conv_layer(nf*4, nf*4, kernel_size=3, padding=1, activation=nn.LeakyReLU, batch_norm=params['batch_norm'])
        self.splat4 = conv_layer(nf*4, nf*8, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU, batch_norm=params['batch_norm'])
        # ===========================Local===========================
        self.local1 = conv_layer(nf*8, nf*8, kernel_size=3, padding=1, batch_norm=False)
        # ===========================predicton===========================
        self.pred = conv_layer(nf*8, 144, kernel_size=1,padding=0, activation=None) # 128 -> 144

    def forward(self, x):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x) # N, C=8,  H=128, W=128
        x = self.splat2(x) # N, C=16, H=64,  W=64
        x = self.splat21(x)
        x = self.splat3(x)
        x = self.splat31(x)
        x = self.splat4(x)
        x = self.local1(x)
        # ===========================Prediction===========================
        x = self.pred(x) # N, C=128, H=16, W=16
        x = x.view(N, 12, 12, 16, 16) # N, C=12, D=12, H=16, W=16

        return x

class Guide(nn.Module):
    def __init__(self, params, c_in=3):
        super(Guide, self).__init__()
        self.params = params
        # Number of relus/control points for the curve
        self.c_in = c_in

        self.conv1x1 = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.conv1x1.weight = torch.nn.Parameter(
           (torch.eye(c_in, dtype=torch.float32) + torch.randn(1, dtype=torch.float32) * 1e-4).reshape(c_in,c_in,1,1)
        )
        self.conv1x1_3 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.relu = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        x1 = self.relu(self.conv1x1(x))
        x3 = self.relu(self.conv1x1_3(x)) 
        x = x1 + x3
        
        x = torch.sum(x, dim=1, keepdim=True) / self.c_in # N, C=1, H, W
        x = x + self.bias # N, C=1, H, W
        x = torch.clamp(x, 0, 1) # N, C=1, H, W

        return x


class Guide_sim(nn.Module):
    def __init__(self, params, c_in=3):
        super(Guide_sim, self).__init__()
        self.c_in = c_in
        self.conv1x1 = nn.Conv2d(c_in, c_in, 1, bias=False)
        weight = torch.tensor([[0.2126, 0, 0],
                               [0, 0.7152, 0],
                               [0, 0, 0.0722]], dtype=torch.float32)
        self.conv1x1.weight = torch.nn.Parameter(
           (nn.Parameter(weight) + torch.randn(1, dtype=torch.float32) * 1e-4).reshape(c_in,c_in,1,1)
        )
        self.conv1x1_2 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.relu = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))
    def forward(self, x):
        x = self.conv1x1(x) + self.relu(self.conv1x1_2(x))
        x = torch.sum(x, dim=1, keepdim=True)# N, C=1, H, W
        x = x + self.bias # N, C=1, H, W
        x = torch.clamp(x, 0, 1) # N, C=1, H, W
        return x
    
class Guide_s(nn.Module):
    def __init__(self, params, c_in=3):
        super(Guide_s, self).__init__()
        self.c_in = c_in
        self.conv1x1 = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.conv1x1.weight = torch.nn.Parameter(
           (torch.eye(c_in, dtype=torch.float32) + torch.randn(1, dtype=torch.float32) * 1e-4).reshape(c_in,c_in,1,1)
        )
        self.relu = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))
    def forward(self, x):
        x = self.relu(self.conv1x1(x))
        x = torch.sum(x, dim=1, keepdim=True) / self.c_in # N, C=1, H, W
        x = x + self.bias # N, C=1, H, W
        x = torch.clamp(x, 0, 1) # N, C=1, H, W
        return x

class HDRnetModel(nn.Module):
    def __init__(self, params):
        super(HDRnetModel, self).__init__()
        self.coefficients = Coefficients(params, nf=16)
        self.guide = Guide(params)
    def forward(self, lowres, fullres, encode_only=False):
        grid = self.coefficients(lowres)
        guide = self.guide(fullres)
        sliced = slicing(grid, guide)
        output = apply(sliced, fullres, encode=False)
        
        return output


if __name__ == "__main__":
    from torchinfo import summary
    params = {'batch_size':1, 'batch_norm': False}
    model = HDRnetModel(params)
    batch_size = 1
    summary(model, input_size=[(batch_size, 3, 256, 256), (batch_size, 3, 3840, 2160)])
