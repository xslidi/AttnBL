import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, act_layer=nn.GELU, drop=0.) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hid_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hid_dim, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


    
def window_partition(input, original_size, window_size=(7, 7), mode='window') -> torch.Tensor:
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, N, C].
        original_size (Tuple[int, int]): The original size of the input feature. Must be revisable to window_size.
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
        mode ('window' | 'grid', optional): the resize mode
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, N, C = input.shape
    H, W = original_size 
    assert H*W == N, "flatten img_tokens has wrong size"
    if mode == 'window':
        # Unfold input
        windows = input.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    elif mode == 'grid':
        # Unfold input
        windows = input.view(B, window_size[0], H // window_size[0], window_size[1], W // window_size[1], C)
        # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
        windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, window_size[0], window_size[1], C)

    windows = windows.view(-1, window_size[0]*window_size[1], C)
    return windows

def window_reverse(windows, original_size, window_size=(7, 7), mode='window') -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    if mode == 'window':
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)       
    elif mode == 'grid':   
        output = output.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
        
    output = output.view(B, H*W, -1)
    return output


class Attention_naive(nn.Module):
    def __init__(self, dim, num_heads=8, input_resolution=64, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., position_bias=False, channel_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        input_resolution = (input_resolution, input_resolution)
        self.input_resolution = input_resolution
        self.position_bias = position_bias
        self.channel_attention = channel_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, relative_pos=None, original_size=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if not self.channel_attention:
            q = q * self.scale                 # B, num_heads, N, C // num_heads
            attn = (q @ k.transpose(-2, -1))   # B, num_heads, N, N

            if relative_pos is not None:
                attn = attn + relative_pos

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N) 

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            q = q * self.scale                   # B, num_heads, N, C // num_heads
            attn = (q.transpose(-1, -2) @ k)     # B, num_heads, C // num_heads, C // num_heads

            if relative_pos is not None:
                attn = attn + relative_pos

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v.transpose(-1, -2)).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hid_dim, input_resolution=64, dropout=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, position_bias=False, attention_on=True, window_size=8, shift_size=0, partition_function=None, reverse_function=None, mode='window', channel_attention=False) -> None:
        super().__init__()
        self.attention_on = attention_on
        if attention_on:
            self.norm1 = norm_layer(embed_dim)
            self.attention = Attention_naive(embed_dim, num_heads, input_resolution, position_bias=position_bias, channel_attention=channel_attention)

        self.window_size = (window_size, window_size)
        self.mode = mode
        if window_size > 0:
            self.partition_function = partition_function
            self.reverse_function = reverse_function

        self.shift_size = shift_size
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H = W = int(input_resolution)
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            img_mask = img_mask.view(1, H * W, 1)
            mask_windows = window_partition(img_mask, (input_resolution,input_resolution), window_size=self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        self.mlp = MLP(embed_dim, embed_dim, hid_dim, act_layer, dropout)
        self.norm2 = norm_layer(embed_dim)

    def forward(self, x, relative_pos=None, original_size=None):
        B, N, C = x.shape
        if original_size is None:        
            H = W = int(math.sqrt(N)) 
            original_size = (H, W)

        if self.window_size[0] > 0:
            x = self.partition_function(x, original_size, self.window_size, mode=self.mode)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))         

        if self.attention_on:
            x = x + self.attention(self.norm1(x), relative_pos, original_size)
        x = x + self.mlp(self.norm2(x))

        if self.window_size[0] > 0:
            x = self.reverse_function(x, original_size, self.window_size, mode=self.mode)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, depth, heads, mlp_dim, input_resolution=64, dropout=0., position_bias=False, attention=True, window_size=0. , shift_size=0.) -> None:
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            layers.append(
            TransformerBlock(in_dim, heads, mlp_dim, input_resolution, dropout, position_bias=position_bias, attention_on=attention, window_size=window_size, shift_size=shift_size, partition_function=window_partition, reverse_function=window_reverse)
            )                                

        self.layers = layers 
        self.relative_pos = None

    def forward(self, x):
        b, c, h, w = x.shape
        original_size = (h, w)
        x = x.flatten(2).transpose(1, 2)    # B, N, C
        for layer in self.layers:
            x = layer(x, self.relative_pos, original_size)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x    



class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, with_pos=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        if with_pos:
            self.pos = PA(embed_dim)
        else:
            self.pos = None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        if self.pos is not None:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.embed_dim, H // self.patch_size, W // self.patch_size)
        return x



def slicing(grid, guide):
    N, C, H, W = guide.shape
    device = grid.get_device()
    steph = 1 / (H - 1) * 2
    stepw = 1 / (W - 1) * 2
    hh, ww = torch.meshgrid(torch.arange(-1, 1+1e-8, steph, device=device), 
                            torch.arange(-1, 1+1e-8, stepw, device=device)) # H, W    
    # if device >= 0:
    #     hh, ww = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device)) # H, W
    # else:
    #     hh, ww = torch.meshgrid(torch.arange(H), torch.arange(W)) # H, W
    # # To [-1, 1] range for grid_sample
    # hh = hh / (H - 1) * 2 - 1
    # ww = ww / (W - 1) * 2 - 1
    guide = guide * 2 - 1
    hh = hh[None, :, :, None].repeat(N, 1, 1, 1) # N, H, W, C=1
    ww = ww[None, :, :, None].repeat(N, 1, 1, 1)  # N, H, W, C=1
    guide = guide.permute(0, 2, 3, 1) # N, H, W, C=1

    guide_coords = torch.cat([ww, hh, guide], dim=3) # N, H, W, 3
    # unsqueeze because extra D dimension
    guide_coords = guide_coords.unsqueeze(1) # N, Dout=1, H, W, 3
    sliced = F.grid_sample(grid, guide_coords, align_corners=False, padding_mode="border") # N, C=12, Dout=1, H, W
    sliced = sliced.squeeze(2) # N, C=12, H, W

    return sliced


def apply(sliced, fullres, encode=False):
    # r' = w1*r + w2*g + w3*b + w4
    rr = fullres * sliced[:, 0:3, :, :] # N, C=3, H, W
    gg = fullres * sliced[:, 4:7, :, :] # N, C=3, H, W
    bb = fullres * sliced[:, 8:11, :, :] # N, C=3, H, W
    rr = torch.sum(rr, dim=1) + sliced[:, 3, :, :] # N, H, W
    gg = torch.sum(gg, dim=1) + sliced[:, 7, :, :] # N, H, W
    bb = torch.sum(bb, dim=1) + sliced[:, 11, :, :] # N, H, W
    output = torch.stack([rr, gg, bb], dim=1) # N, C=3, H, W
    feature_1 = sliced[:, 0:4, :, :]
    feature_2 = sliced[:, 4:8, :, :]
    feature_3 = sliced[:, 8:, :, :]
    feats = [feature_1, feature_2, feature_3]
    if encode:
        return output, feats
    return output

    