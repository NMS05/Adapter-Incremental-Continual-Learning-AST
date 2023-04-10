import torch
import torch.nn as nn

class Masked_Attention(nn.Module):
    def __init__(self, A, f_dim, t_dim, dim=768, num_heads=8):
        super(Masked_Attention, self).__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = original attention layer with fully trained QKV weights and projection weights
        self.qkv = A.qkv
        self.attn_drop = nn.Dropout(0.0)
        self.proj = A.proj

        # masking params
        self.f_dim = f_dim
        self.t_dim = t_dim

        mask = self.get_mask(self.f_dim, self.t_dim) # shape = (N,N) where N = f_dim * t_dim
        self.mask = nn.functional.pad(mask,(1,0,1,0),value=1.0) # for cls token

    def get_mask(self, fdim, tdim):

        mask = []
        index=0

        for i in range(fdim):
          for j in range(tdim):

            mask.append([])
            for m in range(fdim):
              for n in range(tdim):

                if m == i or n ==j:
                  mask[index].append(1)
                else:
                  mask[index].append(0)

            index += 1

        mask = torch.cuda.HalfTensor(mask)
        return mask

    def masked_attn(self,q,k,mask):
        # shape of q & k = (BS, 8, N, 96)
        # shape of k.transpose(-2, -1) = (BS, 8, 96, N)
        BS, num_heads, _, _ = q.shape
        attn = (q @ k.transpose(-2, -1))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(BS, num_heads, -1, -1)
        attn = torch.mul(attn,mask)
        return attn
    
    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x) # shape=(B,N,768*3)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # shape = (3, B, num_heads, N, 96)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # shape = (B, num_heads, N, 96)

        # attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, 96) @ (B, num_heads, 96, N) = (B, num_heads, N, N)
        attn = self.masked_attn(q,k,self.mask) * self.scale # (B, num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, num_heads, N, N)@ (B, num_heads, N, 768) = (B, num_heads, N, 96) -->> (B, N, num_heads, 96) -->> (B, N, 768)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class conv_pass(nn.Module):
    def __init__(self, dim):
        super(conv_pass, self).__init__()

        # Adapter params
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # 2D-CNN for spectrogram
        self.spec_down = nn.Linear(768, dim)
        self.spec_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.zeros_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)
        self.spec_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.spec_conv.weight)
        nn.init.zeros_(self.spec_conv.bias)


    def forward(self, x, f_dim, t_dim):
        B, N, C = x.shape

        x_down = self.spec_down(x)
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, f_dim, t_dim, self.dim).permute(0, 3, 1, 2)
        x_patch = self.spec_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, f_dim * t_dim, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.spec_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.spec_up(x_down)

        return x_up


class ConvPass(nn.Module):
    def __init__(self, Encoder, dim, f_dim, t_dim):
        super(ConvPass, self).__init__()

        self.f_dim = f_dim
        self.t_dim = t_dim

        # Attention Layer
        self.norm1 = Encoder.norm1
        self.attn = Masked_Attention(Encoder.attn, self.f_dim, self.t_dim)

        # Feed Forward Layers
        self.norm2 = Encoder.norm2
        self.mlp = Encoder.mlp

        # Conv Adapter
        self.conv1 = conv_pass(dim=dim)
        self.conv2 = conv_pass(dim=dim)

    def forward(self, x):
        # Attn skip connections
        x = x + self.attn(self.norm1(x)) + self.conv1(self.norm1(x), self.f_dim, self.t_dim)
        # FFN + skip conections
        x = x + self.mlp(self.norm2(x)) + self.conv2(self.norm2(x), self.f_dim, self.t_dim)
        return x