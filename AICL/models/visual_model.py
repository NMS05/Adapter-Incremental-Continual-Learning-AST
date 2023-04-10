import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple
from models.pet_modules import ConvPass

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



"""
AST Model with Convolutional Adapters (parameter-efficiency) and Frequency-Time Factorized Attention (compute-efficiency) 
"""
class ASTModel(nn.Module):
    def __init__(self, label_dim, input_fdim, input_tdim, unsqueeze=False, fstride=10, tstride=10):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        
        #######################################################################################################################################
        # <donot modify>
        #######################################################################################################################################
        self.unsqueeze = unsqueeze # required for SpeechCommands and AVE

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.v = timm.create_model('vit_base_patch16_384', pretrained=True)

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Linear(self.original_embedding_dim, label_dim)

        # automatically get the intermediate shape
        self.f_dim, self.t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = self.f_dim * self.t_dim
        self.v.patch_embed.num_patches = num_patches

        # the conv projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj


        # the positional embedding
        # get the positional embedding from vit model, skip the first cls tokens, reshape it to original 2D shape (24*24).
        new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)

        # cut (from middle) or interpolate the second dimension of the positional embedding
        if self.t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(self.t_dim / 2): int(self.oringal_hw / 2) - int(self.t_dim / 2) + self.t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, self.t_dim), mode='bilinear')

        # cut (from middle) or interpolate the first dimension of the positional embedding
        if self.f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(self.f_dim / 2): int(self.oringal_hw / 2) - int(self.f_dim / 2) + self.f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.f_dim, self.t_dim), mode='bilinear')

        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))
        
        #######################################################################################################################################
        # <\donot modify>
        #######################################################################################################################################

        # Freeze Parameters
        for p in self.v.parameters(): p.requires_grad=False # Freeze all params

        # CONVOLUTIONAL ADAPTER
        for i in range(12): self.v.blocks[i] = ConvPass(self.v.blocks[i], 64, self.f_dim, self.t_dim)

    #######################################################################################################################################
    #######################################################################################################################################

    def get_shape(self, fstride, tstride, input_fdim, input_tdim):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):

        if self.unsqueeze==True: x = x.unsqueeze(1)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        x = x[:, 0]

        x = self.mlp_head(x)
        return x