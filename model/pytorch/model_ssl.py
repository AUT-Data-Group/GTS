import os
import math
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from logging import config
from einops import rearrange, reduce, repeat
from IPython.display import display
DEBUG = os.environ.get("DEBUG")

try:
    from . import data_transform as T
    from .transformer import PatchEmbed, TransformerContainer, ClassificationHead
except ImportError as e:
    import data_transform as T
    from transformer import PatchEmbed, TransformerContainer, ClassificationHead



logger = logging.getLogger("Model")

logger.setLevel("INFO")

log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            # "filename":"all_messages.log"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}

config.dictConfig(log_config)


class ViViTSSL(nn.Module):

    """ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
        <https://arxiv.org/abs/2103.15691>
    Input dim: torch.Size([12, 64, 414])
     outpur dim: torch.Size([12, 64, 414])

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to 12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_p (float): Probability of dropout layer. Defaults to 0..
        tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
        conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
            Defaults to 'fact_encoder'.
        norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
        copy_strategy (str): Copy or Initial to zero towards the new additional layer.
        extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
        use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
        return_cls_token (bool): Whether to use cls_token to predict class label.
    """
    supported_attention_types = [
        'fact_encoder', 'joint_space_time', 'divided_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 embed_dims=768,
                 num_heads=1,
                 num_transformer_layers=1,
                 in_channels=2,
                 dropout_p=0.,
                 tube_size=2,
                 conv_type='Conv2d',
                 attention_type='fact_encoder',
                 norm_layer=nn.LayerNorm,
                 return_cls_token=True,
                 horizon=12,
                 mask_ratio=0.3,
                 mode="spatial",
                 **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')

        # num_frames = num_frames//tube_size
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.tube_size = tube_size
        self.num_time_transformer_layers = 4
        self.return_cls_token = return_cls_token
        logging.info(f"num_frames = {self.num_frames}, embed_dims = {self.embed_dims},"
        f"num_transformer_layers = {self.num_transformer_layers},"
        f"attention_type = {self.attention_type},"  
        f"conv_type = {self.conv_type}," 
        f"tube_size = {self.tube_size},"
        f"return_cls_token= {self.return_cls_token}")
        #tokenize & position embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            tube_size=tube_size,
            conv_type=conv_type)
        num_patches = self.patch_embed.num_patches

        # Divided Space Time Transformer Encoder - Model 2
        transformer_layers = nn.ModuleList([])

        spatial_transformer = TransformerContainer(
            num_transformer_layers=num_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        temporal_transformer = TransformerContainer(
            num_transformer_layers=self.num_time_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        transformer_layers.append(spatial_transformer)
        transformer_layers.append(temporal_transformer)

        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
        # whether to add one cls_token in temporal pos_enb
        num_frames = num_frames + 1
        num_patches = num_patches + 1
        if mode == "spatial":
            num_patches -=1
            num_patches *= (1-self.mask_ratio)
            num_patches = int(num_patches)
            num_patches += 1
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
        if mode == "temporal":
            self.time_embed = nn.Parameter(torch.zeros(1,int((1 - mask_ratio)*num_frames),embed_dims))
        else:
            self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(768, 414)
        self.conv = nn.Conv2d(1, 12, kernel_size=1)
        # self.decoder_embed = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(embed_dims, 414, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        #  norm_layer(decoder_embed_dim, patch_size**2 * in_chans, bias=True) steal from source code
        self.mode = mode

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        # if npatch == N and w == h:
        if npatch == N:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        print(patch_pos_embed.shape, "^^^^^", N)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def encode(self, x, b, t, c, h, w):
        # t, b, Y = x.shape
        # c, h, w = 2, Y//2, 1
        x = self.patch_embed(x)

        # Add Position Embedding
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.drop_after_pos(x)

        # fact encoder - CRNN style
        spatial_transformer, temporal_transformer, = *self.transformer_layers,
        x = spatial_transformer(x)

        # Add Time Embedding
        cls_tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
        x = reduce(x, 'b t p d -> b t d', 'mean')
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.time_embed
        x = self.drop_after_time(x)

        x = temporal_transformer(x)

        x = self.norm(x)
        # # Return Class Token
        # if self.return_cls_token:
        #     return x[:, 0]
        # else:
        #     return x[:, 1:].mean(1)
        z = x[:, 0]
        return z

    def random_vertex_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        D = D//2
        x = x.reshape(N, L, D,2)
        
        len_keep = int(D * (1 - mask_ratio))
        
        noise = torch.rand(N, D, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:,:len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, L, 2).reshape(N, L, len_keep, 2))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L, D, 2], device=x.device)
        mask[:,:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, L, 2).reshape(N, L, D, 2))

        return x_masked.view(N, L, -1), mask.view(N, L, -1), ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        t, b, Y = x.shape
        x = x.reshape(b, t, Y)
        METHODS = {
            "temporal": self.random_masking,
            "spatial": self.random_vertex_masking
        }
        x, mask, ids_restore = METHODS[self.mode](x, mask_ratio)
        b,t, Y = x.shape
        c, h, w = 2, Y//2, 1
        x = x.reshape(b, t, c, h, w)
        z = self.encode(x, b, t, c, h, w)
        return z, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        logger.info(f"Decode x shape = {x.shape}")
        # x = self.decoder_embed(x)

        # # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # # add pos embed
        # x = x + self.decoder_pos_embed

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)

        # # predictor projection
        # x = self.decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]
        fc = self.decoder_pred(x)
        return rearrange(self.conv(rearrange(fc.unsqueeze(1), "b x (n c) -> b x n c", n=207, c=2)), "b x n c -> b x (n c)")

    def forward_loss(self, x, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        """
        x.shape
        torch.Size([12, 64, 414])
        (Pdb) pred.shape
        torch.Size([64, 8, 414])
        (Pdb) mask.shape
        torch.Size([64, 12])
        """
        t, b, Y = x.shape
        x = x.reshape(b, t, Y)
        target = x
        if True:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # torch.Size([8, 64, 414])
        # torch.Size([64, 12, 414])
        loss = (pred - target) ** 2
        if self.mode != "spatial":
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio):
        # # Return Class Token
        # if self.return_cls_token:
        #     return x[:, 0]
        # else:
        #     return x[:, 1:].mean(1)
        # ==================================
        # z = x[:, 0].clone()
        # fc = self.fc(x[:, 0])
        # recon = rearrange(self.conv(rearrange(fc.unsqueeze(1), "b x (n c) -> b x n c", n=207, c=2)), "b x n c -> x b (n c)")
        # return F.mse_loss(recon[:, t//3: t//2, :, :], masked_gt), z
        # torch.Size([12, 64, 414])
        if not DEBUG:
            wandb.config.mask_ratio = mask_ratio
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask




class ViViTComplete(nn.Module):

    def __init__(self, ckp):
        self.encoder = self.load_encoder(ckp)
        self.fc = nn.Linear(768, 414)
        self.conv = nn.Conv2d(1, 12, kernel_size=1)
    
    def load_encoder(self, ckp):
        checkpoint = torch.load(ckp)
        model = ViViTSSL(12,207,1, in_channels=2, mask_ratio=0.3, mode="spatial")
        model.load_state_dict(checkpoint["model_state_dict"])
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Identity()
        model.conv = nn.Identity()
        return model
        

    def forward(self, x):
        x = self.encoder(x)
        fc = self.fc(x[:, 0])
        return rearrange(self.conv(rearrange(fc.unsqueeze(1), "b x (n c) -> b x n c", n=207, c=2)), "b x n c -> x b (n c)")