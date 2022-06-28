import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from logging import config
from einops import rearrange, reduce, repeat
from IPython.display import display

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


class ViViT(nn.Module):

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
                 **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')

        # num_frames = num_frames//tube_size
        self.num_frames = num_frames
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

        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
        self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(768, 414)
        self.conv = nn.Conv2d(1, 12, kernel_size=1)
    
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
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_mask(self):
        pass

    def forward_encoder(self, ):
        pass

    def forward_decoder(self):
        pass

    def forward_loss(self):
        pass

    def forward(self, x, mask_ratio=0.3):
        #Tokenize
        t, b, Y = x.shape
        c, h, w = 2, Y//2, 1
        x = x.reshape(b, t, c, h, w)
        masked_gt = x[:, t//3: t//2, :, :].clone()
        x = x.clone()
        x[:,t//3: t//2,:,:] = 2 * 117.0/255.0 - 1.0
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
        z = x[:, 0].clone()
        fc = self.fc(x[:, 0])
        recon = rearrange(self.conv(rearrange(fc.unsqueeze(1), "b x (n c) -> b x n c", n=207, c=2)), "b x n c -> x b (n c)")
        return F.mse_loss(recon[:, t//3: t//2, :, :], masked_gt), z

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask