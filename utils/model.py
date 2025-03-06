import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, Any

import pytorch_lightning as pl
from .lightning import LitModule


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, temp=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * temp, requires_grad=requires_grad
        )
        print("requires_grad", requires_grad, flush=True)

    def forward(self, atac_embeds, rna_embeds):
        # normalized features
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        similarity = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale

        loss = clip_loss(similarity)

        return loss, similarity


from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class Tokenizer(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, feature_size, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = PatchEmbeddings(
            feature_size, config.num_patches, config.hidden_size
        )

        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.num_patches = num_patches
        self.patch_size = self.patch_embeddings.patch_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        # batch_size, num_channels, height, width = pixel_values.shape
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(
            pixel_values
        )  # , interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(nn.Module):
    """
    Single-cell profile to Patch Embedding.
    """

    def __init__(self, feature_size, num_patches, hidden_size):
        super().__init__()

        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size

        self.projection = nn.Linear(patch_size, hidden_size)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        x = self.projection(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, config, feature_size):
        super().__init__()
        self.model_type = "Transformer"
        self.tokenizer = Tokenizer(config, feature_size)
        encoder_layers = TransformerEncoderLayer(
            config.hidden_size, config.num_heads, config.ffn_dim, config.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """

        src = self.tokenizer(src)
        output = self.transformer_encoder(src, src_mask)
        return output



