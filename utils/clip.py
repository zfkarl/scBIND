import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import os
import numpy as np
from dataclasses import dataclass
from .lightning import LitModule
from .vit import ViTConfig, ViTModel
import time
from tqdm import tqdm
from anndata import AnnData, concat
from utils.plot import plot_umap, plot_paired_umap
from utils.metrics import matching_metrics
from utils.logger import create_logger
import scanpy as sc
import torchmetrics

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    cell_loss = contrastive_loss(similarity)
    text_loss = contrastive_loss(similarity.T)
    return (cell_loss + text_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )

    def forward(self, cell_embeds, text_embeds):
        # normalized features
        # cell_embeds = cell_embeds / cell_embeds.norm(dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_cell = torch.matmul(cell_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_cell.T

        loss = clip_loss(logits_per_cell)

        return loss, logits_per_cell  # , logits_per_text


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


class CLIPModel(LitModule):
    def __init__(
        self,
        config,
        cell_config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.cell_model = ViTModel(cell_config)


        self.cell_projection = nn.Linear(
            self.cell_model.config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.cell_model.config.text_emb_size+self.cell_model.config.gene_emb_size, config.projection_dim, bias=False
        )

        self.criterion = CLIPLoss(
            self.config.logit_scale, requires_grad=self.config.requires_grad
        )


    def forward(
        self,
        cell_values=None,
        text_values=None,
        gene_values=None
    ):
        cell_embeds = self._get_cell_features(cell_values)
        text_embeds = self._get_text_features(text_values,gene_values)

        return cell_embeds, text_embeds

    def _step(self, batch, batch_idx, mode):  

        cell_values, text_values, gene_values, labels = batch[0], batch[1], batch[2], batch[3]
        # print('cell:',cell_values.dtype)
        # print('text:',text_values.dtype)
        # print('label:',labels.dtype)

        cell_embeds, text_embeds =self.forward(cell_values, text_values,gene_values)
        # print('cell_embeds:',cell_embeds.dtype)
        # print('text_embeds:',text_embeds.dtype)
        
        loss, similarity = self.criterion(cell_embeds, text_embeds)

        acc, matchscore, foscttm = matching_metrics(similarity)
        log_dict = {
            f"acc/{mode}": acc,
            f"matchscore/{mode}": matchscore,
            f"foscttm/{mode}": foscttm,
            f"loss/{mode}": loss,
        }

        # logit_scale learnable
        if self.config.requires_grad:
            log_dict.update({"logit_scale": self.criterion.logit_scale})

        if mode == "predict":
            return cell_embeds, text_embeds, log_dict

        # log_dict.update({f'loss/{mode}': loss})
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    def _get_text_features(self, text_values=None,gene_values=None):

        text_emb = torch.cat((text_values,gene_values),dim=-1)
        # print('text_values:',text_values.shape)
        # print('gene_values:',gene_values.shape)
        # print('text emb:', text_emb.shape)
        
        text_features = self.text_projection(text_emb)

        if self.config.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _get_cell_features(self, cell_values=None):
        cell_outputs = self.cell_model(cell_values)

        cell_features = cell_outputs[1]  # pooled_output
        cell_features = self.cell_projection(cell_features)

        if self.config.normalize:
            cell_features = cell_features / cell_features.norm(dim=-1, keepdim=True)

        return cell_features

    def _get_batch_features(self, dataloader, modality = "cell", out_dir="." ):

        self.to("cuda")

        fn = self._get_cell_features if modality == "cell" else self._get_text_features
        if modality == "cell":
            adata = torch.concat(
                [
                    fn(batch[0].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()
        elif modality == "text":
            adata = torch.concat(
                [
                    fn(batch[1].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()

        cell_type = torch.concat(
                [
                    fn(batch[2].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()

        adata = AnnData(adata)
        sc.settings.figdir = out_dir
        plot_umap(adata, color=cell_type, metric="cosine", save=f"_{modality}.png")
        adata.write(f"{out_dir}/{modality}.h5ad")
        return adata

    def get_batch_features(
        self, dataloader,  out_dir="."
    ):
        log = create_logger("", fh=out_dir + "/log.txt")
        if not self.config.normalize:
            out_dir = f"{out_dir}_no_norm"

        if dataloader is not None:
            cell_embeds = self._get_batch_features(
                dataloader, modality="cell", out_dir=out_dir
            )
            text_embeds = self._get_batch_features(
                dataloader, modality="text", out_dir=out_dir
            )

            acc, match_score, foscttm = matching_metrics(
                x=cell_embeds.obsm["X_umap"], y=text_embeds.obsm["X_umap"]
            )
            if log is not None:
                log.info(
                    f"cell-text\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}"
                )
            else:
                print(
                    f"cell-text\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}",
                    flush=True,
                )
            concat_embeds = concat(
                [cell_embeds, text_embeds],
                label="modality",
                keys=["cell", "text"],
                index_unique="#",
            )
            sc.settings.figdir = out_dir
            if dataloader is not None:
                plot_umap(
                    concat_embeds,
                    color=["modality"],
                    metric="cosine",
                    save="_concat.png",
                )
            concat_embeds.write(f"{out_dir}/concat.h5ad")


class Classifier(LitModule):
    def __init__(
        self,
        num_classes,
        config,
        cell_config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.cell_model = ViTModel(cell_config)


        self.cell_projection = nn.Linear(
            self.cell_model.config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.cell_model.config.text_emb_size+self.cell_model.config.gene_emb_size, config.projection_dim, bias=False
        )

        
        self.classification_head = nn.Linear(config.projection_dim, num_classes)
        # self.classification_head = nn.Sequential(
        #     nn.Linear(config.projection_dim, 64),  
        #     nn.ReLU(),  
        #     nn.Linear(64, num_classes)  
        # )
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes,task="multiclass")
        self.precision = torchmetrics.Precision(num_classes=num_classes,task="multiclass",average = "macro")
        self.recall = torchmetrics.Recall(num_classes=num_classes,task="multiclass",average = "macro")
        self.f1 = torchmetrics.F1Score(num_classes=num_classes,task="multiclass",average = "macro")
        self.auroc = torchmetrics.AUROC(num_classes=num_classes,task="multiclass",average = "macro")


    def forward(
        self,
        cell_values=None,
        text_values=None,
        gene_values=None
    ):
        cell_embeds = self._get_cell_features(cell_values)
        text_embeds = self._get_text_features(text_values, gene_values)

        return cell_embeds, text_embeds

    def training_step(self, batch, batch_idx):  

        cell_values, text_values, gene_values,labels = batch[0], batch[1], batch[2], batch[3]
        cell_embeds, text_embeds =self.forward(cell_values, text_values, gene_values)
        logits = self.classification_head(cell_embeds)
        
        loss = self.criterion(logits, labels)

        log_dict = {
            f"training loss": loss,
        }

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):  

        cell_values, text_values, gene_values,labels = batch[0], batch[1], batch[2], batch[3]
        cell_embeds, text_embeds =self.forward(cell_values, text_values, gene_values)
        logits = self.classification_head(cell_embeds)
        
        loss = self.criterion(logits, labels)

        accuracy = self.accuracy(logits, labels)

        log_dict = {
            f"validation loss": loss,
            f"validation acc": accuracy,
        }

        
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

        return loss
    
    
    def test_step(self, batch, batch_idx):  

        cell_values, text_values, gene_values,y = batch[0], batch[1], batch[2], batch[3]
        cell_embeds, text_embeds =self.forward(cell_values, text_values, gene_values)
        y_pred = self.classification_head(cell_embeds)
        
        accuracy = self.accuracy(y_pred, y)
        precision = self.precision(y_pred, y)
        recall = self.recall(y_pred, y)
        f1 = self.f1(y_pred, y)
        auroc = self.auroc(y_pred, y)

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_auroc", auroc)
    

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    def _get_text_features(self, text_values=None,gene_values=None):

        text_emb = torch.cat((text_values,gene_values),dim=1)
        
        text_features = self.text_projection(text_emb)

        if self.config.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _get_cell_features(self, cell_values=None):
        cell_outputs = self.cell_model(cell_values)

        cell_features = cell_outputs[1]  # pooled_output
        cell_features = self.cell_projection(cell_features)

        if self.config.normalize:
            cell_features = cell_features / cell_features.norm(dim=-1, keepdim=True)

        return cell_features




class CLIPModel_Cell_Only(LitModule):
    def __init__(
        self,
        config,
        cell_config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.cell_model = ViTModel(cell_config)


        self.cell_projection = nn.Linear(
            self.cell_model.config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.cell_model.config.text_emb_size, config.projection_dim, bias=False
        )

        self.criterion = CLIPLoss(
            self.config.logit_scale, requires_grad=self.config.requires_grad
        )


    def forward(
        self,
        cell_values=None,
        text_values=None,
        gene_values=None
    ):
        cell_embeds = self._get_cell_features(cell_values)
        text_embeds = self._get_text_features(text_values,gene_values)

        return cell_embeds, text_embeds

    def _step(self, batch, batch_idx, mode):  

        cell_values, text_values, gene_values, labels = batch[0], batch[1], batch[2], batch[3]
        # print('cell:',cell_values.dtype)
        # print('text:',text_values.dtype)
        # print('label:',labels.dtype)

        cell_embeds, text_embeds =self.forward(cell_values, text_values,gene_values)
        # print('cell_embeds:',cell_embeds.dtype)
        # print('text_embeds:',text_embeds.dtype)
        
        loss, similarity = self.criterion(cell_embeds, text_embeds)

        acc, matchscore, foscttm = matching_metrics(similarity)
        log_dict = {
            f"acc/{mode}": acc,
            f"matchscore/{mode}": matchscore,
            f"foscttm/{mode}": foscttm,
            f"loss/{mode}": loss,
        }

        # logit_scale learnable
        if self.config.requires_grad:
            log_dict.update({"logit_scale": self.criterion.logit_scale})

        if mode == "predict":
            return cell_embeds, text_embeds, log_dict

        # log_dict.update({f'loss/{mode}': loss})
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    def _get_text_features(self, text_values=None,gene_values=None):

        #text_emb = torch.cat((text_values,gene_values),dim=1)
        
        text_features = self.text_projection(text_values)

        if self.config.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _get_cell_features(self, cell_values=None):
        cell_outputs = self.cell_model(cell_values)

        cell_features = cell_outputs[1]  # pooled_output
        cell_features = self.cell_projection(cell_features)

        if self.config.normalize:
            cell_features = cell_features / cell_features.norm(dim=-1, keepdim=True)

        return cell_features

    def _get_batch_features(self, dataloader, modality = "cell", out_dir="." ):

        self.to("cuda")

        fn = self._get_cell_features if modality == "cell" else self._get_text_features
        if modality == "cell":
            adata = torch.concat(
                [
                    fn(batch[0].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()
        elif modality == "text":
            adata = torch.concat(
                [
                    fn(batch[1].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()

        cell_type = torch.concat(
                [
                    fn(batch[2].to("cuda")).detach().cpu()
                    for batch in tqdm(dataloader, desc=modality)
                ]
            ).numpy()

        adata = AnnData(adata)
        sc.settings.figdir = out_dir
        plot_umap(adata, color=cell_type, metric="cosine", save=f"_{modality}.png")
        adata.write(f"{out_dir}/{modality}.h5ad")
        return adata

    def get_batch_features(
        self, dataloader,  out_dir="."
    ):
        log = create_logger("", fh=out_dir + "/log.txt")
        if not self.config.normalize:
            out_dir = f"{out_dir}_no_norm"

        if dataloader is not None:
            cell_embeds = self._get_batch_features(
                dataloader, modality="cell", out_dir=out_dir
            )
            text_embeds = self._get_batch_features(
                dataloader, modality="text", out_dir=out_dir
            )

            acc, match_score, foscttm = matching_metrics(
                x=cell_embeds.obsm["X_umap"], y=text_embeds.obsm["X_umap"]
            )
            if log is not None:
                log.info(
                    f"cell-text\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}"
                )
            else:
                print(
                    f"cell-text\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}",
                    flush=True,
                )
            concat_embeds = concat(
                [cell_embeds, text_embeds],
                label="modality",
                keys=["cell", "text"],
                index_unique="#",
            )
            sc.settings.figdir = out_dir
            if dataloader is not None:
                plot_umap(
                    concat_embeds,
                    color=["modality"],
                    metric="cosine",
                    save="_concat.png",
                )
            concat_embeds.write(f"{out_dir}/concat.h5ad")