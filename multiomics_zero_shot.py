#!/usr/bin/env python

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping,ModelCheckpoint


from utils.dataset import CellTextDataModule_BMMC,CellTextDataModule_BMMC_pretrain
from utils.clip import CLIPModel, CLIPModel_Cell_Only
from utils.vit import ViTConfig
from utils.callback import Monitor
from utils.config import get_model_config
from utils.logger import create_logger

import os
import argparse
import torch
import numpy as np
import scvi
import scib
import scanpy as sc
from pathlib import Path

HOME = Path.home()
print("Start", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mdi")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    # Dataset
    parser.add_argument("--rna_text_path", type=str, default='dataset/sc-data/data/4o_text_embeddings_bmmc_pretrain.npy')
    parser.add_argument("--rna_text_path2", type=str, default='dataset/sc-data/data/text_embeddings_bmmc.npy')
    parser.add_argument("--gene_emb_path", type=str, default='dataset/sc-data/data/bmmc-gene-emb.npy')
    # DataModule
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="bmmc_4o")
    parser.add_argument("--experiment", action=argparse.BooleanOptionalAction, default=True)
    # Module
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup_steps", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_imputed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument( "--requires_grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--version", type=str, default="bmmc")
    parser.add_argument("--pretrain_epochs", type=int, default=500)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--logit_scale", type=float, default=1)  # 2.6592)
    parser.add_argument("--num_patches", type=int, default=128)
    parser.add_argument( "--early_stop", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.checkpoint is None:
        
        rna_data = CellTextDataModule_BMMC_pretrain(args.batch_size, args.rna_text_path, args.gene_emb_path)


        model_config = get_model_config("small")
        cell_config = ViTConfig(
            **{
                "modality": "cell",
                "num_patches": args.num_patches,
                "feature_size": rna_data.dataset.input_size,
                "text_emb_size": rna_data.dataset.text_emb_size,
                "gene_emb_size": rna_data.dataset.gene_emb_size,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        model = CLIPModel_Cell_Only(
            args,
            cell_config=cell_config,
        )


        # out_dir
        if args.experiment:
            args.default_root_dir = f"results/{args.data_dir}/{args.logit_scale}_{args.requires_grad}_{args.pretrain_epochs}_{args.lr}_{args.version}"
        else:
            args.default_root_dir = (
                f"results/{args.data_dir}/{args.logit_scale}_{args.pretrain_epochs}"
            )
        # os.makedirs(args.default_root_dir, exist_ok=True)
        print("default_root_dir:", args.default_root_dir, flush=True)

        # trainer
        logger = TensorBoardLogger(
            save_dir=args.default_root_dir, default_hp_metric=False, version=""
        )

 
        checkpoint_callback = ModelCheckpoint(
            monitor='loss/val',  
            save_last = args.default_root_dir + "/lightning_logs/checkpoints",
            dirpath= args.default_root_dir + "/lightning_logs/checkpoints",  # 模型保存路径
            filename=f'best-pretrain',  
            save_top_k=1,  
            mode='min'  
        )

        trainer = Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=1,
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_epochs=args.pretrain_epochs,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps = 1
        )

        # pretrain
        print('Pretrain on RNA Data!')
        trainer.fit(model, rna_data)
        print('Finish Pretrain on RNA Data!')
        
        model = CLIPModel_Cell_Only.load_from_checkpoint(args.default_root_dir+ '/lightning_logs/checkpoints/last.ckpt') 
        rna_data_test = CellTextDataModule_BMMC(args.batch_size, args.rna_text_path2, args.gene_emb_path)
        rna_loader = torch.utils.data.DataLoader(rna_data_test.dataset, shuffle=False,batch_size = 512, drop_last=False,num_workers=4)
        
        cell_embeds = []
        for batch in rna_loader:
            sample, text , gen_text, label = batch
            sample = sample.cuda()
            with torch.no_grad():
                cell_emb = model._get_cell_features(sample)


            cell_embeds.append(cell_emb.cpu().numpy())

        cell_embeds = np.concatenate(cell_embeds, axis=0).reshape(-1,args.projection_dim)
        
        cell_embeds = cell_embeds / np.linalg.norm(
            cell_embeds, axis=1, keepdims=True
        )
        
        adata =sc.read('dataset/sc-data/data/BMMC_multiomics.h5ad')
        
        adata.obsm['X_scBIND_zero_shot'] = cell_embeds
        adata.write('dataset/sc-data/data/BMMC_multiomics.h5ad')
        
        results = scib.metrics.metrics(
            adata,
            adata_int=adata,
            batch_key="batch_id",
            label_key="celltype",
            embed='X_scBIND_zero_shot',
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,  # use the clustering, bias to the best matching
            ari_=True,  # use the clustering, bias to the best matching
            cell_cycle_=False,
            kBET_=False,  # kBET return nan sometimes, need to examine
            ilisi_=False,
            clisi_=False,
        )
        print(results)
    else:
        model = CLIPModel_Cell_Only.load_from_checkpoint(args.checkpoint) 
        print("normalize", args.normalize, flush=True)
        model.config.normalize = args.normalize
        args.default_root_dir = args.checkpoint.split("lightning_logs/")[0]
        
        rna_data = CellTextDataModule_BMMC(args.batch_size, args.rna_text_path2, args.gene_emb_path)


        model_config = get_model_config("small")
        cell_config = ViTConfig(
            **{
                "modality": "cell",
                "num_patches": args.num_patches,
                "feature_size": rna_data.dataset.input_size,
                "text_emb_size": rna_data.dataset.text_emb_size,
                "gene_emb_size": rna_data.dataset.gene_emb_size,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        logger = TensorBoardLogger(
                save_dir=args.default_root_dir, default_hp_metric=False, version=""
            )

 
        rna_loader = torch.utils.data.DataLoader(rna_data.dataset, shuffle=False,batch_size = 512, drop_last=False,num_workers=4)
        
        cell_embeds = []
        text_embeds = []
        for batch in rna_loader:
            sample, text , gen_text, label = batch
            sample, text , gen_text, label = sample.cuda(), text.cuda() , gen_text.cuda(), label.cuda()
            with torch.no_grad():
                cell_emb = model._get_cell_features(sample)
                text_emb = model._get_text_features(text)

            cell_embeds.append(cell_emb.cpu().numpy())
            text_embeds.append(text_emb.cpu().numpy())
            
        cell_embeds = np.concatenate(cell_embeds, axis=0).reshape(-1,args.projection_dim)
        cell_embeds = cell_embeds / np.linalg.norm(
            cell_embeds, axis=1, keepdims=True
        )
        text_embeds = np.concatenate(text_embeds, axis=0).reshape(-1,args.projection_dim)
        text_embeds = text_embeds / np.linalg.norm(
            text_embeds, axis=1, keepdims=True
        )
        
        adata =sc.read('dataset/sc-data/data/BMMC_multiomics.h5ad')
        
        adata.obsm['X_scBIND_zero_shot2'] = cell_embeds
        adata.obsm['Text_scBIND2'] = text_embeds
        adata.write('dataset/sc-data/data/BMMC_multiomics.h5ad')
        
        results = scib.metrics.metrics(
            adata,
            adata_int=adata,
            batch_key="batch_id",
            label_key="celltype",
            embed='X_scBIND_zero_shot2',
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,  # use the clustering, bias to the best matching
            ari_=True,  # use the clustering, bias to the best matching
            cell_cycle_=False,
            kBET_=False,  # kBET return nan sometimes, need to examine
            ilisi_=False,
            clisi_=False,
        )
        print(results)