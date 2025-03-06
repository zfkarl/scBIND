# scBIND: A General Single-cell Data Analysis Framework\\ with LLM-based Bi-level Knowledge Databases

## Introduction
scBIND is a general framework for single-cell data analysis, which injects LLM priors via bi-level knowledge databases.

![image](https://github.com/zfkarl/scBIND/blob/master/imgs/scBIND.png)

## Getting Started
#### Requirements
- Python 3.10, PyTorch>=1.21.0,  numpy>=1.24.0, are required for the current codebase.

#### LLM Embeddings
##### 1. Cell-level Text Embeddings
We use the GPT-4o mini model to generate cell-level function descriptions and use the `text-embedding-3-large' model to extract embeddings. Download embeddings from https://drive.google.com/drive/folders/1aArcZjDckc7my9gPvVqN0h8X-7a0brLV.

##### 2. Feature-level Text Embeddings 
We use the GPT-3.5 model to generate feature-level function descriptions and use the `text-embedding-ada-002' model to extract embeddings. We also provide the preprocessed version in https://drive.google.com/drive/folders/1aArcZjDckc7my9gPvVqN0h8X-7a0brLV.

#### Datasets
##### CITE-seq and ASAP-seq Data 
Download dataset from https://github.com/SydneyBioX/scJoint/blob/main/data.zip.

#### Cell Type Annotation 
##### Pre-training on CITE-seq Data 
<pre>python train_cite.py --lr 1.5e-4 --batch_size 256 </pre> 

##### Fine-tuning on CITE-seq Data 
<pre>python train_cite.py --lr 1e-4 --batch_size 256 --checkpoint `your pre-trained checkpoint' </pre> 

##### Pre-training on ASAP-seq Data 
<pre>python train_asap.py --lr 1.5e-4 --batch_size 256 </pre> 

##### Fine-tuning on ASAP-seq Data 
<pre>python train_asap.py --lr 1e-4 --batch_size 256 --checkpoint `your pre-trained checkpoint' </pre> 

## Acknowledgement
Our codebase is built based on scCLIP, timm, transformers, and Pytorch Lightning. We thank the authors for the nicely organized code!
