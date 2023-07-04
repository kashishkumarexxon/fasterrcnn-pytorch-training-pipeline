# Databricks notebook source
# MAGIC %sh
# MAGIC pip install -r requirements.txt

# COMMAND ----------

import wandb
wandb.login(key='5b77c3d8af0481c8ddb39fd537e228ef96d65281')

# COMMAND ----------

import torch
torch.__version__

# COMMAND ----------

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

# COMMAND ----------

# MAGIC %sh
# MAGIC python train.py --data data_configs/autoddc.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name autoddc --batch 16 

# COMMAND ----------


