# Databricks notebook source
!git clone -b autoddc https://github.com/kashishkumarexxon/fasterrcnn-pytorch-training-pipeline.git

# COMMAND ----------

# MAGIC %cd fasterrcnn-pytorch-training-pipeline

# COMMAND ----------

!ls

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install -r requirements.txt

# COMMAND ----------

import wandb
wandb.login(key='5b77c3d8af0481c8ddb39fd537e228ef96d65281')

# COMMAND ----------

import torch
torch.__version__

# COMMAND ----------

!pip list

# COMMAND ----------

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

# COMMAND ----------

# MAGIC %sh
# MAGIC export CUDA_VISIBLE_DEVICES=""

# COMMAND ----------

# MAGIC %sh
# MAGIC python train.py --data data_configs/autoddc.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name autoddc --batch 16 --device cpu

# COMMAND ----------

import matplotlib.pyplot as plt
import glob as glob

# COMMAND ----------

results_dir_path = '/content/fastercnn-pytorch-training-pipeline/outputs/training/custom_training'
valid_images = glob.glob(f"{results_dir_path}/*.jpg")

for i in range(2):
    plt.figure(figsize=(10, 7))
    image = plt.imread(valid_images[i])
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# COMMAND ----------

# No verbose mAP.
!python eval.py --weights outputs/training/autoddc/best_model.pth --config data_configs/autoddc.yaml --model fasterrcnn_resnet50_fpn_v2

# COMMAND ----------

# Verbose mAP.
!python eval.py --weights outputs/training/autoddc/best_model.pth --config data_configs/autoddc.yaml --model fasterrcnn_resnet50_fpn_v2 --verbose

# COMMAND ----------


