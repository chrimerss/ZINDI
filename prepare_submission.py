'''
This file prepares submission for the competition

@ Zhi Li (12/08/2024)
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT
import pandas as pd
from functools import partial
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import torch.nn as nn
import torch.optim as optim
from fine_tune import FloodViT, load_data, load_model

BASE_PATH= '.'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, _, _, _, _, test_images_tensor, test_timeseries_tensor= load_data()
pretrained_model= load_model()
model = FloodViT(pretrained_model, 1).to(device)

# Load the checkpoint
checkpoint = torch.load("checkpoints/model_state.pth", map_location='cuda')
# Restore state
model.load_state_dict(checkpoint)

model.eval()  # Set the model to training mode

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

test_labels= model(test_images_tensor.to(device), test_timeseries_tensor.to(device).squeeze()).cpu().detach().numpy()

probs = sigmoid(test_labels)
probs.shape

sample_submission = pd.read_csv(BASE_PATH + '/SampleSubmission.csv')
sample_submission['label'] = probs.flatten()

sample_submission.to_csv('BenchmarkSubmission.csv', index = False)


