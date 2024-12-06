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

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)
BASE_PATH = '.'
_MAX_INT = np.iinfo(np.uint16).max

# ======load Prithvi model and weights========
def load_model():
    weights_path = "./prithvi/Prithvi_100M.pt"
    checkpoint = torch.load(weights_path, map_location=torch.device('cuda:0'))

    # read model config
    model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]

    # let us use only 1 frame for now (the model was trained on 3 frames)
    model_args["num_frames"] = 1

    model_args['img_size']= 128

    # instantiate model
    prithvi = MaskedAutoencoderViT(**model_args)

    _ = prithvi.load_state_dict(checkpoint, strict=False)

    return prithvi

# =========Load data==========
def load_data() -> tuple:
    '''
    Load data for fine tuning

    Return:
    ----------------------
    train_image_tensor: Tensor object - preprocessed satellite image has a dimension of (training_size, 1, 128,128,6)
    train_timeseries_tensor: Tensor object - preprocessed time series of precipitation has a dimension of (training_size, time_steps)
    train_label_tensor: Tensor object - time series of flood or non-flood (binary), has a dimension of (training_size, time_steps)
    valid_image_tensor: Tensor object - preprocessed satellite image has a dimension of (valid_size, 1, 128,128,6)
    valid_timeseries_tensor: Tensor object - preprocessed time series of precipitation has a dimension of (valid_size, time_steps)
    valid_label_tensor: Tensor object - time series of flood or non-flood (binary), has a dimension of (valid_size, time_steps)
    '''
    data = pd.read_csv(os.path.join(BASE_PATH, 'Train.csv'))
    data_test = pd.read_csv(os.path.join(BASE_PATH, 'Test.csv'))

    data['event_id'] = data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    data['event_idx'] = data.groupby('event_id', sort=False).ngroup()
    data_test['event_id'] = data_test['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    data_test['event_idx'] = data_test.groupby('event_id', sort=False).ngroup()

    data['event_t'] = data.groupby('event_id').cumcount()
    data_test['event_t'] = data_test.groupby('event_id').cumcount()
    data['logRain']= np.log(data.precipitation+1e-6) # add a epsilo to avoid zeros
    data['normRain']= (data.logRain - data.logRain.mean()) / data.logRain.std()

    images_path = os.path.join(BASE_PATH, 'composite_images.npz')
    images = np.load(images_path)
    print(images)
    print('The folder contains', len(images), 'images, both for train and test.')
    print('There are', len(data['event_id'].unique()), 'train event ids and', len(data_test['event_id'].unique()), 'test event ids.')

    BAND_NAMES =  ('B2', 'B3', 'B4', 'B8', 'B11', 'slope')
    # Image shape
    H, W, NUM_CHANNELS = IMG_DIM = (128, 128, len(BAND_NAMES))

    sample_image = next(iter(images.values()))
    assert sample_image.shape == IMG_DIM
    assert sample_image.dtype == np.uint16
    
    rng = np.random.default_rng(seed=0xf100d)

    event_ids = data['event_id'].unique()
    new_split = pd.Series(
        data=np.random.choice(['train', 'valid'], size=len(event_ids), p=[0.8, 0.2]),
        index=event_ids,
        name='split',
    )
    data_new = data.join(new_split, on='event_id')

    train_df = data_new[(data_new['split'] == 'train')]
    train_timeseries = train_df.pivot(index='event_id', columns='event_t', values='normRain').to_numpy()
    train_labels = train_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()

    valid_df = data_new[data_new['split'] == 'valid']
    valid_timeseries = valid_df.pivot(index='event_id', columns='event_t', values='normRain').to_numpy()
    valid_labels = valid_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()

    event_splits = data_new.groupby('event_id')['split'].first()

    #convert to tensor
    b,t= train_timeseries.shape
    train_timeseries_tensor= torch.from_numpy(train_timeseries).to(torch.float32).view(b, 1, t)
    train_labels_tensor= torch.from_numpy(train_labels).to(torch.float32)
    b,t= test_timeseries.shape
    valid_timeseries_tensor = torch.from_numpy(valid_timeseries).to(torch.float32).view(b, 1, t)
    valid_labels_tensor= torch.from_numpy(valid_tensor).to(torch.float32).view(b, 1, t)
    
    train_images = []
    valid_images = []

    for event_id in event_splits.index:
        img = preprocess_image(images[event_id])
        if event_splits[event_id] == 'train':
            train_images.append(img)
        else:
            valid_images.append(img)

    for event_id in data_test['event_id'].unique():
        img = preprocess_image(images[event_id])

    train_images = np.stack(train_images, axis=0)
    valid_images = np.stack(valid_images, axis=0)

    train_images_tensor= torch.from_numpy(train_images).permute([0, 3, 1, 2]).to(torch.float32).reshape(train_images.shape[0], 6, 1, 128, 128)
    valid_images_tensor= torch.from_numpy(valid_images).permute([0, 3, 1, 2]).to(torch.float32).reshape(valid_images.shape[0], 6, 1, 128, 128)

    return (train_images_tensor, train_timeseries_tensor, train_labels_tensor, valid_images_tensor, valid_timeseries_tensor, valid_labels_tensor)
    

def decode_slope(x: np.ndarray) -> np.ndarray:
  # Convert 16-bit discretized slope to float32 radians
  return (x / _MAX_INT * (math.pi / 2.0)).astype(np.float32)

def normalize(x: np.ndarray, mean: int, std: int) -> np.ndarray:
  return (x - mean) / std

rough_S2_normalize = partial(normalize, mean=1250, std=500)

def preprocess_image(x: np.ndarray) -> np.ndarray:
  return np.concatenate([
      rough_S2_normalize(x[..., :-1].astype(np.float32)),
      decode_slope(x[..., -1:]),
  ], axis=-1, dtype=np.float32)


class LoadData(Dataset):
    def __init__(self, train_images_tensor, train_timeseries_tensor, train_labels_tensor):
        """
        Args:
            data (torch.Tensor): Input images, shape (num_samples, channels, height, width)
            labels (torch.Tensor): Labels, shape (num_samples, num_classes) or (num_samples,)
            transform (callable, optional): Optional transform to apply to data
        """
        super(LoadData, self).__init__()
        self.data = train_images_tensor
        self.labels = train_labels_tensor
        self.ts= train_timeseries_tensor.squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        t = self.ts[idx]
        y = self.labels[idx]

        return x, t, y

#Combine pretrained model and fine-tuned model
class FloodViT(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(FloodViT, self).__init__()
        
        # Copy components from the pretrained model
        self.pretrained_model= pretrained_model
        # features, _, _ = model.forward_encoder(normalized, mask_ratio=0)
        
        # Define a global pooling layer and fully connected layer for classification


        self.fcl1= nn.Linear(49152, 730)
        self.fcl2= nn.Linear(730*2, 730)
        self.relu= nn.ReLU()
        self.batchnorm= nn.BatchNorm1d(730)

    def forward(self, x, t):
        # Preprocessing and patch embedding
        x,_,_= self.pretrained_model.forward_encoder(x, mask_ratio=0)
        b, token, e= x.shape
        b, it= t.squeeze().shape
        # Reshape to spatial dimensions: (batch_size, channels, height, width)
        x = x[:, 1:, :].view(b, -1)
        x= self.fcl1(x)
        x= self.relu(x)
        # t= self.batchnorm(t)
        x= torch.concat([x, t], dim=-1)
        x= self.fcl2(x)
        x= self.relu(x)
        
        return x


if __name__=='__main__':
    # load data
    train_image_tensor, 
    train_timeseries_tensor, 
    train_label_tensor, 
    valid_image_tensor, 
    valid_timeseries_tensor, 
    valid_label_tensor= load_data()

    # load model
    pretrained_model= load_model()
    model= FloodViT(pretrained_model, 1)
    # Freeze pretrained_model parameters
    for name, param in new_model.named_parameters():
        if name.startswith("pretrained_model") or name.startswith('decoder'):
            param.requires_grad = False
    
    # define optimizer and loss function
    pos_weight = torch.tensor([5.0]).to(device)
    criterion= nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer= optim.AdamW(new_model.parameters(), lr=1e-3)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model= model.to(device)
    dataset = LoadData(transform=True)

    # Create DataLoader for batch processing
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 100  # Number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for inputs, t, label in train_loader:
            inputs, t, label = inputs.to(device), t.to(device), label.to(device)  # Move data to GPU/CPU
            
            pred= model(inputs, t)

            loss= criterion(pred, label)


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Pred: {pred.sum()}, Label: {label.sum()}")