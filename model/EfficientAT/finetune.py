import argparse
import torch
import torch.nn.functional as F
import glob
import librosa
import numpy as np
from torch import nn
from torch import autocast
from contextlib import nullcontext
import torchinfo
import torchsummary
import os

import pandas as pd
import pdb

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels
from preprocess import *

labels_map = {
    0: "Vehicle_horn",
    1: "Baby_cry",
    2: "Fire_Alarm",
    3: "Gun_fire",
    4: "Glass"
}

dataset_classes = 527
output_classes = 5

parser = argparse.ArgumentParser(description='Example of parser. ')

# general
parser.add_argument('--experiment_name', type=str, default="AudioSet")
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)

# model name decides, which pre-trained model is loaded
parser.add_argument('--model_name', type=str, default='mn40_as_ext')
parser.add_argument('--audio_path', type=str, required=False, default="resources/--2XRMjyizo_0.wav")

# training
parser.add_argument('--pretrained_name', type=str, default=None)
parser.add_argument('--model_width', type=float, default=1.0)
parser.add_argument('--head_type', type=str, default="mlp")
parser.add_argument('--se_dims', type=str, default="c")
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--mixup_alpha', type=float, default=0.3)
parser.add_argument('--epoch_len', type=int, default=100000)
parser.add_argument('--roll', action='store_true', default=False)
parser.add_argument('--wavmix', action='store_true', default=False)
parser.add_argument('--gain_augment', type=int, default=0)
parser.add_argument('--weight_decay', type=int, default=0.0001)

PATH = "./torch_checkpoints/finetune_ver_01.pt"

# lr schedule
parser.add_argument('--max_lr', type=float, default=0.0003)
parser.add_argument('--warm_up_len', type=int, default=8)
parser.add_argument('--ramp_down_start', type=int, default=80)
parser.add_argument('--ramp_down_len', type=int, default=95)
parser.add_argument('--last_lr_value', type=float, default=0.01)

# preprocessing
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--resample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)
parser.add_argument('--fmin', type=int, default=0)
parser.add_argument('--fmax', type=int, default=None)

# overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_ext"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_no_im_pre"])
parser.add_argument('--ensemble', nargs='+', default=["mn10_as", "mn40_as_ext"])

args = parser.parse_args()

model_name = args.model_name

device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
print(f"{device=}")

audio_path = args.audio_path
sample_rate = args.sample_rate
window_size = args.window_size
hop_size = args.hop_size
n_mels = args.n_mels

args = parser.parse_args()

# fully connected layer 527 -> 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define layers
        layers = [nn.Linear(dataset_classes, output_classes),
                  nn.Sigmoid()]
        # fully connected layer
        self.fc = nn.Sequential(*layers)
        
        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class Full_network(nn.Module):
    def __init__(self, args):
        super(Full_network, self).__init__()
        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=args.n_mels,
                            sr=args.resample_rate,
                            win_length=args.window_size,
                            hopsize=args.hop_size,
                            n_fft=args.n_fft,
                            freqm=args.freqm,
                            timem=args.timem,
                            fmin=args.fmin,
                            fmax=args.fmax
                            )
        
        self.mel.to(device)
        self.mel.train()

        # load pre-trained model
        if len(args.ensemble) > 0:
            self.model = get_ensemble_model(args.ensemble)
        else:
            self.model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        self.model.to(device)
        self.model.train()

        # **freeze** model parameters for finetuning
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = Net()
        self.fc.to(device)
        self.fc.train()

    def forward(self, waveform):
        waveform = waveform.squeeze()
        with autocast(device_type=device.type) if args.cuda else nullcontext():
            spec = self.mel(waveform)
            preds, features = self.model(spec.unsqueeze(1))
        
        preds=preds.float() # change the torch.float16 to torch.float32
        output = self.fc(preds)
        return output
    
###############################################################
# debugging network 
# temp = torch.randn(527, requires_grad=True)
# network = Net()
# print(network(temp).shape) # torch.size(4)
###############################################################
# network = Full_network(args)
# print(network(audio_path).shape)
# pdb.set_trace()
###############################################################

train_set = MyCustomDataset(root_path="../audioset-processing/output/train_dataset")

val_set = MyCustomDataset("../audioset-processing/output/val_dataset")

train_len = len(train_set)
val_len = len(val_set)
print(f"number of training dataset: {train_len}")
print(f"number of validation dataset: {val_len}")

train_loader = DataLoader(train_set,
                shuffle=True,
                # worker_init_fn=worker_init_fn,
                num_workers=args.num_workers,
                batch_size=args.batch_size)

val_loader = DataLoader(val_set,
                shuffle=False,
                # worker_init_fn=worker_init_fn,
                num_workers=args.num_workers,
                batch_size=args.batch_size)

network = Full_network(args)
PATH = "torch_checkpoints/finetune_ver_02.pt"
checkpoint = torch.load(PATH)

network.load_state_dict(checkpoint, )
for i_batch, sample_batch in enumerate(train_loader):
    waveform_batch = sample_batch[0].to(device)
    label_batch = sample_batch[1].to(device)
    output = network(waveform_batch)
    loss = output, label_batch
    break

for name, p in network.named_parameters():
    if p.requires_grad == True:
        print(f"parameter name: {name}")

optimizer = torch.optim.Adam(network.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(reduction="sum")

# # print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

def train(args):
    train_loss = []
    val_loss = []
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i_batch, sample_batch in enumerate(train_loader):
            x, y = sample_batch
            x = x.to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=5)
            output = network(x)
            loss = criterion(output.to(torch.float32), y_onehot.to(torch.float32))
            
            running_loss += loss.item()
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch+1)%10==0:
                print(f"loss: {loss.item():.4f}")

        with torch.no_grad():    
            train_loss = running_loss / train_len

            val_loss = 0.0
            for i_batch, sample_batch in enumerate(val_loader):
                x, y = sample_batch
                x = x.to(device)
                y = y.to(device)
                y_onehot = F.one_hot(y, num_classes=5)
                output = network(x)
                loss = criterion(output.to(torch.float32), y_onehot.to(torch.float32))
                val_loss += loss
            val_loss = val_loss / val_len
            print(f"epoch={epoch+1}, {train_loss=}, {val_loss=}")

train(args)
torch.save(network.state_dict(), PATH)