import argparse
import torch
import glob
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

import pandas as pd
import pdb

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels
from finetune import Full_network
import pdb

parser = argparse.ArgumentParser(description='Example of parser. ')
# model name decides, which pre-trained model is loaded
parser.add_argument('--model_name', type=str, default='mn40_as_ext')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--audio_path', type=str, required=False, default="resources/temp.wav") # test audio_path

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
parser.add_argument('--ensemble', nargs='+', default=["mn40_as", "mn40_as_ext"])

"""
Running Inference on an audio clip.
"""
args = parser.parse_args()

# get parser arguments
model_name = args.model_name
device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
audio_path = args.audio_path
sample_rate = args.sample_rate
window_size = args.window_size
hop_size = args.hop_size
n_mels = args.n_mels

# load pre-trained model
if len(args.ensemble) > 0:
    model = get_ensemble_model(args.ensemble)
else:
    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)

model.to(device)
model.eval()

# model to preprocess waveform into mel spectrograms
mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
mel.to(device)
mel.eval()

PATH = "./torch_checkpoints/finetune_ver_02.pt" # 200 epochs
network = Full_network(args)
state_dict = torch.load(PATH, map_location=torch.device("cpu"))

# sellect the final layers
for key, value in state_dict.copy().items():
    if "fc.fc" not in key:
        state_dict.pop(key)

# load final layer parameters
network.load_state_dict(state_dict, strict=False)


def inference(network, audio_path):
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze()

    result = network.fc(preds)

    result_dict = {}
    result_dict.update({"Vehicle_horn": result[0].item()})
    result_dict.update({"Baby_cry": result[1].item()})
    result_dict.update({"Fire_Alarm": result[2].item()})
    result_dict.update({"Gun_fire": result[3].item()})
    result_dict.update({"Glass": result[4].item()})
    return result_dict

# comment this to not show test inference for the first time
print(inference(network, audio_path))

def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")

# if __name__ == '__main__':
