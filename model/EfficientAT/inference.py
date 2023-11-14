import sys
sys.path.insert(0, '.')

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

import pdb

parser = argparse.ArgumentParser(description='Example of parser. ')
# model name decides, which pre-trained model is loaded
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--audio_path', type=str, required=False, default="resources/metro_station-paris.wav")

# preprocessing
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--n_mels', type=int, default=128)

# ---------------------------------------- Choose model ----------------------------------------
parser.add_argument('--model_name', type=str, default='mn40_as_ext')

# overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
# We don't use model from model_name if ensemble argument is not empty

# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_ext"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_no_im_pre"])
parser.add_argument('--ensemble', nargs='+', default=["mn40_as", "mn40_as_ext"])
# ----------------------------------------------------------------------------------------------

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

def inference(audio_path):
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    result = {}
    output = {}

    # ========= Sound to detect =========
    #   10 sounds to detect
    # - Car horn
    # - Infant Crying
    # - Glass
    # - Screaming
    # - Explosion 
    # - Dog bark
    # - Siren
    # - Drill + Tools
    # - Bicycle Bell
    # - Fire Alarm
    # ========= Sound to detect =========

    # Car horn sound
    result.update({"Vehicle horn": preds[308]}) # 0.4
    result.update({"Train horn":preds[331]}) # 0.1
    result.update({"Foghorn":preds[401]}) # 0.1
    result.update({"Air horn, truck horn": preds[318]}) # 0.4
    output.update({"Car horn": preds[308]*0.4 + preds[331]*0.1 + preds[401]*0.1 + preds[318]*0.4})

    # Infant Crying sound
    result.update({"Baby Crying": preds[23]}) # 0.6
    result.update({"Baby Laughter": preds[17]}) # 0.2
    result.update({"Giggle": preds[18]}) # 0.5
    result.update({"Whimper": preds[24]}) # 0.5
    result.update({"Child speech, kid speaking": preds[3]}) # 0.5
    result.update({"Crying, sobbing": preds[22]}) # 0.5
    output.update({"Infant Crying": preds[23]*0.6 + preds[17]*0.2 + preds[18]*0.05 + preds[24]*0.05 + preds[3]*0.05 + preds[22]*0.05})

    # Glass
    result.update({"Glass": preds[441]}) # 0.45
    result.update({"Chink, clink": preds[442]}) # 0.45
    result.update({"Crack": preds[440]}) # 0.1
    output.update({"Glass": preds[441]*0.45 + preds[442]*0.45 + preds[440]*0.1})

    # TODO: Screaming
    result.update({"Screaming": preds[14]}) # ???
    output.update({"Screaming": preds[14]*5.00})
    
    # Explosion
    result.update({"Explosion": preds[426]}) # 0.30
    result.update({"Artillery fire"}: preds[430]) # 0.60
    result.update("Gunshot": preds[427]) # 0.10
    output.update({"Explosion": preds[426]*0.30 + preds[430]*0.60 + preds[427]*0.1})

    # Dog bark
    result.update({"Animal": preds[72]}) # 0.3
    result.update({"Dog": preds[74]}) # 0.3
    result.update({"Domestic animals, pets": preds[73]}) # 0.3
    result.update({"Bow-wow": preds[78]}) # 0.1
    output.update({"Dog bark": preds[72]*0.30 + preds[74]*0.30 + preds[73]*0.30 + preds[78]*0.10})

    # Siren
    result.update({"Police car(siren)": preds[323]}) # 0.33
    result.update({"Ambulance(siren)": preds[324]}) # 0.33
    result.update({"Fire truck(siren)": preds[325]}) # 0.33
    # output.update({"Car siren": preds[323]*0.33 + preds[324]*0.33 + preds[325]*0.33})

    # Drill + Tools
    result.update({"Drill": preds[425]}) # 0.33
    result.update({"Tools": preds[418]}) # 0.33
    result.update({"Power tool": preds[424]}) # 0.33
    output.update({"Tools": preds[425]*0.33 + preds[418]*0.33 + preds[424]*0.33})

    # Bicycle bell 
    result.update({"Bicycle": preds[341]}) # 0.1
    result.update({"Bicycle bell": preds[203]}) # 0.9
    output.update({"Bicycle bell": preds[341]*0.1 + preds[203]*0.9})

    # Fire Alarm
    result.update({"Fire alarm": preds[400]}) # 0.35
    result.update({"Smoke detector, smoke alarm":preds[399]}) # 0.35
    result.update({"Siren":preds[396]}) # 0.2
    result.update({"Buzzer":preds[398]}) # 0.1
    output.update({"Fire alarm": (preds[400]*0.35 + preds[399]*0.35 + preds[396]*0.2 + preds[398]*0.1)})

    # Gunshot: Not to be used
    # result.update({"Gunshot": preds[427]}) # 0.5
    # result.update({"Machine Gun": preds[428]}) # 0.4
    # result.update({"Fusillade": preds[429]}) # 0.1
    # output.update({"Gunshot": preds[427]*0.5 + preds[428]*0.4 + preds[429]*0.1})

    # Print audio tagging top probabilities
    # print("************* Acoustic Event Detected: *****************")
    # for k in range(10):
    #     print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
    #         preds[sorted_indexes[k]]))
    # print("********************************************************")
    
    return output

# comment this to not show test inference for the first time
print(inference(audio_path))

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