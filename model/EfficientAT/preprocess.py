import os
import glob
import torch
import numpy as np
import librosa
import argparse

from torch.utils.data import Dataset, DataLoader
import random
import pdb

# from datasets.audioset import get_ft_weighted_sampler
# from helpers.init import worker_init_fn


# ------------------------------- config -------------------------------
parser = argparse.ArgumentParser(description='Example of parser. ')

# general
parser.add_argument('--experiment_name', type=str, default="AudioSet")
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)

# evaluation
# overwrite 'pretrained_name' by 'ensemble' to evaluate an ensemble
parser.add_argument('--ensemble', nargs='+', default=[])
parser.add_argument('--model_name', type=str, default="mn10_as")

# training
parser.add_argument('--pretrained_name', type=str, default=None)
parser.add_argument('--model_width', type=float, default=1.0)
parser.add_argument('--head_type', type=str, default="mlp")
parser.add_argument('--se_dims', type=str, default="c")
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--mixup_alpha', type=float, default=0.3)
parser.add_argument('--epoch_len', type=int, default=100000)
parser.add_argument('--roll', action='store_true', default=False)
parser.add_argument('--wavmix', action='store_true', default=False)
parser.add_argument('--gain_augment', type=int, default=0)
parser.add_argument('--weight_decay', type=int, default=0.0001)
# lr schedule
parser.add_argument('--max_lr', type=float, default=0.001)
parser.add_argument('--warm_up_len', type=int, default=8)
parser.add_argument('--ramp_down_start', type=int, default=80)
parser.add_argument('--ramp_down_len', type=int, default=95)
parser.add_argument('--last_lr_value', type=float, default=0.01)

# knowledge distillation
parser.add_argument('--teacher_preds', type=str,
                    default=os.path.join("resources", "passt_enemble_logits_mAP_495.npy"))
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--kd_lambda', type=float, default=0.1)

# preprocessing
parser.add_argument('--resample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)
parser.add_argument('--fmin', type=int, default=0)
parser.add_argument('--fmax', type=int, default=None)

args = parser.parse_args()
# ------------------------------- config -------------------------------



# set device
device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

# helper function for data prerpocessing
# from waveform to 32000Hz file
def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform

# make custom dataset for audioset
class MyCustomDataset(Dataset):
    def __init__(self, root_path, sample_rate=32000, resample_rate=32000, classes_num=527, 
                 clip_length=10, in_mem=False, gain_augment=0):
        """
        Args:
            root_path(string): path to audioset data folder

        """
        self.sample_rate = 32000 # ****** hardcoded 32000Hz ******

        folder_list = ["Vehicle_horn", "Baby_cry", "Fire_Alarm", "Gun_fire", "Glass"]

        # Get image list
        self.audio_list = []
        self.num_class = []
        for folder in folder_list:
            folder_path = os.path.join(root_path, folder)
            audio_files = glob.glob(folder_path + '/*')
            self.num_class.append(len(audio_files))
            self.audio_list += audio_files

        self.labels = np.array([0]*self.num_class[0] + [1]*self.num_class[1] + [2]*self.num_class[2] + [3]*self.num_class[3] + [4]*self.num_class[4])
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment

        # Calculate len
        self.data_len = len(self.audio_list)        
        
    def __getitem__(self, index):
        # Get audio name from audio path
        single_audio_path = self.audio_list[index]
        # change audio_path to waveform
        (waveform, sr) = librosa.core.load(single_audio_path, sr=self.sample_rate, mono=True)
        waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = self.resample(waveform)
        waveform = torch.from_numpy(waveform[None, :])
        waveform = waveform.squeeze()

        # Get label
        label = self.labels[index]

        return (waveform, label)

    def __len__(self):
        return self.data_len # of how many examples(images?) you have
    
    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.resample_rate == 32000:
            return waveform
        elif self.resample_rate == 16000:
            return waveform[0:: 2]
        elif self.resample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')
    
if __name__ =="__main__":
    # ------------------------------ Debugging ------------------------------
    dataset = MyCustomDataset(root_path="../audioset-processing/output/dataset")

    myCustomDataset = MyCustomDataset("../audioset-processing/output/dataset")

    data_len = len(myCustomDataset)
    train_len = data_len//5
    val_len =  data_len - train_len

    # split train_set: val_set into 0.8:0.2
    train_set, val_set = torch.utils.data.random_split(myCustomDataset, [train_len, val_len])

    train_loader = DataLoader(train_set,
                    shuffle=False,
                    # worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)

    val_loader = DataLoader(val_set,
                    shuffle=False,
                    # worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)

    # for i_batch, sample_batch in enumerate(train_loader):
    #     pass
    #     pdb.set_trace()

    for i_batch, sample_batch in enumerate(val_loader):
        pass
        pdb.set_trace()
