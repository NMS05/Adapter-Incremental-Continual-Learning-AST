import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

"""
ESC50 dataset
"""
class ESC50(Dataset):
    def __init__(self, annotations_file, audio_dir, spec_mean=-9.642373, spec_std=27.26192):

        # filename in column 1 and labels in column 3
        self.annos = pd.read_csv(annotations_file) 
        # all files in '.wav' format
        self.audio_dir = audio_dir

        # Specrogram Parameters
        self.sampling_frequency = 16000
        self.mel = T.MelSpectrogram(
                      sample_rate=16000,
                      n_fft=400,
                      win_length=400,
                      hop_length=160,
                      n_mels=128,
                  )
                  
        self.a2d = T.AmplitudeToDB()
        # mean and std already calculated
        self.spec_mean = spec_mean
        self.spec_std = spec_std


    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        audio_file_name = self.annos.iloc[idx, 1]
        audio_path = os.path.join(self.audio_dir, audio_file_name)

        # load audio file and resample if required
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)
        
        # mono waveform
        waveform = waveform[0]
        
        # normalize raw waveform
        waveform = (waveform - torch.mean(waveform)) / torch.std(waveform)
        # generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # normalize spectrogram.
        spectrogram = (spectrogram - self.spec_mean) / self.spec_std
        
        spectrogram = spectrogram.unsqueeze(0)

        # labels for ESC 50
        label = int(self.annos.iloc[idx, 3])

        return spectrogram, label
    
"""
Google SpeechCommands-V2 dataset
"""

class SpeechCommands(Dataset):
    def __init__(self, split, spec_mean=-26.961325, spec_std=49.383118):

        self.sampling_frequency = 16000
        self.mel = T.MelSpectrogram(
                      sample_rate=16000,
                      n_fft=400,
                      win_length=400,
                      hop_length=160,
                      n_mels=128,
                  )
        
        self.a2d = T.AmplitudeToDB()

        self.spec_mean = spec_mean
        self.spec_std = spec_std

        if split=='train':
            self.data = torchaudio.datasets.SPEECHCOMMANDS(root="data/SpeechCommands/",download=True,subset="training")
        elif split=='test':
            self.data = torchaudio.datasets.SPEECHCOMMANDS(root="data/SpeechCommands/",download=True,subset="testing")

        self.class_names = open("data/SpeechCommands/SpeechCommands_class_names.txt").read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        waveform, sample_rate, str_label, _, _ = self.data[idx]
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)
        
        # normalize raw waveform
        waveform = (waveform - torch.mean(waveform)) / torch.std(waveform)
        # generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # normalize spectrogram
        spectrogram = (spectrogram - self.spec_mean) / self.spec_std

        # labels
        label = int(self.class_names.index(str_label))

        return spectrogram.squeeze(0), label


"""
AVE Dataset (data preprocessed for classification - each audio clip belongs to exactly one class)
"""
class AVE(Dataset):
    def __init__(self, annotations_file, audio_dir, spec_mean=2.5812705, spec_std=24.051544):
        super(AVE, self).__init__()

        self.annos = pd.read_csv(annotations_file, header=None)  # columns as [file_name, label]
        self.audio_dir = audio_dir  # all files in '.wav' format

        self.sampling_frequency = 16000
        self.mel = T.MelSpectrogram(
                      sample_rate=16000,
                      n_fft=400,
                      win_length=400,
                      hop_length=160,
                      n_mels=128,
                  )
                  
        self.a2d = T.AmplitudeToDB()
        self.spec_mean = spec_mean
        self.spec_std = spec_std

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        # select one clip
        clip_name = self.annos.iloc[idx, 0]

        # load the audio file with torch audio
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        # use mono audio instead os stereo audio (use left by default)
        waveform = waveform[0]

        # resample
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)

        # normalize raw waveform
        waveform = (waveform - torch.mean(waveform)) / torch.std(waveform)
        # generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # normalize spectrogram.
        spectrogram = (spectrogram - self.spec_mean) / self.spec_std
        spectrogram = spectrogram.type(torch.float32)

        # make sure that shape is uniform
        if spectrogram.shape[1] < 1006: spectrogram = torch.cat((spectrogram,torch.zeros(128,1006-spectrogram.shape[1])),dim=-1)

        # assign integer to labels
        label = int(self.annos.iloc[idx, 1])
        
        return spectrogram, label  