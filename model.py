import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

kwrgs = {
    'n_fft': 512,
    'win_length': 320,
    'hop_length': 161,
    'n_mels': 80,
}

class SyncNetModel(nn.Module):
    def __init__(self, nOut = 1024):
        super(SyncNetModel, self).__init__();

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,2), padding=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(1,2)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(192, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(2,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(5,4), stride=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ); 
        
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=(1,5,5), stride=(1,2,2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0, 1, 1)),

            nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 512, kernel_size=(1,5,5), stride=(1,1,1), padding=(0, 1, 2)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0, 1, 1))
        )

        self.netfcaud = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, nOut, kernel_size=(1,1)),
        );

        self.netfclip = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, nOut, kernel_size=(1, 1, 1)),
        )
        self.mfcc_layer = MFCC(sample_rate=100, n_mfcc=80, melkwargs=kwrgs)
    def forward_aud(self, audios):
        
        audio = self.mfcc_layer(audios).transpose(2, 1).detach()
        audio = audio.unsqueeze(1)
        x = self.netcnnaud(audio)
        out = self.netfcaud(x)
        
        return out

    def forward_vid(self, frame, return_feat=False):

        x = self.netcnnlip(frame)
        out  = self.netfclip(x)
        if return_feat:
            return out, x
        else:
            return out