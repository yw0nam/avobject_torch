import numpy as np
from utils import *
from torch.utils import data
import torch
class Datagen(data.Dataset):
    
    def __init__(self, data_txt,resize=(270, 480), 
                 fps=25, sample_rate=16000, window_size=50, random_sample=False):

        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.random_sample = random_sample
        self.data_list = []
        cnt = 0
        with open(data_txt) as listfile:
            for line in listfile:
                data = line.split()
                if abs(int(data[3])) > self.window_size+2:
                    self.data_list.append(data)
                else:
                    cnt += 1
        
        print('Read %d data'%len(self.data_list), 'Skiped data %d'%(cnt))
        self.data_length = len(self.data_list)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_list)
            
    def getdata(self, path):
        'Generates one sample of data'
        
        video = load_video(path[0], resize=self.resize)
        audio = load_wav(path[1]).astype('float32')
        
        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio, video = trunkate_audio_and_video(video, audio, aud_fact)
        assert aud_fact * video.shape[0] == audio.shape[0]

        audio = np.array(audio)
        
        end_frame_idx= video.shape[0] - (self.window_size)
        
        start = np.random.randint(0, end_frame_idx)
        frame = video[start:start+self.window_size]
        audio = audio[start*640:(start+self.window_size)*640]

        frame = frame.transpose([3, 0, 1, 2]) # t c h w -> c t h w
        audio = torch.FloatTensor(audio)
        out_dict = {
            'video': frame,
            'audio': audio,
        }
        return out_dict
    
    def getdata_random_audio(self, rand_path):
        'Generates one sample of data'
        
        audio = load_wav(rand_path[1]).astype('float32')
        
        aud_fact = int(np.round(self.sample_rate / self.fps))
        aud_cutoff = int(audio.shape[0] / aud_fact)
        aud_feats = audio[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact
        aud_feats = aud_feats[:aud_in_frames * aud_fact]

        audio = np.array(aud_feats)
        aud_end_frame_idx= audio.shape[0] // aud_fact - (self.window_size)
        
        start = np.random.randint(0, aud_end_frame_idx)
        audio = audio[start*640:(start+self.window_size)*640]
        audio = torch.FloatTensor(audio)
        return audio
    

    def __getitem__(self, index):
        out_dict = self.getdata(self.data_list[index])
        if self.random_sample:
            rand_idx = np.random.randint(0, self.data_length)
            out_dict['false_audio'] = self.getdata_random_audio(self.data_list[rand_idx])
        return out_dict