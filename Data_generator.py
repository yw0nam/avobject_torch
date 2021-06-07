import numpy as np
from utils import *
from torch.utils import data
import torch
class Datagen(data.Dataset):
    
    def __init__(self, data_txt,resize=(270, 480), 
                 fps=25, sample_rate=16000, window_size=50, sample_random=False):

        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.sample_random = sample_random
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
            
    def getdata(self, path, rand_path=None):
        'Generates one sample of data'
        if rand_path:
            aud_path = rand_path[1]
            vid_path = path[0]
        else:
            aud_path = path[1]
            vid_path = path[0]
        video = load_video(vid_path, resize=self.resize)
        audio = load_wav(aud_path).astype('float32')
        
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
    

    def __getitem__(self, index):
        if not self.sample_random:
            out_dict = self.getdata(self.data_list[index])
        else:
            rand_idx = np.random.randint(0, self.data_length)
            out_dict = self.getdata(self.data_list[index], self.data_list[rand_idx])
        return out_dict