import librosa
import numpy as np
from scipy.io import wavfile
import av
import python_speech_features as psf
import torchvision
from PIL import Image
import torch

def load_wav(path, fr=0, to=10000, sample_rate=16000):
    """Loads Audio wav from path at time indices given by fr, to (seconds)"""

    _, wav = wavfile.read(path)
    fr_aud = int(np.round(fr * sample_rate))
    to_aud = int(np.round((to) * sample_rate))

    wav = wav[fr_aud:to_aud]

    return wav

def load_video(vid_path, resize=None):

    container = av.open(vid_path)

    ims = [frame.to_image() for frame in container.decode(video=0)]
    frames = np.array([np.array(im) for im in ims])
    if resize:
        ims = [Image.fromarray(frm) for frm in frames]
        ims = [
            torchvision.transforms.functional.resize(im,
                                                     resize)
            for im in ims
        ]
        frames = np.array([np.array(im) for im in ims])

    return frames.astype('float32')

def trunkate_audio_and_video(video, aud_feats, aud_fact):

    aud_in_frames = aud_feats.shape[0] // aud_fact

    # make audio exactly devisible by video frames
    aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

    aud_feats = aud_feats[:aud_cutoff * aud_fact]
    aud_in_frames = aud_feats.shape[0] // aud_fact

    min_len = min(aud_in_frames, video.shape[0])

    # --- trunkate all to min
    video = video[:min_len]
    aud_feats = aud_feats[:min_len * aud_fact]
    if not aud_feats.shape[0] // aud_fact == video.shape[0]:
        ipdb.set_trace(context=20)

    return aud_feats, video

def logsoftmax_2d(logits):
    # Log softmax on last 2 dims because torch won't allow multiple dims
    orig_shape = logits.shape
    logprobs = torch.nn.LogSoftmax(dim=-1)(
        logits.reshape(list(logits.shape[:-2]) + [-1])).reshape(orig_shape)
    return logprobs

