import logging
from pathlib import Path
from typing import List, Tuple  

import torch
import torchaudio
# from torio.io import StreamingMediaDecoder
from torch.utils.data import Dataset
# import numpy as np
# import pyloudnorm as pyln   

log = logging.getLogger()

def load_waveform(
    path: Path, 
    target_sr: int = 48000, 
    duration: float = 8.0, 
    mono: bool = False
) -> Tuple[torch.Tensor,int]:
    """
    Load audio track from `path`.  If it's a video container, use MoviePy;
    otherwise torchaudio.load.
    Returns (waveform, sample_rate).
    """
    # ext = path.suffix.lower()
    max_samples = int(target_sr * duration)
    # if ext in VIDEO_EXTS:
    #     dec = StreamingMediaDecoder(path)
    #     dec.add_basic_audio_stream(
    #         frames_per_chunk=max_samples, 
    #         sample_rate=target_sr,
    #         num_channels=2,
    #     )
    #     dec.fill_buffer()
    #     audio_chunk = dec.pop_chunks()[0]
    #     if audio_chunk.shape[0] < max_samples:
    #         pad = max_samples - audio_chunk.shape[0]
    #         arr = torch.nn.functional.pad(audio_chunk, (0, 0, 0, pad), mode="constant", value=0)
    #     else:
    #         arr = audio_chunk[:max_samples]
    #     waveform = arr.T.float()
    # else:
    #     # pure audio file
    waveform, sr_orig = torchaudio.load(str(path))  # (channels, N)
    if mono:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # resample if needed
    if sr_orig != target_sr:
        # kaiser_best window
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr_orig, new_freq=target_sr,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    # clip or pad
    if waveform.size(1) < max_samples:
        pad = max_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :max_samples]

    sr = target_sr
        
    return waveform, sr


class AudioDataset(Dataset):
    def __init__(
        self,
        datalist: List[Path],
        audio_length: float = 8.0,
        sr: int = 48000,
        mono: bool = False,
        limit_num: int = None,
    ):
        self.datalist = datalist[:limit_num] if limit_num else datalist
        self.sr = sr
        self.audio_length = audio_length
        self.mono = mono
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        filename = self.datalist[idx]
        # load and clip to duration
        waveform, sr = load_waveform(filename, target_sr=self.sr, duration=self.audio_length, mono=self.mono)
        # return stereo Tensor and sample key
        return waveform.float(), filename.stem
    
    def __len__(self):
        return len(self.datalist)


# VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


# def int16_to_float32(x):
#     """
#     Convert a NumPy array of int16 values to float32.
#     Parameters:
#         x (numpy.ndarray): A NumPy array of int16 values.
#     Returns:
#         numpy.ndarray: A NumPy array of float32 values, scaled from the int16 input.
#     """

#     return (x / 32767.0).to(torch.float32)

# def float32_to_int16(x):
#     """
#     Converts a NumPy array of float32 values to int16 values.
#     This function clips the input array values to the range [-1.0, 1.0] and then scales them to the range of int16 
#     (-32768 to 32767).
#     Parameters:
#         x (numpy.ndarray): A NumPy array of float32 values to be converted.
#     Returns:
#         numpy.ndarray: A NumPy array of int16 values.
#     """

#     x = torch.clip(x, min=-1., max=1.)
#     return (x * 32767.).to(torch.int16)
